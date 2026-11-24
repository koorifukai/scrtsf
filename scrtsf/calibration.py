"""Numbered optical trials with explicit final reports and confirmation."""

import math
import os
import struct

from . import protocol as p
from .visual import CONTROL_PROFILE, PROFILE_ORDER, PROFILES

CAL_CONTROL = 11
CAL_PATTERN = 12
STAMP = struct.Struct("!IH")
FRAME_SECONDS = 0.7
INTERVALS = (FRAME_SECONDS, 0.35)
MIN_SUCCESS = 0.6
PATTERN_MESSAGE = b"ScreenTransfer calibration v1"
PATTERN_FRAMES = 3
PATTERN_PASS = 2
BLOCK_ENTRY_SIZE = p.BLOCK + STAMP.size
PATTERN_ORDER = tuple(sorted(PROFILE_ORDER, key=lambda profile: (
    -(PROFILES[profile].capacity // BLOCK_ENTRY_SIZE), PROFILES[profile].color)))
PATTERN_DURATION = len(PATTERN_ORDER) * PATTERN_FRAMES * (FRAME_SECONDS + 0.2)


class PatternSender:
    def __init__(self):
        session = os.urandom(8)
        self.frames = [
            (profile, p.packet(CAL_PATTERN, session, slot,
                               PATTERN_MESSAGE + os.urandom(PROFILES[profile].capacity - len(PATTERN_MESSAGE))))
            for slot, profile in enumerate(profile for profile in PATTERN_ORDER for _ in range(PATTERN_FRAMES))
        ]
        self.slot = 0

    def frame(self):
        profile, data = self.frames[self.slot]
        self.slot = (self.slot + 1) % len(self.frames)
        return profile, data


class PatternReceiver:
    def __init__(self):
        self.session = None
        self.seen = {profile: set() for profile in PATTERN_ORDER}
        self.saw_anchors = False

    def handle(self, decoded):
        profile, (kind, session, slot, payload) = decoded
        rank = slot // PATTERN_FRAMES
        if (kind != CAL_PATTERN or not 0 <= rank < len(PATTERN_ORDER)
                or profile != PATTERN_ORDER[rank] or len(payload) != PROFILES[profile].capacity
                or not payload.startswith(PATTERN_MESSAGE)):
            return
        if self.session is None:
            self.session = session
        if session == self.session:
            self.seen[profile].add(slot)

    def result(self, finished=False):
        profiles = PATTERN_ORDER if finished else PATTERN_ORDER[:1]
        for profile in profiles:
            count = len(self.seen[profile])
            if count >= PATTERN_PASS:
                return (f"Recommended: {PROFILES[profile].label} - {count}/{PATTERN_FRAMES} test frames decoded.\n"
                        "Select this profile on the sender for no-ACK transfer.")
        if finished:
            if any(self.seen.values()):
                return "Calibration pattern decoded, but no preset passed the 2-of-3 test."
            if self.saw_anchors:
                return "Screen anchors detected, but no calibration frame decoded."
            return "Not detected"
        return None


def useful_bytes(profile):
    return min(p.WINDOW, (PROFILES[profile].capacity - STAMP.size) // BLOCK_ENTRY_SIZE) * p.BLOCK


class Calibration:
    def __init__(self, profiles, seconds):
        self.candidates = [(profile, interval) for profile in profiles for interval in INTERVALS]
        self.seconds = seconds
        self.index = 0
        self.trial = int.from_bytes(os.urandom(4), "big")
        self.phase = "start"
        self.confirming = False
        self.results = []
        self.selected = None
        self.done = False
        self.confirmed_rate = 0.0
        self.interval = INTERVALS[0]
        self._begin()

    def _begin(self):
        self.trial = (self.trial + 1) & 0xFFFFFFFF
        self.profile, self.hold = self.candidates[self.index]
        self.frames = max(3, math.ceil(self.seconds / self.hold))
        self.sent = 0
        self.received = 0
        self.started = None
        self.duration = None
        self.phase = "start"

    def frame(self, session, seq, now):
        self.interval = INTERVALS[0]
        control = {"trial": self.trial, "action": self.phase}
        if self.phase == "start":
            control.update(profile=self.profile, hold=self.hold, frames=self.frames)
        elif self.phase == "send":
            if self.started is None:
                self.started = now
            payload = STAMP.pack(self.trial, self.sent) + os.urandom(PROFILES[self.profile].capacity - STAMP.size)
            self.sent += 1
            self.interval = self.hold
            if self.sent == self.frames:
                self.phase = "report"
            return self.profile, p.packet(p.CAL, session, seq, payload)
        elif self.phase == "report":
            if self.duration is None:
                self.duration = max(self.hold, now - self.started)
            control.update(sent=self.sent)
        elif self.phase == "finish":
            control.update(profile=self.selected[0], hold=self.selected[1])
        else:
            raise RuntimeError(f"Invalid calibration phase: {self.phase}")
        return CONTROL_PROFILE, p.packet(CAL_CONTROL, session, seq, p.encode_json(control))

    def handle(self, status):
        if status.get("trial") != self.trial:
            return
        phase = status.get("phase")
        if self.phase == "start" and phase == "ready":
            self.phase = "send"
        elif self.phase == "report" and phase == "report":
            if self.duration is None:
                return
            received, streak = status.get("received"), status.get("streak")
            if type(received) is not int or type(streak) is not int or not (0 <= received <= self.sent and 0 <= streak <= self.sent):
                raise ValueError("Invalid calibration report")
            self.received = received
            good = received / self.sent >= MIN_SUCCESS and streak <= max(2, self.sent // 3)
            score = received * useful_bytes(self.profile) / self.duration if good else 0.0
            result = (score, self.profile, self.hold)
            if self.confirming:
                if score > 0:
                    self.confirmed_rate = score
                    self.selected = (self.profile, self.hold)
                    self.phase = "finish"
                else:
                    self.results = [item for item in self.results if item[1:] != result[1:]]
                    self._confirm_best()
            else:
                self.results.append(result)
                self.index += 1
                if self.index == len(self.candidates):
                    self._confirm_best()
                else:
                    self._begin()
        elif self.phase == "finish" and phase == "finished":
            self.done = True

    def _confirm_best(self):
        viable = [item for item in self.results if item[0] > 0]
        if not viable:
            raise RuntimeError("No reliable calibration setting. Keep all anchors visible and improve focus or stability.")
        _, profile, hold = max(viable)
        self.candidates = [(profile, hold)]
        self.index = 0
        self.confirming = True
        self._begin()

    def label(self):
        mode = "Confirm" if self.confirming else f"Cal {self.index + 1}/{len(self.candidates)}"
        progress = self.sent / self.frames * 100
        return (f"{mode}: {PROFILES[self.profile].name} {1 / self.hold:.1f}fps | "
                f"{progress:.0f}% {self.sent}/{self.frames} images | {self.phase}")


class TrialReceiver:
    def __init__(self):
        self.trial = None
        self.phase = None
        self.seen = set()
        self.latest_seq = -1

    def control(self, payload, seq):
        value = p.decode_json(payload)
        trial = value.get("trial")
        if type(trial) is not int or not 0 <= trial < 2 ** 32:
            raise ValueError("Invalid calibration trial ID")
        action = value.get("action")
        if action not in ("start", "report", "finish"):
            raise ValueError("Invalid calibration action")
        if action == "start":
            if seq < self.latest_seq:
                return
            if trial == self.trial:
                return
            profile, frames, hold = value.get("profile"), value.get("frames"), value.get("hold")
            if type(profile) is not int or not 0 <= profile < len(PROFILES) or type(frames) is not int or not 1 <= frames <= 1000 or hold not in INTERVALS:
                raise ValueError("Invalid calibration trial")
            self.trial, self.profile, self.frames = trial, profile, frames
            self.hold = hold
            self.latest_seq = seq
            self.phase = "ready"
            self.seen = set()
        elif trial == self.trial and action == "report":
            if type(value.get("sent")) is not int or value["sent"] != self.frames:
                raise ValueError("Incomplete calibration trial")
            self.phase = "report"
        elif trial == self.trial and action == "finish":
            profile, hold = value.get("profile"), value.get("hold")
            if type(profile) is not int or profile != self.profile or hold != self.hold:
                raise ValueError("Invalid selected calibration setting")
            self.phase = "finished"

    def add(self, profile, payload):
        if self.phase not in ("ready", "send") or len(payload) < STAMP.size:
            return
        trial, index = STAMP.unpack_from(payload)
        if trial != self.trial or profile != self.profile:
            return
        if index >= self.frames:
            raise ValueError("Invalid calibration frame index")
        self.seen.add(index)
        self.phase = "send"

    def status(self):
        longest = current = 0
        for index in range(getattr(self, "frames", 0)):
            current = 0 if index in self.seen else current + 1
            longest = max(longest, current)
        return {"state": "cal", "trial": self.trial, "phase": self.phase,
                "received": len(self.seen), "streak": longest}
