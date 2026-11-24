"""State machines for the bidirectional optical transfer."""

import os
import struct
import time
import zlib
from collections import deque
from pathlib import Path

from . import protocol as p
from .calibration import CAL_CONTROL, FRAME_SECONDS, Calibration, TrialReceiver
from .visual import CONTROL_PROFILE, PROFILE_ORDER, PROFILES

ENTRY = struct.Struct("!IH")
CAL_SECONDS = 5.0
MAX_QUEUE_BYTES = 32 * 1024 * 1024
MAX_QUEUE_FILES = 64
NO_ACK_META, NO_ACK_DATA = 7, 8
META_FRAGMENT, NO_ACK_META_FRAGMENT = 9, 10
FRAGMENT = struct.Struct("!HH")
MAX_METADATA = 1024


def short_name(name):
    return name if len(name) <= 20 else name[:17] + "..."


def metadata_fragment(meta, index):
    data = p.encode_json(meta)
    if len(data) > MAX_METADATA:
        raise ValueError("File metadata is too large")
    size = PROFILES[CONTROL_PROFILE].capacity - FRAGMENT.size
    offset = index * size % (((len(data) + size - 1) // size) * size)
    return FRAGMENT.pack(offset, len(data)) + data[offset:offset + size]


class MetadataBuffer:
    def __init__(self):
        self.sessions = {}

    def add(self, session, payload):
        if len(payload) <= FRAGMENT.size:
            raise ValueError("Short metadata fragment")
        offset, total = FRAGMENT.unpack_from(payload)
        content = payload[FRAGMENT.size:]
        if not (0 < total <= MAX_METADATA and offset + len(content) <= total):
            raise ValueError("Invalid metadata fragment")
        if session not in self.sessions:
            if len(self.sessions) >= MAX_QUEUE_FILES:
                raise ValueError("Too many metadata sessions")
            self.sessions[session] = (total, {})
        size, parts = self.sessions[session]
        if size != total:
            raise ValueError("Conflicting metadata length")
        for index, byte in enumerate(content, offset):
            if index in parts and parts[index] != byte:
                raise ValueError("Conflicting metadata fragment")
            parts[index] = byte
        if len(parts) == total:
            data = bytes(parts[index] for index in range(total))
            del self.sessions[session]
            return data
        return None


def prepare_queue(paths):
    paths = [Path(path) for path in paths]
    if not paths or len(paths) > MAX_QUEUE_FILES:
        raise ValueError(f"Select between 1 and {MAX_QUEUE_FILES} files")
    files, names = [], set()
    total = 0
    for path in paths:
        meta, content = p.prepare(path)
        p.validate_meta(meta)
        name = meta["name"].casefold()
        if name in names:
            raise ValueError(f"Duplicate file name in queue: {meta['name']}")
        names.add(name)
        total += len(content)
        if total > MAX_QUEUE_BYTES:
            raise ValueError("Compressed queue exceeds the 32 MiB limit")
        files.append((os.urandom(8), meta, content))
    queue_blocks = sum(max(1, (len(content) + p.BLOCK - 1) // p.BLOCK) for _, _, content in files)
    queue_id = os.urandom(8).hex()
    for index, (_, meta, _) in enumerate(files):
        meta.update(queue_blocks=queue_blocks, file_index=index + 1, file_count=len(files), queue_id=queue_id)
        if len(p.encode_json(meta)) > MAX_METADATA:
            raise ValueError("File metadata is too large")
    return paths, files


class Sender:
    def __init__(self, paths, now, max_profile=None):
        self.paths, self.files = prepare_queue(paths)
        self.session = os.urandom(8)
        self.state = "hello"
        self.profile = CONTROL_PROFILE
        self.file_index = -1
        self.base = 0
        self.mask = 0
        self.cursor = 0
        self.last_progress = now
        self.received = 0
        self.seq = 0
        self.message = "Point both webcams at the opposite screen"
        self.done = False
        self.frame_interval = FRAME_SECONDS
        self.last_ack = now
        self.calibration = None
        self.resuming = False
        self.confirmed_queue = 0
        self.total_queue = sum(max(1, (len(content) + p.BLOCK - 1) // p.BLOCK) for _, _, content in self.files)
        self.file_started = now
        self.last_tuning = now
        self.expected_rate = 0
        self.progress_samples = deque([(now, 0)], maxlen=128)
        if max_profile is not None and max_profile not in PROFILE_ORDER:
            raise ValueError("Invalid profile")
        self.profile_order = (PROFILE_ORDER if max_profile is None else
                              PROFILE_ORDER[:PROFILE_ORDER.index(max_profile) + 1])

    def next_file(self, now):
        if self.file_index >= 0:
            self.confirmed_queue += max(1, (len(self.content) + p.BLOCK - 1) // p.BLOCK)
        self.file_index += 1
        if self.file_index == len(self.paths):
            self.state = "end"
            return
        self.session, self.meta, self.content = self.files[self.file_index]
        self.state = "meta"
        self.received = 0
        self.last_progress = now
        self.base = self.mask = self.cursor = 0
        self.meta_index = 0
        self.file_started = now
        self.last_tuning = now
        self.progress_samples.clear()
        self.progress_samples.append((now, 0))

    def start_calibration(self, resuming=False):
        self.calibration = Calibration(self.profile_order, CAL_SECONDS)
        self.state = "cal"
        self.resuming = resuming
        self.message = "Recalibrating without discarding blocks" if resuming else "Calibrating"

    def frame(self, now):
        self.seq = (self.seq + 1) & 0xFFFFFFFF
        if self.state == "hello":
            return CONTROL_PROFILE, p.packet(p.HELLO, self.session, self.seq)
        if self.state == "cal":
            result = self.calibration.frame(self.session, self.seq, now)
            self.frame_interval = self.calibration.interval
            return result
        if self.state == "meta":
            self.frame_interval = FRAME_SECONDS
            payload = metadata_fragment(self.meta, self.meta_index)
            self.meta_index += 1
            return CONTROL_PROFILE, p.packet(META_FRAGMENT, self.session, self.seq, payload)
        if self.state == "data":
            while len(self.progress_samples) > 1 and self.progress_samples[1][0] <= now - 10:
                self.progress_samples.popleft()
            first_time, first_count = self.progress_samples[0]
            recent_rate = (self.received - first_count) * p.BLOCK / max(1, now - first_time)
            deteriorated = now - self.last_tuning >= 30 and self.expected_rate > 0 and recent_rate < self.expected_rate * 0.4
            if (now - self.last_progress > 15 or deteriorated) and now - self.last_ack < 8:
                self.start_calibration(resuming=True)
                return self.frame(now)
            self.frame_interval = self.data_interval
            indices = [self.base + i for i in range(p.WINDOW) if self.mask & (1 << i)]
            payload = bytearray()
            if indices:
                start = self.cursor % len(indices)
                ordered = indices[start:] + indices[:start]
                for index in ordered:
                    block = self.content[index * p.BLOCK:(index + 1) * p.BLOCK]
                    entry = ENTRY.pack(index, len(block)) + block
                    if len(payload) + len(entry) > PROFILES[self.profile].capacity:
                        break
                    payload.extend(entry)
                    self.cursor += 1
            return self.profile, p.packet(p.DATA, self.session, self.seq, bytes(payload))
        if self.state == "end":
            self.frame_interval = FRAME_SECONDS
            return CONTROL_PROFILE, p.packet(p.END, self.session, self.seq)
        raise RuntimeError(f"Unknown sender state: {self.state}")

    def handle(self, decoded, now):
        _, (kind, session, _, payload) = decoded
        if kind != p.STATUS or session != self.session:
            return
        status = p.decode_json(payload)
        self.last_ack = now
        state = status.get("state")
        if state == "error":
            raise RuntimeError("Receiver: " + str(status.get("message", "unknown error")))
        if self.state == "hello" and state == "hello":
            self.start_calibration()
        elif self.state == "cal" and state == "cal":
            self.calibration.handle(status)
            if self.calibration.done:
                self.profile, self.data_interval = self.calibration.selected
                self.expected_rate = self.calibration.confirmed_rate
                self.last_tuning = now
                self.progress_samples.clear()
                self.progress_samples.append((now, self.received))
                self.last_progress = now
                if self.resuming:
                    self.state = "data"
                else:
                    self.next_file(now)
        elif self.state in ("meta", "data") and state in ("receiving", "verifying"):
            base, mask, received = status.get("base"), status.get("mask"), status.get("received")
            count = (len(self.content) + p.BLOCK - 1) // p.BLOCK
            if any(type(n) is not int for n in (base, mask, received)) or not (0 <= base <= count and 0 <= mask < 2 ** p.WINDOW and 0 <= received <= count):
                raise ValueError("Invalid receiver status")
            if base + mask.bit_length() > count:
                raise ValueError("Invalid missing block mask")
            if received > self.received:
                self.received = received
                self.last_progress = now
                self.progress_samples.append((now, received))
            self.base, self.mask = base, mask
            self.state = "data"
        elif self.state in ("meta", "data") and state == "complete":
            self.next_file(now)
        elif self.state == "end" and state == "end":
            self.done = True

    def label(self):
        if self.state == "cal":
            return self.calibration.label()
        if self.state in ("meta", "data"):
            total = (len(self.content) + p.BLOCK - 1) // p.BLOCK
            file_percent = self.received / total * 100 if total else 0
            queue = (self.confirmed_queue + self.received) / self.total_queue * 100
            rate = self.received * p.BLOCK / max(0.001, self.last_ack - self.file_started)
            eta = (total - self.received) * p.BLOCK / rate if rate else None
            remaining = f"~{eta:.0f}s" if eta is not None else "?"
            return (f"Send {self.file_index + 1}/{len(self.paths)} {short_name(self.paths[self.file_index].name)} | "
                    f"File {file_percent:.1f}% queue {queue:.1f}% | {self.received}/{total}b ACK | "
                    f"{rate / 1024:.1f}K/s ETA {remaining}")
        percent = self.confirmed_queue / self.total_queue * 100 if self.total_queue else 0
        if self.state == "hello":
            return "Waiting for peer"
        return f"{self.state} | queue {percent:.1f}% | {self.confirmed_queue}/{self.total_queue}b ACK"


class NoAckSender:
    """Continuously cycle a queue without relying on a return channel."""

    def __init__(self, paths, profile=CONTROL_PROFILE):
        if not 0 <= profile < len(PROFILES):
            raise ValueError("Invalid profile")
        _, self.files = prepare_queue(paths)
        self.frame_interval = FRAME_SECONDS
        self.profile = profile
        self.file_index = 0
        self.seq = 0
        self.phase = 0
        self.offset = 0
        self.cycle = 0
        self.message = ""
        self.display_label = ""
        self.meta_index = 0
        self.cycle_blocks = 0
        self.total_blocks = sum(max(1, (len(content) + p.BLOCK - 1) // p.BLOCK) for _, _, content in self.files)

    def frame(self, now):
        del now
        self.seq = (self.seq + 1) & 0xFFFFFFFF
        session, meta, content = self.files[self.file_index]
        count = (len(content) + p.BLOCK - 1) // p.BLOCK
        self.update_label(meta, count)
        fragment_size = PROFILES[CONTROL_PROFILE].capacity - FRAGMENT.size
        fragment_count = (len(p.encode_json(meta)) + fragment_size - 1) // fragment_size
        metadata_frames = max(2, fragment_count) + self.cycle % 2
        if self.phase < metadata_frames or self.phase % 10 == 0:
            payload = metadata_fragment(meta, self.meta_index)
            self.meta_index += 1
            self.phase += 1
            if count == 0 and self.phase >= metadata_frames:
                self.cycle_blocks += 1
                self.update_label(meta, count)
                self._advance()
            return CONTROL_PROFILE, p.packet(NO_ACK_META_FRAGMENT, session, self.seq, payload)
        payload = bytearray()
        while self.offset < count:
            index = (self.offset + self.cycle) % count
            block = content[index * p.BLOCK:(index + 1) * p.BLOCK]
            entry = ENTRY.pack(index, len(block)) + block
            if len(payload) + len(entry) > PROFILES[self.profile].capacity:
                break
            payload.extend(entry)
            self.offset += 1
            self.cycle_blocks += 1
        self.phase += 1
        self.update_label(meta, count)
        if self.offset == count:
            self._advance()
        return self.profile, p.packet(NO_ACK_DATA, session, self.seq, bytes(payload))

    def _advance(self):
        self.offset = 0
        self.phase = 0
        self.file_index += 1
        if self.file_index == len(self.files):
            self.file_index = 0
            self.cycle += 1
            self.cycle_blocks = 0
        self.meta_index = self.cycle

    def update_label(self, meta, count):
        self.display_label = (f"No ACK c{self.cycle + 1} | {self.file_index + 1}/{len(self.files)} {short_name(meta['name'])} | "
                              f"cycle {self.cycle_blocks / self.total_blocks * 100:.1f}% {self.offset}/{count}b | Delivery unconfirmed")

    def label(self):
        return self.display_label


class Receiver:
    def __init__(self, directory, no_ack=False):
        if not directory:
            raise ValueError("Select an output folder")
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise ValueError("Select an existing output folder")
        self.session = b"\0" * 8
        self.state = "waiting"
        self.incoming = None
        self.counts = [set() for _ in PROFILES]
        self.message = "Point both webcams at the opposite screen"
        self.saved = []
        self.no_ack = no_ack
        self.transfers = {}
        self.completed = set()
        self.queue_bytes = 0
        self.metadata = MetadataBuffer()
        self.calibration = None
        self.cal_active = False
        self.frame_interval = FRAME_SECONDS
        self.pending = []
        self.verification_announced = False
        self.saved_blocks = 0
        self.file_started = None

    def handle_no_ack(self, kind, session, payload):
        if kind == NO_ACK_META:
            meta = p.decode_json(payload)
            if session in self.transfers:
                if self.transfers[session].meta != meta:
                    raise ValueError("Conflicting metadata for session")
                return
            incoming = p.Incoming(meta)
            if len(self.transfers) >= MAX_QUEUE_FILES or self.queue_bytes + meta["length"] > MAX_QUEUE_BYTES:
                self.state = "error"
                self.message = "No-ACK queue exceeds receiver limits"
                return
            self.transfers[session] = incoming
            self.queue_bytes += meta["length"]
            self.incoming = incoming
            if not incoming.count:
                self._finish_no_ack(session)
        elif kind == NO_ACK_DATA and session in self.transfers:
            incoming = self.transfers[session]
            if session in self.completed or any(item[0] == session for item in self.pending):
                return
            self.incoming = incoming
            for index, block in parse_entries(payload):
                incoming.add(index, block)
            if incoming.received == incoming.count:
                self.pending.append((session, incoming))
                self.state = "verifying"

    def _finish_no_ack(self, session):
        incoming = self.transfers[session]
        try:
            path = incoming.save(self.directory)
        except (OSError, ValueError, zlib.error) as exc:
            self.state = "error"
            self.message = str(exc)
            return
        self.completed.add(session)
        self.saved.append(path)
        self.message = f"Saved {path.name}"
        self.state = "waiting"

    def handle(self, decoded):
        profile, (kind, session, seq, payload) = decoded
        if kind in (META_FRAGMENT, NO_ACK_META_FRAGMENT):
            if (kind == NO_ACK_META_FRAGMENT) != self.no_ack:
                return
            payload = self.metadata.add(session, payload)
            if payload is None:
                return
            kind = NO_ACK_META if self.no_ack else p.META
        if self.no_ack:
            if self.state != "error":
                self.handle_no_ack(kind, session, payload)
            return
        if kind == CAL_CONTROL and session == self.session:
            if self.calibration is None:
                self.calibration = TrialReceiver()
            self.calibration.control(payload, seq)
            self.cal_active = self.calibration.trial is not None
            return
        if kind == p.HELLO:
            if self.state in ("waiting", "hello", "cal"):
                if session != self.session:
                    self.counts = [set() for _ in PROFILES]
                    self.calibration = None
                    self.cal_active = False
                self.session, self.state = session, "hello"
        elif kind == p.CAL and session == self.session and self.state in ("hello", "cal"):
            if self.calibration is not None:
                self.calibration.add(profile, payload)
            else:
                self.state = "cal"
                self.counts[profile].add(seq)
        elif kind == p.CAL and session == self.session and self.cal_active:
            self.calibration.add(profile, payload)
        elif kind == p.META:
            if session != self.session and self.state in ("hello", "cal", "complete"):
                meta = p.decode_json(payload)
                incoming = p.Incoming(meta)
                if self.incoming and self.incoming.meta.get("queue_id") != meta.get("queue_id"):
                    self.saved_blocks = 0
                self.session, self.incoming, self.state = session, incoming, "receiving"
                self.cal_active = False
                self.calibration = None
                self.file_started = time.monotonic()
                if incoming.count == 0:
                    self.finish()
        elif kind == p.DATA and session == self.session and self.state == "receiving":
            self.cal_active = False
            for index, block in parse_entries(payload):
                self.incoming.add(index, block)
            if self.incoming.received == self.incoming.count:
                self.state = "verifying"
                self.pending.append((session, self.incoming))
        elif kind == p.END and session == self.session and self.state == "complete":
            self.state = "end"
            self.message = "All files received. Press Esc to close."

    def finish(self):
        try:
            path = self.incoming.save(self.directory)
        except (OSError, ValueError, zlib.error) as exc:
            self.state = "error"
            self.message = str(exc)
            return
        self.saved.append(path)
        self.saved_blocks += max(1, self.incoming.count)
        self.state = "complete"
        self.message = f"Saved {path.name}"

    def frame(self):
        if self.pending and self.state != "error":
            if not self.verification_announced:
                self.verification_announced = True
            else:
                session, _ = self.pending.pop(0)
                self.verification_announced = False
                if self.no_ack:
                    self._finish_no_ack(session)
                else:
                    self.finish()
        status = {"state": self.state}
        if self.cal_active and self.calibration is not None:
            status = self.calibration.status()
        elif self.state == "cal":
            status["counts"] = [len(items) for items in self.counts]
        elif self.state in ("receiving", "verifying"):
            base, mask = self.incoming.missing_window()
            status.update(base=base, mask=mask, received=self.incoming.received)
        elif self.state == "error":
            status["message"] = self.message[:200]
        payload = p.encode_json(status)
        while len(payload) > PROFILES[CONTROL_PROFILE].capacity and "message" in status:
            status["message"] = status["message"][:len(status["message"]) // 2]
            payload = p.encode_json(status)
        return CONTROL_PROFILE, p.packet(p.STATUS, self.session, 0, payload)

    def label(self):
        if self.cal_active and self.calibration is not None:
            trial = self.calibration
            return f"Cal {PROFILES[trial.profile].name} | {len(trial.seen)}/{trial.frames} images | ACK {trial.phase}"
        if self.no_ack:
            active = sum(item.received for session, item in self.transfers.items() if session not in self.completed)
            total = sum(item.count for session, item in self.transfers.items() if session not in self.completed)
            queue_id = self.incoming.meta.get("queue_id") if self.incoming else None
            received = sum(max(1, item.count) if session in self.completed else item.received
                           for session, item in self.transfers.items() if item.meta.get("queue_id") == queue_id)
            denominator = self.incoming.meta.get("queue_blocks") if self.incoming else None
            percent = f"{received / denominator * 100:.1f}%" if denominator else "unknown extent"
            current = self.pending[0][1] if self.pending else self.incoming
            file_info = (f"{short_name(current.meta['name'])} | File {current.received / max(1, current.count) * 100:.1f}%"
                         if current else "Waiting for file metadata")
            return f"No ACK {file_info} queue {percent} | {active}/{total}b | {len(self.saved)} saved | {self.state}"
        if self.state in ("receiving", "verifying", "complete") and self.incoming:
            count = self.incoming.count
            percent = self.incoming.received / count * 100 if count else 100
            denominator = self.incoming.meta.get("queue_blocks")
            confirmed = self.saved_blocks if self.state == "complete" else self.saved_blocks + self.incoming.received
            queue = f"{confirmed / denominator * 100:.1f}%" if denominator else "unknown extent"
            elapsed = max(0.001, time.monotonic() - self.file_started) if self.file_started is not None else 0
            rate = self.incoming.received * self.incoming.block_size / elapsed if elapsed else 0
            eta = (count - self.incoming.received) * self.incoming.block_size / rate if rate else None
            remaining = f"~{eta:.0f}s" if eta is not None else "?"
            return (f"{self.state} {short_name(self.incoming.meta['name'])} | File {percent:.1f}% queue {queue} | "
                    f"{self.incoming.received}/{count}b | {rate / 1024:.1f}K/s ETA {remaining}")
        return f"{self.state}: {self.message}"


def parse_entries(payload):
    entries = []
    offset = 0
    while offset < len(payload):
        if len(payload) - offset < ENTRY.size:
            raise ValueError("Truncated block header")
        index, size = ENTRY.unpack_from(payload, offset)
        offset += ENTRY.size
        if offset + size > len(payload):
            raise ValueError("Truncated block")
        entries.append((index, payload[offset:offset + size]))
        offset += size
    return entries
