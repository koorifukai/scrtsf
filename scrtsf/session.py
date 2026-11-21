"""State machines for the bidirectional optical transfer."""

import os
import struct
import zlib
from pathlib import Path

from . import protocol as p
from .visual import PROFILES

ENTRY = struct.Struct("!IH")
CAL_SECONDS = 5.0
FRAME_SECONDS = 0.4
MAX_QUEUE_BYTES = 32 * 1024 * 1024
MAX_QUEUE_FILES = 64
NO_ACK_META, NO_ACK_DATA = 7, 8


class Sender:
    def __init__(self, paths, now):
        self.paths = [Path(path) for path in paths]
        if not self.paths:
            raise ValueError("Select at least one file")
        for path in self.paths:
            if not path.is_file() or path.stat().st_size > p.MAX_FILE:
                raise ValueError(f"Not a file or larger than 3 MiB: {path}")
        self.session = os.urandom(8)
        self.state = "hello"
        self.cal_start = now
        self.cal_profile = 0
        self.sent = [0] * len(PROFILES)
        self.seen = [0] * len(PROFILES)
        self.profile = 0
        self.file_index = -1
        self.base = 0
        self.mask = 0
        self.cursor = 0
        self.last_progress = now
        self.received = 0
        self.seq = 0
        self.message = "Point both webcams at the opposite screen"
        self.done = False

    def next_file(self, now):
        self.file_index += 1
        if self.file_index == len(self.paths):
            self.state = "end"
            return
        self.meta, self.content = p.prepare(self.paths[self.file_index])
        p.validate_meta(self.meta)
        if len(p.encode_json(self.meta)) > PROFILES[0].capacity:
            raise ValueError("File metadata is too large")
        self.session = os.urandom(8)
        self.state = "meta"
        self.received = 0
        self.last_progress = now
        self.base = self.mask = self.cursor = 0

    def frame(self, now):
        self.seq = (self.seq + 1) & 0xFFFFFFFF
        if self.state == "hello":
            return 0, p.packet(p.HELLO, self.session, self.seq)
        if self.state == "cal":
            elapsed = now - self.cal_start
            stage = min(int(elapsed // CAL_SECONDS), len(PROFILES) - 1)
            if elapsed >= CAL_SECONDS * len(PROFILES):
                choices = [i for i in range(len(PROFILES))
                           if self.sent[i] and self.seen[i] / self.sent[i] >= 0.6]
                if not choices:
                    self.message = "Calibration failed; improve camera position"
                    self.state = "hello"
                    self.session = os.urandom(8)
                    return 0, p.packet(p.HELLO, self.session, self.seq)
                self.profile = max(choices, key=lambda i: self.seen[i] / self.sent[i] *
                                   max(1, PROFILES[i].capacity // (p.BLOCK + ENTRY.size)))
                self.next_file(now)
                return self.frame(now)
            self.cal_profile = stage
            self.sent[stage] += 1
            data = os.urandom(PROFILES[stage].capacity)
            return stage, p.packet(p.CAL, self.session, self.seq, data)
        if self.state == "meta":
            return 0, p.packet(p.META, self.session, self.seq, p.encode_json(self.meta))
        if self.state == "data":
            if self.profile > 0 and now - self.last_progress > 15:
                self.profile -= 1
                self.last_progress = now
                self.message = "Link slowed; switching to a more robust profile"
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
            return 0, p.packet(p.END, self.session, self.seq)
        raise RuntimeError(f"Unknown sender state: {self.state}")

    def handle(self, decoded, now):
        _, (kind, session, _, payload) = decoded
        if kind != p.STATUS or session != self.session:
            return
        status = p.decode_json(payload)
        state = status.get("state")
        if state == "error":
            raise RuntimeError("Receiver: " + str(status.get("message", "unknown error")))
        if self.state == "hello" and state == "hello":
            self.state = "cal"
            self.cal_start = now
            self.sent = [0] * len(PROFILES)
            self.seen = [0] * len(PROFILES)
            self.message = "Calibrating"
        elif self.state == "cal" and state == "cal":
            counts = status.get("counts")
            if isinstance(counts, list) and len(counts) == len(PROFILES) and all(type(n) is int and n >= 0 for n in counts):
                self.seen = counts
        elif self.state in ("meta", "data") and state == "receiving":
            base, mask, received = status.get("base"), status.get("mask"), status.get("received")
            count = (len(self.content) + p.BLOCK - 1) // p.BLOCK
            if any(type(n) is not int for n in (base, mask, received)) or not (0 <= base <= count and 0 <= mask < 2 ** p.WINDOW and 0 <= received <= count):
                raise ValueError("Invalid receiver status")
            if base + mask.bit_length() > count:
                raise ValueError("Invalid missing block mask")
            if received > self.received:
                self.received = received
                self.last_progress = now
            self.base, self.mask = base, mask
            self.state = "data"
        elif self.state in ("meta", "data") and state == "complete":
            self.next_file(now)
        elif self.state == "end" and state == "end":
            self.done = True

    def label(self):
        if self.state == "cal":
            return f"Calibrating: {PROFILES[self.cal_profile].name} | Keep both screens visible | Esc: stop"
        if self.state in ("meta", "data"):
            total = (len(self.content) + p.BLOCK - 1) // p.BLOCK
            return f"Sending {self.file_index + 1}/{len(self.paths)}: {self.paths[self.file_index].name} | {self.received}/{total} blocks | {PROFILES[self.profile].name} | Esc: stop"
        return f"{self.message} | {self.state} | Esc: stop"


class NoAckSender:
    """Continuously cycle a queue without relying on a return channel."""

    def __init__(self, paths, profile=0):
        if not paths or len(paths) > MAX_QUEUE_FILES:
            raise ValueError(f"Select between 1 and {MAX_QUEUE_FILES} files")
        if not 0 <= profile < len(PROFILES):
            raise ValueError("Invalid profile")
        self.files = []
        names = set()
        total = 0
        for path in map(Path, paths):
            meta, content = p.prepare(path)
            p.validate_meta(meta)
            if len(p.encode_json(meta)) > PROFILES[0].capacity:
                raise ValueError("File metadata is too large")
            name = meta["name"].casefold()
            if name in names:
                raise ValueError(f"Duplicate file name in queue: {meta['name']}")
            names.add(name)
            total += len(content)
            if total > MAX_QUEUE_BYTES:
                raise ValueError("Compressed queue exceeds the 32 MiB limit")
            self.files.append((os.urandom(8), meta, content))
        self.profile = profile
        self.file_index = 0
        self.seq = 0
        self.phase = 0
        self.offset = 0
        self.cycle = 0
        self.message = ""
        self.display_label = ""

    def frame(self, now):
        del now
        self.seq = (self.seq + 1) & 0xFFFFFFFF
        session, meta, content = self.files[self.file_index]
        count = (len(content) + p.BLOCK - 1) // p.BLOCK
        self.display_label = (f"No ACK | cycle {self.cycle + 1} | {self.file_index + 1}/{len(self.files)} "
                              f"{meta['name']} | sent {self.offset}/{count} blocks | Esc: stop")
        if self.phase % 10 == 0:
            self.phase += 1
            if count == 0:
                self._advance()
            return 0, p.packet(NO_ACK_META, session, self.seq, p.encode_json(meta))
        payload = bytearray()
        while self.offset < count:
            index = (self.offset + self.cycle) % count
            block = content[index * p.BLOCK:(index + 1) * p.BLOCK]
            entry = ENTRY.pack(index, len(block)) + block
            if len(payload) + len(entry) > PROFILES[self.profile].capacity:
                break
            payload.extend(entry)
            self.offset += 1
        self.phase += 1
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
            if not incoming.count:
                self._finish_no_ack(session)
        elif kind == NO_ACK_DATA and session in self.transfers:
            incoming = self.transfers[session]
            if session in self.completed:
                return
            for index, block in parse_entries(payload):
                incoming.add(index, block)
            if incoming.received == incoming.count:
                self._finish_no_ack(session)

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

    def handle(self, decoded):
        profile, (kind, session, seq, payload) = decoded
        if self.no_ack:
            if self.state != "error":
                self.handle_no_ack(kind, session, payload)
            return
        if kind == p.HELLO:
            if self.state in ("waiting", "hello", "cal"):
                if session != self.session:
                    self.counts = [set() for _ in PROFILES]
                self.session, self.state = session, "hello"
        elif kind == p.CAL and session == self.session and self.state in ("hello", "cal"):
            self.state = "cal"
            self.counts[profile].add(seq)
        elif kind == p.META:
            if session != self.session and self.state in ("hello", "cal", "complete"):
                meta = p.decode_json(payload)
                incoming = p.Incoming(meta)
                self.session, self.incoming, self.state = session, incoming, "receiving"
                if incoming.count == 0:
                    self.finish()
        elif kind == p.DATA and session == self.session and self.state == "receiving":
            for index, block in parse_entries(payload):
                self.incoming.add(index, block)
            if self.incoming.received == self.incoming.count:
                self.finish()
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
        self.state = "complete"
        self.message = f"Saved {path.name}"

    def frame(self):
        status = {"state": self.state}
        if self.state == "cal":
            status["counts"] = [len(items) for items in self.counts]
        elif self.state == "receiving":
            base, mask = self.incoming.missing_window()
            status.update(base=base, mask=mask, received=self.incoming.received)
        elif self.state == "error":
            status["message"] = self.message[:200]
        return 0, p.packet(p.STATUS, self.session, 0, p.encode_json(status))

    def label(self):
        if self.no_ack:
            active = sum(item.received for session, item in self.transfers.items() if session not in self.completed)
            total = sum(item.count for session, item in self.transfers.items() if session not in self.completed)
            return f"No ACK | {len(self.saved)} saved | {active}/{total} blocks | {self.message} | Esc: stop"
        if self.state == "receiving":
            return f"Receiving {self.incoming.meta['name']} | {self.incoming.received}/{self.incoming.count} blocks | Esc: stop"
        return f"{self.state}: {self.message} | Esc: stop"


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
