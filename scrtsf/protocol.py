"""Bounded file containers and packets shared by both ends of the link."""

import hashlib
import json
import math
import struct
import zlib
from pathlib import Path

MAX_FILE = 3 * 1024 * 1024
BLOCK = 512
WINDOW = 32
HEADER = struct.Struct("!4sB8sIH")
CRC = struct.Struct("!I")
MAGIC = b"STF1"
HELLO, CAL, META, DATA, STATUS, END = range(1, 7)


def packet(kind: int, session: bytes, seq: int, payload: bytes = b"") -> bytes:
    if len(session) != 8 or len(payload) > 65535:
        raise ValueError("Invalid packet size or session ID")
    body = HEADER.pack(MAGIC, kind, session, seq, len(payload)) + payload
    return body + CRC.pack(zlib.crc32(body))


def unpack(data: bytes):
    if len(data) < HEADER.size + CRC.size:
        raise ValueError("Short packet")
    magic, kind, session, seq, size = HEADER.unpack_from(data)
    end = HEADER.size + size
    if magic != MAGIC or end + CRC.size > len(data):
        raise ValueError("Invalid packet header")
    if zlib.crc32(data[:end]) != CRC.unpack_from(data, end)[0]:
        raise ValueError("Packet checksum mismatch")
    return kind, session, seq, data[HEADER.size:end]


def encode_json(value) -> bytes:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def decode_json(data: bytes):
    value = json.loads(data.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected an object")
    return value


def prepare(path: Path):
    size = path.stat().st_size
    if size > MAX_FILE:
        raise ValueError(f"{path.name}: file exceeds the 3 MiB limit")
    data = path.read_bytes()
    if len(data) > MAX_FILE:
        raise ValueError(f"{path.name}: file exceeds the 3 MiB limit")
    compressed = zlib.compress(data, 6)
    use_compression = len(compressed) < len(data)
    content = compressed if use_compression else data
    meta = {
        "name": path.name,
        "size": len(data),
        "length": len(content),
        "compressed": use_compression,
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    return meta, content


def validate_meta(meta):
    name = meta.get("name")
    size, length = meta.get("size"), meta.get("length")
    digest, compressed = meta.get("sha256"), meta.get("compressed")
    if not isinstance(name, str) or not name or len(name.encode("utf-8")) > 255:
        raise ValueError("Invalid file name")
    if name in (".", "..") or any(c in name for c in '\\/:*?"<>|') or name[-1] in " .":
        raise ValueError("Unsafe file name")
    if name.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        raise ValueError("Reserved file name")
    if any(ord(c) < 32 for c in name):
        raise ValueError("Invalid file name")
    if type(size) is not int or type(length) is not int or not (0 <= size <= MAX_FILE and 0 <= length <= MAX_FILE):
        raise ValueError("Invalid file size")
    if type(compressed) is not bool or not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Invalid file metadata")
    try:
        bytes.fromhex(digest)
    except ValueError as exc:
        raise ValueError("Invalid file checksum") from exc


class Incoming:
    def __init__(self, meta):
        validate_meta(meta)
        self.meta = meta
        self.count = math.ceil(meta["length"] / BLOCK)
        self.blocks = [None] * self.count
        self.received = 0

    def add(self, index: int, data: bytes):
        if not 0 <= index < self.count:
            raise ValueError("Invalid block index")
        expected = min(BLOCK, self.meta["length"] - index * BLOCK)
        if len(data) != expected:
            raise ValueError("Invalid block length")
        if self.blocks[index] is None:
            self.blocks[index] = data
            self.received += 1

    def missing_window(self):
        for base in range(0, self.count, WINDOW):
            mask = sum(1 << i for i in range(min(WINDOW, self.count - base))
                       if self.blocks[base + i] is None)
            if mask:
                return base, mask
        return self.count, 0

    def save(self, directory: Path) -> Path:
        if self.received != self.count:
            raise ValueError("File is incomplete")
        content = b"".join(self.blocks)
        if self.meta["compressed"]:
            decoder = zlib.decompressobj()
            data = decoder.decompress(content, self.meta["size"] + 1)
            if not decoder.eof or decoder.unused_data or len(data) != self.meta["size"]:
                raise ValueError("Invalid compressed file")
        else:
            data = content
        if len(data) != self.meta["size"] or hashlib.sha256(data).hexdigest() != self.meta["sha256"]:
            raise ValueError("File checksum mismatch")
        path = directory / self.meta["name"]
        with path.open("xb") as output:
            output.write(data)
        return path
