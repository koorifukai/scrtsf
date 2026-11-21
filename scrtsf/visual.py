"""Optical frames with corner markers and interleaved Reed-Solomon codewords."""

from dataclasses import dataclass

import cv2
import numpy as np
import reedsolo

from . import protocol

CELL = 8
MARGIN = 16
MARKER = 64
GAP = 12
PARITY = 32
PAYLOAD = 255 - PARITY
DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
DETECTOR = cv2.aruco.ArucoDetector(DICTIONARY)


@dataclass(frozen=True)
class Profile:
    name: str
    rows: int
    cols: int
    color: bool

    @property
    def words(self):
        return (self.rows * self.cols * (3 if self.color else 1) // 8) // 255

    @property
    def capacity(self):
        return self.words * PAYLOAD - protocol.HEADER.size - protocol.CRC.size

    @property
    def width(self):
        return 2 * (MARGIN + MARKER + GAP) + self.cols * CELL

    @property
    def height(self):
        return 2 * (MARGIN + MARKER + GAP) + self.rows * CELL


PROFILES = (
    Profile("mono", 64, 112, False),
    Profile("color", 64, 112, True),
    Profile("dense color", 88, 156, True),
)


def _mask(size):
    return np.random.default_rng(20261121).integers(0, 256, size, dtype=np.uint8)


def _encoded(data: bytes, profile: Profile):
    if len(data) > profile.words * PAYLOAD:
        raise ValueError("Packet too large for profile")
    data = data.ljust(profile.words * PAYLOAD, b"\0")
    codec = reedsolo.RSCodec(PARITY)
    words = [codec.encode(data[i:i + PAYLOAD]) for i in range(0, len(data), PAYLOAD)]
    raw = np.frombuffer(b"".join(words), dtype=np.uint8).reshape(profile.words, 255).T.flatten()
    return np.bitwise_xor(raw, _mask(raw.size))


def render(data: bytes, profile_id: int, screen: tuple[int, int], label: str = ""):
    p = PROFILES[profile_id]
    bits = np.unpackbits(_encoded(data, p))
    channels = 3 if p.color else 1
    padded = np.zeros(p.rows * p.cols * channels, dtype=np.uint8)
    padded[:len(bits)] = bits
    cells = (padded.reshape(p.rows, p.cols, channels) * 255).astype(np.uint8)
    if not p.color:
        cells = np.repeat(cells, 3, axis=2)
    image = np.full((p.height, p.width, 3), 255, np.uint8)
    x = y = MARGIN + MARKER + GAP
    image[y:y + p.rows * CELL, x:x + p.cols * CELL] = np.repeat(np.repeat(cells, CELL, axis=0), CELL, axis=1)
    positions = ((MARGIN, MARGIN), (p.width - MARGIN - MARKER, MARGIN),
                 (p.width - MARGIN - MARKER, p.height - MARGIN - MARKER),
                 (MARGIN, p.height - MARGIN - MARKER))
    for corner, (mx, my) in enumerate(positions):
        marker = cv2.aruco.generateImageMarker(DICTIONARY, profile_id * 4 + corner, MARKER)
        image[my:my + MARKER, mx:mx + MARKER] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    sw, sh = screen
    scale = min(sw / p.width, (sh - 36) / p.height)
    if scale <= 0:
        raise ValueError("Screen is too small")
    width, height = int(p.width * scale), int(p.height * scale)
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((sh, sw, 3), np.uint8)
    left, top = (sw - width) // 2, (sh - 36 - height) // 2
    canvas[top:top + height, left:left + width] = resized
    cv2.putText(canvas, label[:140], (10, sh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def decode(image):
    corners, ids, _ = DETECTOR.detectMarkers(image)
    if ids is None:
        raise ValueError("Looking for four markers")
    found = {int(i): c.reshape(4, 2) for i, c in zip(ids.flatten(), corners)}
    for profile_id, p in enumerate(PROFILES):
        if all(profile_id * 4 + i in found for i in range(4)):
            break
    else:
        raise ValueError("Looking for four matching markers")
    src = np.float32([found[profile_id * 4 + i][i] for i in range(4)])
    dst = np.float32([[MARGIN, MARGIN], [p.width - MARGIN - 1, MARGIN],
                      [p.width - MARGIN - 1, p.height - MARGIN - 1],
                      [MARGIN, p.height - MARGIN - 1]])
    transform = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, transform, (p.width, p.height))
    x = y = MARGIN + MARKER + GAP
    grid = warped[y:y + p.rows * CELL, x:x + p.cols * CELL]
    samples = grid.reshape(p.rows, CELL, p.cols, CELL, 3)[:, 2:6, :, 2:6, :].mean(axis=(1, 3))
    if not p.color:
        samples = cv2.cvtColor(samples.astype(np.uint8), cv2.COLOR_BGR2GRAY)[:, :, None]
    low = np.percentile(samples, 15, axis=(0, 1))
    high = np.percentile(samples, 85, axis=(0, 1))
    if np.any(high - low < 40):
        raise ValueError("Insufficient contrast")
    bits = (samples > (low + high) / 2).astype(np.uint8).reshape(-1)
    raw = np.packbits(bits)[:p.words * 255]
    raw = np.bitwise_xor(raw, _mask(raw.size)).reshape(255, p.words).T
    codec = reedsolo.RSCodec(PARITY)
    try:
        data = b"".join(bytes(codec.decode(row.tobytes())[0]) for row in raw)
        return profile_id, protocol.unpack(data)
    except (reedsolo.ReedSolomonError, ValueError, IndexError) as exc:
        raise ValueError("Frame could not be decoded") from exc
