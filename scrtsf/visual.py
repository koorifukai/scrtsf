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
PARAMETERS = cv2.aruco.DetectorParameters()
PARAMETERS.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
REFINED_DETECTOR = cv2.aruco.ArucoDetector(DICTIONARY, PARAMETERS)


@dataclass(frozen=True)
class Profile:
    name: str
    rows: int
    cols: int
    color: bool

    @property
    def label(self):
        return f"{self.name} ({self.cols}x{self.rows} cells)"

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
    Profile("Mono fine", 64, 112, False),
    Profile("Color fine", 64, 112, True),
    Profile("Color dense", 88, 156, True),
    Profile("Mono coarse", 32, 64, False),
    Profile("Mono medium", 48, 88, False),
    Profile("Mono extra fine", 128, 192, False),
    Profile("Color extra fine", 128, 192, True),
    Profile("Mono ultra fine", 160, 256, False),
    Profile("Color ultra fine", 160, 256, True),
)
PROFILE_ORDER = (3, 4, 0, 1, 2, 5, 6, 7, 8)
CONTROL_PROFILE = PROFILE_ORDER[0]


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
    label = " ".join(label.splitlines())
    status_height = 36
    scale = min(sw / p.width, (sh - status_height) / p.height)
    if scale <= 0:
        raise ValueError("Screen is too small")
    width, height = int(p.width * scale), int(p.height * scale)
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((sh, sw, 3), np.uint8)
    left, top = (sw - width) // 2, (sh - status_height - height) // 2
    canvas[top:top + height, left:left + width] = resized
    while label and cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] > sw - 20:
        label = label[:-1]
    cv2.putText(canvas, label, (10, sh - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


class AnchorsNotFound(ValueError):
    pass


def detect_anchors(image, refined=True):
    detector = REFINED_DETECTOR if refined else DETECTOR
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detected_ids = set()
    for reflected in (False, True):
        frame = cv2.flip(gray, 1) if reflected else gray
        corners, ids, _ = detector.detectMarkers(frame)
        if ids is None:
            continue
        found = {int(i): c.reshape(4, 2) for i, c in zip(ids.flatten(), corners)}
        detected_ids.update(found)
        for profile_id in range(len(PROFILES)):
            if all(profile_id * 4 + i in found for i in range(4)):
                return profile_id, found, reflected
    if detected_ids:
        raise AnchorsNotFound("Looking for four matching markers")
    raise AnchorsNotFound("Looking for four markers")


def locate(image, refined=True):
    profile_id, found, reflected = detect_anchors(image, refined)
    p = PROFILES[profile_id]
    src = np.float32([found[profile_id * 4 + i][i] for i in range(4)])
    if refined:
        positions = ((MARGIN, MARGIN), (p.width - MARGIN - MARKER, MARGIN),
                     (p.width - MARGIN - MARKER, p.height - MARGIN - MARKER),
                     (MARGIN, p.height - MARGIN - MARKER))
        dst = np.float32([[[x, y], [x + MARKER - 1, y],
                          [x + MARKER - 1, y + MARKER - 1], [x, y + MARKER - 1]]
                         for x, y in positions]).reshape(-1, 2)
        points = np.concatenate([found[profile_id * 4 + i] for i in range(4)])
        transform, _ = cv2.findHomography(points, dst)
        if transform is None:
            raise ValueError("Could not fit screen geometry")
    else:
        dst = np.float32([[MARGIN, MARGIN], [p.width - MARGIN - 1, MARGIN],
                         [p.width - MARGIN - 1, p.height - MARGIN - 1],
                         [MARGIN, p.height - MARGIN - 1]])
        transform = cv2.getPerspectiveTransform(src, dst)
    frame = cv2.flip(image, 1) if reflected else image
    warped = cv2.warpPerspective(frame, transform, (p.width, p.height))
    return profile_id, src, warped


def sample_grid(warped, p, shifted=False, centered=False):
    x = y = MARGIN + MARKER + GAP
    grid = warped[y:y + p.rows * CELL, x:x + p.cols * CELL]
    cells = grid.reshape(p.rows, CELL, p.cols, CELL, 3)
    if centered:
        samples = cells[:, 3:6, :, 3:6, :].mean(axis=(1, 3))
    elif shifted:
        samples = cells[:, 3:6, :, 4:7, :].mean(axis=(1, 3))
    else:
        samples = cells[:, 2:6, :, 2:6, :].mean(axis=(1, 3))
    if not p.color:
        samples = cv2.cvtColor(samples.astype(np.uint8), cv2.COLOR_BGR2GRAY)[:, :, None]
    return samples


def decode_samples(samples, p, threshold_fraction=0.5):
    low = np.percentile(samples, 15, axis=(0, 1))
    high = np.percentile(samples, 85, axis=(0, 1))
    if np.any(high - low < 40):
        raise ValueError("Insufficient contrast")
    bits = (samples > low + (high - low) * threshold_fraction).astype(np.uint8).reshape(-1)
    raw = np.packbits(bits)[:p.words * 255]
    raw = np.bitwise_xor(raw, _mask(raw.size)).reshape(255, p.words).T
    codec = reedsolo.RSCodec(PARITY)
    try:
        data = b"".join(bytes(codec.decode(row.tobytes())[0]) for row in raw)
        return protocol.unpack(data)
    except (reedsolo.ReedSolomonError, ValueError, IndexError) as exc:
        raise ValueError("Frame could not be decoded") from exc


def decode(image):
    profile_id, _, warped = locate(image)
    p = PROFILES[profile_id]
    try:
        packet = decode_samples(sample_grid(warped, p), p)
    except ValueError:
        if p.color:
            raise
        try:
            packet = decode_samples(sample_grid(warped, p, shifted=True), p, 0.7)
        except ValueError:
            try:
                packet = decode_samples(sample_grid(warped, p, centered=True), p, 0.6)
            except ValueError:
                # Preserve the geometry that decoded earlier camera captures.
                try:
                    _, _, warped = locate(image, refined=False)
                except AnchorsNotFound as exc:
                    raise ValueError("Frame could not be decoded") from exc
                packet = decode_samples(sample_grid(warped, p, shifted=True), p, 0.7)
    return profile_id, packet
