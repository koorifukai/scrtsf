"""Local diagnostics for individual camera images."""

import cv2
import numpy as np

from . import protocol, visual
from .session import NO_ACK_DATA, NO_ACK_META


def load_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The selected file is not a readable image")
    return image


def inspect_image(image, mirror=False):
    frame = cv2.flip(image, 1) if mirror else image
    _, ids, _ = visual.DETECTOR.detectMarkers(frame)
    marker_ids = [] if ids is None else sorted(int(value) for value in ids.flatten())
    height, width = frame.shape[:2]
    lines = [
        f"Image: {width} x {height} pixels",
        f"Camera input flip: {'on' if mirror else 'off'}",
        f"Marker IDs detected: {', '.join(map(str, marker_ids)) if marker_ids else 'none'}",
    ]
    try:
        profile_id, src, warped = visual.locate(frame)
    except ValueError as exc:
        lines.extend([f"Grid: {exc}", "Try the camera input flip, reposition the camera, or improve focus and lighting."])
        return "\n".join(lines)
    profile = visual.PROFILES[profile_id]
    top = np.linalg.norm(src[1] - src[0])
    bottom = np.linalg.norm(src[2] - src[3])
    left = np.linalg.norm(src[3] - src[0])
    right = np.linalg.norm(src[2] - src[1])
    horizontal = (top + bottom) * visual.CELL / (2 * (profile.width - 2 * visual.MARGIN - 1))
    vertical = (left + right) * visual.CELL / (2 * (profile.height - 2 * visual.MARGIN - 1))
    samples = visual.sample_grid(warped, profile)
    channels = 3 if profile.color else 1
    encoded_bits = profile.words * 255 * 8
    total_bits = profile.rows * profile.cols * channels
    flat = samples.reshape(-1)
    data_samples = flat[:encoded_bits]
    low, high = np.percentile(data_samples, [15, 85])
    lines.extend([
        f"Profile: {profile.name} ({profile.cols} x {profile.rows} cells)",
        f"Approximate camera pixels per cell: {horizontal:.1f} horizontal, {vertical:.1f} vertical",
        f"Data-cell brightness percentiles: {low:.0f} to {high:.0f} (0-255)",
        f"Unused padding: {total_bits - encoded_bits} of {total_bits} cell bits; a dark strip can be normal.",
    ])
    try:
        decoded_profile, packet = visual.decode(frame)
    except ValueError as exc:
        lines.extend([f"Packet: not decoded ({exc})",
                      "Try a clearer, closer image or different camera positioning; this is one frame only."])
    else:
        names = {protocol.HELLO: "hello", protocol.CAL: "calibration", protocol.META: "metadata",
                 protocol.DATA: "data", protocol.STATUS: "status", protocol.END: "end",
                 NO_ACK_META: "no-ACK metadata", NO_ACK_DATA: "no-ACK data"}
        kind = names.get(packet[0], f"type {packet[0]}")
        lines.append(f"Packet: decoded and CRC verified ({kind}; profile {visual.PROFILES[decoded_profile].name})")
    return "\n".join(lines)
