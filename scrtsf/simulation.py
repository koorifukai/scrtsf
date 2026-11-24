"""Synthetic camera effects for a local loopback test."""

import cv2
import numpy as np


LEVELS = {
    "Light": (0.02, 0.4, 2.0, 94),
    "Medium": (0.05, 0.6, 4.0, 88),
    "Strong": (0.08, 0.9, 8.0, 75),
}


def distort(image, level, rng, footprint=None, handheld=False, reflected=False, quarter_turns=0):
    perspective, blur, noise, quality = LEVELS[level]
    if footprint is not None:
        scale = min(footprint[0] / image.shape[1], footprint[1] / image.shape[0])
        image = cv2.resize(image, (max(1, round(image.shape[1] * scale)),
                                  max(1, round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    height, width = image.shape[:2]
    x, y = width * perspective, height * perspective
    src = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    dst = np.float32([[x, y * 0.5], [width - 1 - x, y],
                      [width - 1 - x * 0.5, height - 1 - y], [x * 0.5, height - 1 - y * 0.5]])
    if handheld:
        dst += rng.uniform(-1.5, 1.5, dst.shape).astype(np.float32)
    result = cv2.warpPerspective(image, cv2.getPerspectiveTransform(src, dst), (width, height))
    if handheld:
        kernel = np.zeros((3, 3), dtype=np.float32)
        kernel[1] = (0.15, 0.7, 0.15)
        result = cv2.filter2D(result, -1, kernel)
    result = cv2.GaussianBlur(result, (3, 3), blur)
    sample = rng.normal(0, noise, result.shape).astype(np.int16)
    result = np.clip(result.astype(np.int16) + sample, 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Could not encode simulated camera image")
    result = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if result is None:
        raise RuntimeError("Could not decode simulated camera image")
    if reflected:
        result = cv2.flip(result, 1)
    result = np.ascontiguousarray(np.rot90(result, quarter_turns))
    if footprint is not None:
        height, width = result.shape[:2]
        camera_height, camera_width = max(480, height + 40), max(640, width + 40)
        top, left = (camera_height - height) // 2, (camera_width - width) // 2
        result = cv2.copyMakeBorder(result, top, camera_height - height - top,
                                   left, camera_width - width - left,
                                   cv2.BORDER_CONSTANT, value=(50, 50, 50))
    return result
