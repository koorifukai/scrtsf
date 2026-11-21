"""Synthetic camera effects for a local loopback test."""

import cv2
import numpy as np


LEVELS = {
    "Light": (0.02, 0.4, 2.0, 94),
    "Medium": (0.05, 0.6, 4.0, 88),
    "Strong": (0.08, 0.9, 8.0, 75),
}


def distort(image, level, rng):
    perspective, blur, noise, quality = LEVELS[level]
    height, width = image.shape[:2]
    x, y = width * perspective, height * perspective
    src = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    dst = np.float32([[x, y * 0.5], [width - 1 - x, y],
                      [width - 1 - x * 0.5, height - 1 - y], [x * 0.5, height - 1 - y * 0.5]])
    result = cv2.warpPerspective(image, cv2.getPerspectiveTransform(src, dst), (width, height))
    result = cv2.GaussianBlur(result, (3, 3), blur)
    sample = rng.normal(0, noise, result.shape).astype(np.int16)
    result = np.clip(result.astype(np.int16) + sample, 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Could not encode simulated camera image")
    result = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if result is None:
        raise RuntimeError("Could not decode simulated camera image")
    return result
