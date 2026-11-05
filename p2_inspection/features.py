"""Hand-built features that survive lighting changes.

A least-squares plane is fitted to each image and subtracted, which removes
the lighting gradient. The residual is then searched for the two defect
shapes: thin lines (scratches) and compact dark blobs (pits).
"""

from __future__ import annotations

from .images import SIZE

FEATURE_NAMES = ("darkest", "dark_pixels", "line_energy", "blob_energy", "residual_std")
_DIRS = ((1, 0), (0, 1), (1, 1), (1, -1))
_C = (SIZE - 1) / 2
_SXX = sum((x - _C) ** 2 for x in range(SIZE)) * SIZE


def detrend(image) -> list[list[float]]:
    n = SIZE * SIZE
    mean = sum(map(sum, image)) / n
    bx = sum((x - _C) * v for row in image for x, v in enumerate(row)) / _SXX
    by = sum((y - _C) * v for y, row in enumerate(image) for v in row) / _SXX
    return [[v - (mean + bx * (x - _C) + by * (y - _C)) for x, v in enumerate(row)] for y, row in enumerate(image)]


def extract(image) -> list[float]:
    r = detrend(image)
    flat = [v for row in r for v in row]
    neg = [[min(v, 0.0) for v in row] for row in r]
    line = 0.0
    for y in range(SIZE):
        for x in range(SIZE):
            for dx, dy in _DIRS:
                s, xx, yy = 0.0, x, y
                for _ in range(5):
                    if not (0 <= xx < SIZE and 0 <= yy < SIZE):
                        break
                    s += neg[yy][xx]
                    xx, yy = xx + dx, yy + dy
                line = min(line, s)
    blob = 0.0
    for y in range(1, SIZE - 1):
        for x in range(1, SIZE - 1):
            s = sum(neg[yy][xx] for yy in (y - 1, y, y + 1) for xx in (x - 1, x, x + 1))
            blob = min(blob, s)
    std = (sum(v * v for v in flat) / len(flat)) ** 0.5
    return [-min(flat), float(sum(v < -0.08 for v in flat)), -line, -blob, std]


def raw_darkest(image) -> float:
    """The baseline's measurement: how far the darkest 3 x 3 patch sits below the image mean.

    No lighting correction, as in a fixed rule-based vision check.
    """
    flat = [v for row in image for v in row]
    mean = sum(flat) / len(flat)
    best = 0.0
    for y in range(1, SIZE - 1):
        for x in range(1, SIZE - 1):
            patch = sum(image[yy][xx] for yy in (y - 1, y, y + 1) for xx in (x - 1, x, x + 1)) / 9
            best = max(best, mean - patch)
    return best
