"""Synthetic part images.

Each part is a 16 x 16 grey-level image of a machined surface: a base grey
level, a lighting gradient that varies from part to part, and sensor noise.
Defects are scratches (thin dark lines) and pits (small dark blobs). Lighting
gradients are the nuisance that defeats a fixed-threshold rule; a detrending
feature extractor removes them.

A public defect-image dataset can replace this generator: anything that yields
(image, label) pairs with images as lists of rows of floats in [0, 1] works.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

SIZE = 16
GRADIENT = 0.003        # lighting slope per pixel, drawn per part
NOISE = 0.03            # sensor noise (standard deviation)
DEPTH = (0.08, 0.18)    # how much darker a defect is than the surface


@dataclass(frozen=True)
class Part:
    image: tuple[tuple[float, ...], ...]
    defective: bool
    kind: str  # "ok", "scratch" or "pit"


def _clip(v: float) -> float:
    return 0.0 if v < 0 else 1.0 if v > 1 else v


def make_part(rng: random.Random, defective: bool) -> Part:
    base = rng.uniform(0.45, 0.65)
    gx, gy = rng.uniform(-GRADIENT, GRADIENT), rng.uniform(-GRADIENT, GRADIENT)  # lighting gradient per pixel
    noise = NOISE
    img = [[base + gx * (x - SIZE / 2) + gy * (y - SIZE / 2) + rng.gauss(0, noise) for x in range(SIZE)] for y in range(SIZE)]
    kind = "ok"
    if defective:
        kind = rng.choice(("scratch", "pit"))
        depth = rng.uniform(*DEPTH)
        if kind == "scratch":
            length = rng.randint(4, 9)
            x, y = rng.randint(2, SIZE - 3), rng.randint(2, SIZE - 3)
            dx, dy = rng.choice(((1, 0), (0, 1), (1, 1), (1, -1)))
            for _ in range(length):
                if 0 <= x < SIZE and 0 <= y < SIZE:
                    img[y][x] -= depth
                x, y = x + dx, y + dy
        else:
            cx, cy = rng.randint(3, SIZE - 4), rng.randint(3, SIZE - 4)
            for y in range(cy - 1, cy + 2):
                for x in range(cx - 1, cx + 2):
                    img[y][x] -= depth * (1.0 if (x, y) == (cx, cy) else 0.6)
    return Part(tuple(tuple(_clip(v) for v in row) for row in img), defective, kind)


def make_dataset(n: int, defect_rate: float, seed: int) -> list[Part]:
    """Defect rate here is the sampling rate, not the plant's prevalence.

    Evaluation sets are enriched with defects so that recall can be measured
    with a usable confidence interval; costs are then computed at the plant's
    real prevalence (recall and FPR do not depend on prevalence).
    """
    rng = random.Random(seed)
    n_def = round(n * defect_rate)
    labels = [True] * n_def + [False] * (n - n_def)
    rng.shuffle(labels)
    return [make_part(rng, d) for d in labels]
