"""Generic 2D planar perspective correction from 4 operator-marked reference points.

Pure geometry -- no I/O, no Rerun, no dataset imports, no camera intrinsics/FK. This is a
plain projective homography between a "canonical" (rectified) rectangle and 4 pixel-space
points marking that same rectangle as seen by a possibly-angled camera -- not a 3D camera
model, consistent with the project's vision-only/no-FK rule (see CLAUDE.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

Point = tuple[float, float]


@dataclass(frozen=True)
class Homography:
    matrix: np.ndarray  # 3x3, maps canonical (u, v) -> pixel (x, y)


def canonical_rect_dims(corners: list[Point], aspect_ratio: float | None) -> tuple[float, float]:
    """Width/height of the canonical rectangle these corners are calibrated against.

    Defaults (``aspect_ratio=None``) to the corners' own average edge-length ratio, so
    canonical-space numbers numerically resemble pixel-space numbers for a near-top-down
    camera (the common case) -- only a genuinely tilted camera produces canonical dimensions
    that diverge much from the raw pixel quadrilateral's own size.
    """
    tl, tr, br, bl = corners
    top = math.hypot(tr[0] - tl[0], tr[1] - tl[1])
    bottom = math.hypot(br[0] - bl[0], br[1] - bl[1])
    left = math.hypot(bl[0] - tl[0], bl[1] - tl[1])
    right = math.hypot(br[0] - tr[0], br[1] - tr[1])
    avg_width = (top + bottom) / 2
    avg_height = (left + right) / 2
    if aspect_ratio is None:
        return avg_width, avg_height
    scale = math.sqrt(avg_width * avg_height)
    return scale * math.sqrt(aspect_ratio), scale / math.sqrt(aspect_ratio)


def compute_homography(corners: list[Point], aspect_ratio: float | None = None) -> Homography:
    """``corners``: exactly 4 pixel-space points, clockwise from top-left, marking a
    rectangular reference region on the (assumed planar) workspace surface."""
    if len(corners) != 4:
        raise ValueError(f"corners must have exactly 4 points, got {len(corners)}")

    width, height = canonical_rect_dims(corners, aspect_ratio)
    canonical: list[Point] = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]

    # Standard 4-point homography: 8 unknowns (h33 fixed at 1), 8 equations from the 4
    # correspondences. Solved directly (not least-squares) since 4 points exactly determine
    # the 8 degrees of freedom -- no need for SVD/DLT-over-N-points machinery.
    a = np.zeros((8, 8))
    b = np.zeros(8)
    for i, ((u, v), (x, y)) in enumerate(zip(canonical, corners)):
        a[2 * i] = [u, v, 1, 0, 0, 0, -u * x, -v * x]
        b[2 * i] = x
        a[2 * i + 1] = [0, 0, 0, u, v, 1, -u * y, -v * y]
        b[2 * i + 1] = y

    h = np.linalg.solve(a, b)
    matrix = np.array(
        [
            [h[0], h[1], h[2]],
            [h[3], h[4], h[5]],
            [h[6], h[7], 1.0],
        ]
    )
    return Homography(matrix=matrix)


def apply_homography(h: Homography, point: Point) -> Point:
    """Projective transform of a canonical point into pixel space."""
    x, y = point
    vec = h.matrix @ np.array([x, y, 1.0])
    return (float(vec[0] / vec[2]), float(vec[1] / vec[2]))


def invert_homography(h: Homography) -> Homography:
    """The pixel-to-canonical inverse of ``h`` -- ``apply_homography(invert_homography(h), p)``
    maps a pixel-space point back into canonical space."""
    return Homography(matrix=np.linalg.inv(h.matrix))
