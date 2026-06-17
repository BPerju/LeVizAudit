"""Pure pattern-generation functions for the guided overlay.

All coordinates are pixel space on the configured camera image (see CLAUDE.md's
vision-only/no-FK rule) — never physical/world units. No I/O, no Rerun, no dataset
imports belong in this module.
"""

from __future__ import annotations

import math

from vizaudit.overlay.config import PatternConfig

Point = tuple[float, float]


def generate_arc_points(
    center: Point,
    radius: float,
    angle_start_deg: float,
    angle_end_deg: float,
    count: int,
) -> list[Point]:
    """Evenly spaced points along an arc, in pixel space.

    Angle 0 points along +x; increasing angle rotates toward +y (clockwise in image
    space, since image y grows downward). ``angle_start_deg``/``angle_end_deg`` of
    ``0``/``180`` is a semicircle below the center; ``0``/``360`` is a full circle.
    """
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    cx, cy = center
    if count == 1:
        angles_deg = [(angle_start_deg + angle_end_deg) / 2]
    else:
        step = (angle_end_deg - angle_start_deg) / (count - 1)
        angles_deg = [angle_start_deg + i * step for i in range(count)]
    return [
        (cx + radius * math.cos(math.radians(a)), cy + radius * math.sin(math.radians(a)))
        for a in angles_deg
    ]


def generate_line_points(start: Point, end: Point, count: int) -> list[Point]:
    """Evenly spaced points from ``start`` to ``end`` inclusive, in pixel space."""
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    sx, sy = start
    ex, ey = end
    if count == 1:
        return [((sx + ex) / 2, (sy + ey) / 2)]
    return [
        (sx + (ex - sx) * i / (count - 1), sy + (ey - sy) * i / (count - 1)) for i in range(count)
    ]


def build_pattern(pattern_config: PatternConfig, count: int) -> list[Point]:
    """Dispatch on ``pattern_config.shape``. The sole extension point for a future shape."""
    if pattern_config.shape == "arc":
        return generate_arc_points(
            center=pattern_config.center,
            radius=pattern_config.radius,
            angle_start_deg=pattern_config.angle_start_deg,
            angle_end_deg=pattern_config.angle_end_deg,
            count=count,
        )
    if pattern_config.shape == "line":
        return generate_line_points(
            start=pattern_config.start,
            end=pattern_config.end,
            count=count,
        )
    raise ValueError(f"Unknown pattern shape: {pattern_config.shape!r}")


def target_for_episode(points: list[Point], episode_index: int) -> Point:
    """Target point for ``episode_index``, wrapping if the session runs longer than
    the configured pattern length."""
    if not points:
        raise ValueError("points must be non-empty")
    return points[episode_index % len(points)]
