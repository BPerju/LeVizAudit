"""Pure pattern-generation functions for the guided overlay.

All coordinates are pixel space on the configured camera image (see CLAUDE.md's
vision-only/no-FK rule) — never physical/world units. No I/O, no Rerun, no dataset
imports belong in this module.
"""

from __future__ import annotations

import math
import random
from typing import Callable, TypeVar

from vizaudit.overlay.config import ExcludeZoneConfig, PatternConfig
from vizaudit.overlay.perspective import Homography, apply_homography, invert_homography

Point = tuple[float, float]
T = TypeVar("T")

_GOLDEN_RATIO_CONJUGATE = (3 - math.sqrt(5)) / 2  # 1 - 1/phi, exact closed form
_ZONE_BOUNDARY_SEGMENTS = 32  # circle-zone -> polygon approximation, for canonical-space checks


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


def _point_in_polygon(point: Point, vertices: list[Point]) -> bool:
    """Standard ray-casting point-in-polygon test."""
    x, y = point
    inside = False
    n = len(vertices)
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_intersect:
                inside = not inside
    return inside


def _point_to_segment_distance(point: Point, a: Point, b: Point) -> float:
    px, py = point
    ax, ay = a
    bx, by = b
    abx, aby = bx - ax, by - ay
    length_sq = abx**2 + aby**2
    if length_sq == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / length_sq))
    proj_x, proj_y = ax + t * abx, ay + t * aby
    return math.hypot(px - proj_x, py - proj_y)


def _point_near_polygon(point: Point, vertices: list[Point], border_width: float) -> bool:
    """True if ``point`` is inside the polygon, or within ``border_width`` of any edge --
    i.e. the polygon "buffered" outward by border_width, the same margin concept applied to
    the workspace/circle boundary."""
    if _point_in_polygon(point, vertices):
        return True
    if border_width <= 0:
        return False
    n = len(vertices)
    return any(
        _point_to_segment_distance(point, vertices[i], vertices[(i + 1) % n]) < border_width
        for i in range(n)
    )


def _point_in_any_zone(
    point: Point, exclude_zones: list[ExcludeZoneConfig], border_width: float = 0.0
) -> bool:
    px, py = point
    for zone in exclude_zones:
        if zone.shape == "circle":
            zx, zy = zone.center
            if (px - zx) ** 2 + (py - zy) ** 2 <= (zone.radius + border_width) ** 2:
                return True
        else:  # "polygon"
            if _point_near_polygon(point, zone.vertices, border_width):
                return True
    return False


def _zone_to_canonical_polygon(zone: ExcludeZoneConfig, inverse_homography: Homography) -> list[Point]:
    """Approximates an exclude_zone as a polygon in canonical space, via the inverse
    homography -- a circle becomes a `_ZONE_BOUNDARY_SEGMENTS`-gon. Used only when a
    homography is active, so border_width (canonical-space) can be applied to exclude_zones
    in the same units as everything else, instead of being skipped for them entirely."""
    if zone.shape == "circle":
        cx, cy = zone.center
        boundary = [
            (
                cx + zone.radius * math.cos(2 * math.pi * i / _ZONE_BOUNDARY_SEGMENTS),
                cy + zone.radius * math.sin(2 * math.pi * i / _ZONE_BOUNDARY_SEGMENTS),
            )
            for i in range(_ZONE_BOUNDARY_SEGMENTS)
        ]
    else:  # "polygon"
        boundary = zone.vertices
    return [apply_homography(inverse_homography, p) for p in boundary]


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Standard sort-and-merge of possibly-overlapping ``(lo, hi)`` intervals into a
    sorted, non-overlapping list. Pure interval algebra -- no notion of pixels, degrees, or
    any particular space; used identically by the grid column scan (pixel/canonical y) and
    the radial angle scan (degrees)."""
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [ordered[0]]
    for lo, hi in ordered[1:]:
        last_lo, last_hi = merged[-1]
        if lo <= last_hi:
            merged[-1] = (last_lo, max(last_hi, hi))
        else:
            merged.append((lo, hi))
    return merged


def _subtract_intervals(
    base_lo: float, base_hi: float, excluded: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """``[base_lo, base_hi]`` minus every interval in ``excluded`` (must already be sorted
    and merged, e.g. via ``_merge_intervals``) -- returns the remaining valid sub-intervals,
    in order. An exclude_zone overlapping the middle of the base interval splits it into two
    pieces; this is exactly what lets a column/shell straddling an obstacle still place
    points correctly above and below (or inside and outside) it, instead of treating the
    whole column/shell as a single range and patching individual points after the fact."""
    valid = []
    cursor = base_lo
    for lo, hi in excluded:
        lo, hi = max(lo, base_lo), min(hi, base_hi)
        if hi <= cursor or lo >= base_hi:
            continue
        if lo > cursor:
            valid.append((cursor, lo))
        cursor = max(cursor, hi)
    if cursor < base_hi:
        valid.append((cursor, base_hi))
    return valid


def _inflate_polygon(vertices: list[Point], border_width: float) -> list[Point]:
    """Pushes every vertex outward from the polygon's centroid by ``border_width`` --- an
    approximation of the true Minkowski-sum offset (which would round the corners), good
    enough here because this is only used to size the *allocation* given to a column/shell
    near a polygon zone: the final placed point still goes through the exact
    ``_point_near_polygon`` check (via ``is_valid``/``_relocate_if_invalid``) before being
    accepted, so a slightly-off corner here costs at most a small relocation nudge, not a
    wrong final point."""
    if border_width <= 0:
        return vertices
    n = len(vertices)
    centroid_x = sum(v[0] for v in vertices) / n
    centroid_y = sum(v[1] for v in vertices) / n
    inflated = []
    for vx, vy in vertices:
        dx, dy = vx - centroid_x, vy - centroid_y
        d = math.hypot(dx, dy)
        if d < 1e-9:
            inflated.append((vx, vy))
            continue
        scale = (d + border_width) / d
        inflated.append((centroid_x + dx * scale, centroid_y + dy * scale))
    return inflated


def _polygon_vertical_intervals(x: float, vertices: list[Point]) -> list[tuple[float, float]]:
    """Standard scanline edge-crossing test: the y-sub-interval(s) where the vertical line
    at ``x`` is inside ``vertices``, found by collecting every edge crossing's y-value,
    sorting, and pairing consecutive crossings (the same parity rule ``_point_in_polygon``
    uses, just solved analytically for an entire line at once instead of one point at a
    time). Exact for convex shapes (rectangle/circle-as-N-gon cuts) and any well-formed
    simple polygon; vertical edges contribute no crossing (a vertical line can't cross
    another vertical line at a single point) and are skipped."""
    ys = []
    n = len(vertices)
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        if x1 == x2:
            continue
        if (x1 <= x <= x2) or (x2 <= x <= x1):
            t = (x - x1) / (x2 - x1)
            ys.append(y1 + t * (y2 - y1))
    ys.sort()
    return [(ys[i], ys[i + 1]) for i in range(0, len(ys) - 1, 2)]


def _excluded_y_intervals_at_x(
    x: float,
    exclude_zones: list[ExcludeZoneConfig],
    border_width: float,
    homography: Homography | None,
    canonical_zone_polygons: list[list[Point]] | None,
) -> list[tuple[float, float]]:
    """The merged y-sub-interval(s) at this ``x`` that fall inside any exclude_zone
    (buffered by ``border_width``) -- the grid fast path's per-column counterpart of
    ``is_valid``'s zone check, but for an entire column at once instead of one point. Circle
    zones get an exact quadratic chord formula; polygon zones (and every zone, when a
    homography is active and zones are pre-approximated as canonical-space polygons) go
    through ``_inflate_polygon`` + ``_polygon_vertical_intervals``."""
    raw: list[tuple[float, float]] = []
    if homography is not None:
        for poly in canonical_zone_polygons or []:
            raw.extend(_polygon_vertical_intervals(x, _inflate_polygon(poly, border_width)))
    else:
        for zone in exclude_zones:
            if zone.shape == "circle":
                zx, zy = zone.center
                eff_r = zone.radius + border_width
                dx = x - zx
                if abs(dx) <= eff_r:
                    half = math.sqrt(eff_r * eff_r - dx * dx)
                    raw.append((zy - half, zy + half))
            else:
                raw.extend(_polygon_vertical_intervals(x, _inflate_polygon(zone.vertices, border_width)))
    return _merge_intervals(raw)


def _segment_circle_intersection_angles(
    a: Point, b: Point, cx: float, cy: float, r: float
) -> list[float]:
    """Angles (degrees, mod 360) where segment ``a``-``b`` crosses the circle of radius
    ``r`` centered at ``(cx, cy)`` -- a standard line-circle intersection, solved as a
    quadratic in the segment's parametric ``s`` (clamped to ``[0, 1]``)."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    ex, ey = ax - cx, ay - cy
    coeff_a = dx * dx + dy * dy
    if coeff_a < 1e-12:
        return []
    coeff_b = 2 * (ex * dx + ey * dy)
    coeff_c = ex * ex + ey * ey - r * r
    disc = coeff_b * coeff_b - 4 * coeff_a * coeff_c
    if disc < 0:
        return []
    sq = math.sqrt(disc)
    angles = []
    for s in ((-coeff_b - sq) / (2 * coeff_a), (-coeff_b + sq) / (2 * coeff_a)):
        if -1e-9 <= s <= 1 + 1e-9:
            s_clamped = min(1.0, max(0.0, s))
            px, py = ax + s_clamped * dx, ay + s_clamped * dy
            angles.append(math.degrees(math.atan2(py - cy, px - cx)) % 360)
    return angles


def _polygon_angle_block(r: float, cx: float, cy: float, vertices: list[Point]) -> list[tuple[float, float]]:
    """The angular interval(s) (degrees, each as ``(lo, hi)`` with ``hi`` possibly > 360 to
    represent a piece that wraps past 0 -- normalize via ``_merge_angle_intervals``) where
    the circle of radius ``r`` centered at ``(cx, cy)`` is inside ``vertices``. Finds every
    edge crossing of that circle (``_segment_circle_intersection_angles``), then -- instead
    of reasoning about polygon winding/parity directly, which gets fiddly for a possibly
    non-convex cut -- tests the midpoint angle of each arc between consecutive crossings
    with the already-correct, already-tested ``_point_in_polygon``."""
    if r <= 0:
        return [(0.0, 360.0)] if _point_in_polygon((cx, cy), vertices) else []
    crossings: list[float] = []
    n = len(vertices)
    for i in range(n):
        crossings.extend(_segment_circle_intersection_angles(vertices[i], vertices[(i + 1) % n], cx, cy, r))
    if not crossings:
        sample = (cx + r, cy)
        return [(0.0, 360.0)] if _point_in_polygon(sample, vertices) else []
    ordered = sorted(set(round(a, 9) for a in crossings))
    blocked = []
    n_c = len(ordered)
    for i in range(n_c):
        lo = ordered[i]
        hi = ordered[i + 1] if i + 1 < n_c else ordered[0] + 360
        mid_deg = ((lo + hi) / 2) % 360
        px, py = cx + r * math.cos(math.radians(mid_deg)), cy + r * math.sin(math.radians(mid_deg))
        if _point_in_polygon((px, py), vertices):
            blocked.append((lo, hi))
    return blocked


def _circle_zone_angle_block(
    r: float, cx: float, cy: float, zx: float, zy: float, rz_eff: float
) -> list[tuple[float, float]]:
    """The angular interval (degrees, possibly wrapping -- see ``_polygon_angle_block``)
    where the circle of radius ``r`` centered at ``(cx, cy)`` is inside the zone disk of
    radius ``rz_eff`` centered at ``(zx, zy)``. Closed-form (the standard "angular extent of
    a circle as seen at a given radius" formula, via the law of cosines on the triangle
    formed by ``center``, the zone's center, and an intersection point), with the three
    degenerate cases (zone concentric with ``center``, the radius-``r`` circle entirely
    inside the zone, no overlap at all) handled directly rather than relying on ``acos``
    clamping to silently do the right thing."""
    d = math.hypot(zx - cx, zy - cy)
    if d < 1e-9:
        return [(0.0, 360.0)] if r <= rz_eff else []
    if r <= 0:
        return [(0.0, 360.0)] if d <= rz_eff else []
    if d + r <= rz_eff:
        return [(0.0, 360.0)]
    if d >= r + rz_eff or r >= d + rz_eff:
        return []
    cos_half = (r * r + d * d - rz_eff * rz_eff) / (2 * r * d)
    cos_half = max(-1.0, min(1.0, cos_half))
    half_deg = math.degrees(math.acos(cos_half))
    theta_zone = math.degrees(math.atan2(zy - cy, zx - cx))
    return [(theta_zone - half_deg, theta_zone + half_deg)]


def _merge_angle_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Like ``_merge_intervals``, but for angular intervals that live on a circle (degrees)
    instead of a line: an interval is allowed to wrap past 360 (e.g. a zone straddling the
    0-degree reference), so a naive linear merge would wrongly treat a piece like
    ``(350, 370)`` as disjoint from one at ``(0, 10)`` even though they're the same arc.
    Standard fix: duplicate every interval at the -360/+360 offsets too, merge on the
    resulting extended real line (where wraparound is no longer ambiguous), then clip back
    down to ``[0, 360)``, splitting any piece that still straddles an edge."""
    if not intervals:
        return []
    expanded = []
    for lo, hi in intervals:
        lo_mod = lo % 360
        span = hi - lo
        expanded.append((lo_mod, lo_mod + span))
    shifted = [
        (lo + offset, hi + offset) for lo, hi in expanded for offset in (-360, 0, 360)
    ]
    merged = _merge_intervals(shifted)
    clipped = []
    for lo, hi in merged:
        clipped_lo, clipped_hi = max(lo, 0.0), min(hi, 360.0)
        if clipped_hi > clipped_lo:
            clipped.append((clipped_lo, clipped_hi))
    return _merge_intervals(clipped)


def _clamp_to_region(
    candidate: Point,
    cx: float,
    cy: float,
    effective_inner: float,
    effective_outer: float,
    angle_start_deg: float,
    angle_end_deg: float,
    bounds: tuple[float, float] | None,
    border_width: float,
) -> Point:
    """Projects ``candidate`` back into the sector/bounds region via simple, closed-form
    geometric clamps: radius into ``[effective_inner, effective_outer]``, angle into
    ``[angle_start_deg, angle_end_deg]``, then -- if still outside ``bounds`` -- shrunk
    toward ``center`` *along the same angle* until inside it. Exact and O(1) (a fixed number
    of shrink steps, no per-point search). Does NOT know about ``exclude_zones`` -- an
    arbitrary polygon has no simple closed-form projection, so callers fall back to a local
    search for that case (see ``_relocate_if_invalid``).

    The ``bounds`` step deliberately shrinks along the ray from ``center`` rather than
    clamping x/y independently (the obvious alternative). Axis-clamping collapses every
    point whose ideal x exceeds the bound onto the exact same vertical line regardless of its
    own y -- for a lattice, many *different* columns share the same row positions, so they
    landed on literally the same points after clamping (a real bug, found by checking actual
    output coordinates for exact duplicates, not just visually). Shrinking along each point's
    own angle keeps every point at its own distinct angle, so two different ideal points only
    ever coincide after this projection in the genuinely degenerate case where they shared
    that angle already."""
    dx, dy = candidate[0] - cx, candidate[1] - cy
    r = math.hypot(dx, dy)
    theta_deg = math.degrees(math.atan2(dy, dx)) % 360
    angle_span = angle_end_deg - angle_start_deg
    if angle_span < 360:
        rel = (theta_deg - angle_start_deg) % 360
        if rel > angle_span:
            # Outside the pie slice -- snap to whichever boundary (start or end) is nearer.
            theta_deg = angle_start_deg if rel - angle_span > 360 - rel else angle_end_deg
    r = max(effective_inner, min(effective_outer, r))
    theta_rad = math.radians(theta_deg)
    cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)
    x, y = cx + r * cos_t, cy + r * sin_t

    if bounds is not None:
        bw, bh = bounds

        def in_bounds(px: float, py: float) -> bool:
            return border_width <= px <= bw - border_width and border_width <= py <= bh - border_width

        if not in_bounds(x, y):
            if in_bounds(cx, cy):
                # Binary search for the largest valid radius along this ray -- `bounds` is
                # convex and `center` is inside it, so validity along the ray is a single
                # contiguous [0, r_boundary] interval, making this exact (to float
                # precision). A small fixed list of shrink fractions was tried first and
                # rejected: many different points all land on the SAME first-valid fraction,
                # which just trades one collapse (the axis-clamp wall) for another (a
                # quantized ring of duplicate/near-duplicate points) -- found, again, by
                # checking actual output coordinates, not by eye.
                lo, hi = 0.0, r
                for _ in range(30):
                    mid = (lo + hi) / 2
                    if in_bounds(cx + mid * cos_t, cy + mid * sin_t):
                        lo = mid
                    else:
                        hi = mid
                x, y = cx + lo * cos_t, cy + lo * sin_t
            else:
                # Degenerate: even `center` itself is outside bounds, so there's no
                # "shrink toward center" direction that helps -- fall back to a plain
                # per-axis clamp (loses angular distinctness, but only in this rare case).
                x = max(border_width, min(bw - border_width, x))
                y = max(border_width, min(bh - border_width, y))
    return (x, y)


def _allocate_shares(sizes: list[float], count: int) -> list[int]:
    """Largest-remainder apportionment (the standard method for splitting a fixed total
    across groups proportionally to size, e.g. parliamentary seat allocation): splits
    ``count`` picks across groups proportionally to ``sizes``, every share within +/-1 of its
    exact proportional value, summing to exactly ``count``. Used by ``"grid"``'s full-disk
    fast path to give each lattice column a point count proportional to its own continuous
    valid-height, before placing that many points evenly spaced within it -- unlike an
    earlier version of this idea, this never selects *from* a pre-built discrete candidate
    list, so there's no separate "thins short groups more than long ones" failure mode; the
    only residual imprecision is the +/-1 rounding here, which is small relative to `count`
    for any practical column count."""
    total = sum(sizes)
    exact = [count * s / total for s in sizes]
    shares = [int(e) for e in exact]
    remainder = count - sum(shares)
    order = sorted(range(len(sizes)), key=lambda i: exact[i] - shares[i], reverse=True)
    for i in order[:remainder]:
        shares[i] += 1
    return shares


def _occupancy_guard(
    is_valid: Callable[[Point], bool], min_separation: float = 1e-6
) -> tuple[Callable[[Point], bool], Callable[[Point], None]]:
    """Wraps ``is_valid`` so a point within ``min_separation`` of one already returned by a
    previous call also counts as invalid -- returns ``(guarded_is_valid, mark_placed)``;
    callers must invoke ``mark_placed`` on every point they actually keep.

    Two cases need this, both involving relocation, never the fast-path's direct placement:
    (1) ``grid``'s lattice rows/columns occasionally place two *different* ideal cells
    exactly co-radial with `center` (e.g. two columns on the row through `center`'s own y
    share angle 0/180, or two rows of the column through `center`'s own x share angle
    90/270) -- when both need relocating for the same reason, a ray-based search finds a
    result for each independently, with nothing stopping them from landing implausibly
    close together (not just exactly equal) if their starting points were close along that
    shared ray. (2) a small ``min_separation`` (the default) only catches literal
    coincidences; a real report -- several points relocated off a shared central obstacle
    landing a small fraction of the expected spacing apart, clearly visually distinct from
    a "missing point" but still visibly clumped -- needs ``min_separation`` set to a
    fraction of the expected inter-point spacing instead, so a relocation search keeps
    looking past merely "different" into "different enough to blend in."

    Doesn't need a spatial index: even a few hundred points is a trivial O(N) scan per
    check."""
    placed: list[Point] = []

    def guarded(candidate: Point) -> bool:
        if not is_valid(candidate):
            return False
        return all(math.hypot(candidate[0] - p[0], candidate[1] - p[1]) >= min_separation for p in placed)

    def mark_placed(point: Point) -> None:
        placed.append(point)

    return guarded, mark_placed


def _relocate_if_invalid(
    candidate: Point,
    is_valid: Callable[[Point], bool],
    clamp: Callable[[Point], Point],
    rng: random.Random,
    search_scale: float,
    shape_label: str,
    index: int,
    center: Point,
    effective_outer: float,
) -> Point:
    """Returns ``candidate`` unchanged if it's already valid. Otherwise tries, in order:

    1. The closed-form ``clamp`` (resolves the common radius/angle/bounds violations
       exactly, with no iteration and no effect on any other point).
    2. A radial search along the point's OWN angle from ``center`` -- the remaining
       violation at this point is an ``exclude_zones`` cut (no simple closed-form
       projection exists for an arbitrary polygon), but a cut is usually small relative to
       the whole disk, so moving outward (then inward) at the point's *own* angle very
       often clears it. This is what keeps points relocated away from a *central* obstacle
       (e.g. marking out the robot's own base) fanned out at their original angles, matching
       the surrounding pattern's structure -- skipping straight to unstructured random
       jitter here was a real, reported bug: independently-jittered points landing near a
       shared central obstacle clumped at essentially random angles relative to each other,
       with neighboring-index points sometimes landing within a fraction of a degree of each
       other and leaving 50+ degree gaps elsewhere, instead of preserving the spiral/lattice's
       own spacing around the obstacle.
    3. A local random search of growing radius around the clamped point, only reached if
       the radial search also fails (e.g. the cut blocks the entire ray, or `center` is
       exactly at the clamped point so there's no ray to follow).

    This relocates exactly the one invalid point to a place that actually satisfies every
    constraint, instead of discarding it from a shared candidate pool: discarding is what
    previously forced a population-level trim back down to ``count``, and that trim is what
    thinned some regions of the pattern much more than others (the "missing points" bug) --
    relocating individual points needs no trim at all, since every index still produces
    exactly one point."""
    if is_valid(candidate):
        return candidate
    clamped = clamp(candidate)
    if is_valid(clamped):
        return clamped
    cx, cy = center
    dx, dy = clamped[0] - cx, clamped[1] - cy
    r0 = math.hypot(dx, dy)
    if r0 > 1e-9:
        cos_t, sin_t = dx / r0, dy / r0
        step = max(search_scale * 0.1, 1e-9)
        for direction in (1, -1):
            r = r0
            for _ in range(200):
                r += direction * step
                if r > effective_outer or r < 0:
                    break
                probe = (cx + r * cos_t, cy + r * sin_t)
                if is_valid(probe):
                    return probe
    for attempt in range(200):
        radius = search_scale * (0.5 + 0.25 * (attempt // 3))
        probe = (clamped[0] + rng.uniform(-radius, radius), clamped[1] + rng.uniform(-radius, radius))
        if is_valid(probe):
            return probe
    raise ValueError(
        f"Could not find a valid position near {shape_label} point index {index} "
        f"(ideal={candidate}) -- exclude_zones/border_width/bounds are likely too "
        f"restrictive for the available sector area"
    )


def _validate_no_excluded_points(
    points: list[Point], exclude_zones: list[ExcludeZoneConfig], shape_label: str
) -> None:
    """arc/line are deterministic -- a point landing in an exclusion zone has no alternate
    point to substitute, so this is a config error rather than something to resample."""
    if not exclude_zones:
        return
    for i, point in enumerate(points):
        if _point_in_any_zone(point, exclude_zones):
            raise ValueError(
                f"{shape_label} pattern point index {i} {point} falls inside an exclude_zones "
                f"entry; adjust this pattern's center/radius/angles/start/end, or the "
                f"conflicting exclude_zones entry, since {shape_label} points cannot be "
                f"resampled automatically"
            )


_RADIAL_CDF_SAMPLES = 300


def _outside_bounds_angle_intervals(
    r: float, cx: float, cy: float, bounds: tuple[float, float], border_width: float
) -> list[tuple[float, float]]:
    """The angular interval(s) (degrees, possibly wrapping) where the circle of radius ``r``
    centered at ``(cx, cy)`` falls OUTSIDE the ``bounds`` rectangle (inset by
    ``border_width``) -- the bounds analogue of ``_circle_zone_angle_block``/
    ``_polygon_angle_block``, reusing the latter by passing the inset rectangle itself as the
    "zone" (giving the *inside* intervals) and then taking the complement within ``[0,
    360)``, since what we want here is the opposite: where the circle leaves the rectangle."""
    bw, bh = bounds
    x_lo, x_hi = border_width, bw - border_width
    y_lo, y_hi = border_width, bh - border_width
    if x_hi <= x_lo or y_hi <= y_lo:
        return [(0.0, 360.0)]  # no valid rectangle at all -- every angle is "outside"
    rect = [(x_lo, y_lo), (x_hi, y_lo), (x_hi, y_hi), (x_lo, y_hi)]
    inside_raw = _polygon_angle_block(r, cx, cy, rect)
    if not inside_raw:
        return [(0.0, 360.0)]
    return _subtract_intervals(0.0, 360.0, _merge_angle_intervals(inside_raw))


def _available_angle_intervals_at_radius(
    r: float,
    cx: float,
    cy: float,
    angle_start_deg: float,
    angle_end_deg: float,
    exclude_zones: list[ExcludeZoneConfig],
    border_width: float,
    homography: Homography | None,
    canonical_zone_polygons: list[list[Point]] | None,
    bounds: tuple[float, float] | None,
) -> list[tuple[float, float]]:
    """The angular sub-interval(s) within ``[angle_start_deg, angle_end_deg]`` where the
    circle of radius ``r`` centered at ``(cx, cy)`` is actually open -- not inside any
    ``exclude_zones`` (buffered by ``border_width``) and not outside ``bounds``. Used both by
    ``_radial_available_area_cdf`` (which only needs the total length, to decide how many
    points a radius shell gets) and, per point, by ``generate_sector_points``'s ``"radial"``
    branch (which needs the actual intervals, to place that point's angle somewhere genuinely
    open at its own assigned radius -- see the docstring on the radial branch's call site for
    why this is necessary and not just an optimization)."""
    full_span = angle_end_deg - angle_start_deg
    blocked_raw: list[tuple[float, float]] = []
    if homography is not None:
        for poly in canonical_zone_polygons or []:
            blocked_raw.extend(_polygon_angle_block(r, cx, cy, _inflate_polygon(poly, border_width)))
    else:
        for zone in exclude_zones:
            if zone.shape == "circle":
                zx, zy = zone.center
                blocked_raw.extend(_circle_zone_angle_block(r, cx, cy, zx, zy, zone.radius + border_width))
            else:
                blocked_raw.extend(_polygon_angle_block(r, cx, cy, _inflate_polygon(zone.vertices, border_width)))
    if bounds is not None:
        blocked_raw.extend(_outside_bounds_angle_intervals(r, cx, cy, bounds, border_width))
    if not blocked_raw:
        return [(angle_start_deg, angle_end_deg)]
    blocked_merged = _merge_angle_intervals(blocked_raw)
    valid = _subtract_intervals(angle_start_deg, angle_end_deg, blocked_merged)
    return valid if valid else []


def _place_in_available_intervals(intervals: list[tuple[float, float]], fraction: float) -> float | None:
    """Maps ``fraction`` (in ``[0, 1)``) to a point within ``intervals`` (a list of disjoint
    ``(lo, hi)`` ranges), positioned proportionally to where ``fraction`` falls within their
    *concatenated* total length -- e.g. with two equal-length intervals, fraction ``0.75``
    lands a quarter of the way into the second one. Returns ``None`` if ``intervals`` is empty
    or has zero total length (nowhere to place the point at all at this radius)."""
    total = sum(hi - lo for lo, hi in intervals)
    if total <= 0:
        return None
    target = max(0.0, min(total, fraction * total))
    cursor = 0.0
    for lo, hi in intervals:
        length = hi - lo
        if target <= cursor + length:
            return lo + (target - cursor)
        cursor += length
    return intervals[-1][1]


def _radial_available_area_cdf(
    cx: float,
    cy: float,
    effective_inner: float,
    effective_outer: float,
    angle_start_deg: float,
    angle_end_deg: float,
    exclude_zones: list[ExcludeZoneConfig],
    border_width: float,
    homography: Homography | None,
    canonical_zone_polygons: list[list[Point]] | None,
    bounds: tuple[float, float] | None,
) -> tuple[list[float], list[float]]:
    """Builds a lookup table of (radius, cumulative *available* area) by numerically
    integrating, over ``_RADIAL_CDF_SAMPLES`` radius samples, the angular span actually open
    at each radius -- i.e. the full sector angle minus whatever ``exclude_zones`` (buffered
    by ``border_width``) block at that radius, via ``_circle_zone_angle_block``/
    ``_polygon_angle_block``, AND minus whatever falls outside ``bounds`` (the workspace
    rectangle), via ``_outside_bounds_angle_intervals``. ``_invert_radial_cdf`` then turns
    this into "the radius at which X% of the available area has been covered," which is what
    lets ``"radial"``'s point-radius formula skip past a mostly-blocked radius band instead of
    assigning it the same share of points as an unobstructed one.

    This is the radial analogue of the grid fast path's per-column ``_subtract_intervals``
    call -- both compute the TRUE available extent (a column's y-range there, a shell's
    angular span here) before deciding how many points that region gets, instead of
    assigning points by the unobstructed formula and patching whatever lands in a zone
    afterward. That patch-after approach was a real, reported regression here specifically:
    every point whose *unobstructed* ideal radius fell inside a central zone's radius range
    relocated outward along its own angle (the right move for an individual point), but
    since that radius range can hold many points' worth of unobstructed share, they all
    landed in a thin ring just past the zone's edge -- 13 of 60 points within 0.6 units of
    the boundary in one measured case, instead of spread across the whole remaining disk.

    ``bounds`` needed the exact same treatment, but was missed in the first pass at this fix
    (which only covered ``exclude_zones``) -- found from a follow-up report that radial was
    "just as broken" even after that fix, since grid's fast path had ALREADY had a working
    bounds-intersection (the per-column ``x_min``/``x_max``/``y_min``/``y_max`` clipping from
    an earlier round) while radial never got an equivalent: it had always relied on
    individual-point relocation for a bounds violation, which is exactly the same "ring
    pileup" failure mode as the zone case, just for the workspace rectangle's edge instead of
    a zone's. Measured: a workspace rectangle clipping roughly a third of the circle away gave
    a nearest-neighbor ratio of 0.37 for radial with NO zone involved at all (vs. grid's 0.97
    under the identical clipping), confirming this was the dominant remaining defect, not the
    zone handling.
    """
    n = _RADIAL_CDF_SAMPLES
    radii = [effective_inner + (effective_outer - effective_inner) * i / n for i in range(n + 1)]
    available_span = []
    for r in radii:
        valid = _available_angle_intervals_at_radius(
            r, cx, cy, angle_start_deg, angle_end_deg, exclude_zones, border_width,
            homography, canonical_zone_polygons, bounds,
        )
        available_span.append(sum(hi - lo for lo, hi in valid))
    cumulative = [0.0]
    for i in range(1, len(radii)):
        r0, r1 = radii[i - 1], radii[i]
        area0, area1 = available_span[i - 1] * r0, available_span[i] * r1
        cumulative.append(cumulative[-1] + 0.5 * (area0 + area1) * (r1 - r0))
    return radii, cumulative


def _invert_radial_cdf(radii: list[float], cumulative: list[float], target_fraction: float) -> float:
    """The radius at which the cumulative available-area table reaches ``target_fraction``
    of its total, via linear interpolation between the two bracketing samples."""
    total = cumulative[-1]
    if total <= 0:
        raise ValueError(
            "Could not place radial point(s) -- exclude_zones/border_width leave no "
            "available area anywhere in the sector"
        )
    target = target_fraction * total
    for i in range(1, len(cumulative)):
        if cumulative[i] >= target:
            c0, c1 = cumulative[i - 1], cumulative[i]
            r0, r1 = radii[i - 1], radii[i]
            if c1 == c0:
                return r1
            t = (target - c0) / (c1 - c0)
            return r0 + t * (r1 - r0)
    return radii[-1]


def _candidate_thetas_in_intervals(intervals: list[tuple[float, float]], num_samples: int) -> list[float]:
    """``num_samples`` angles evenly spaced across the concatenated ``intervals`` (NOT a
    uniform sample over the full ``[0, 360)`` range) -- the candidate set
    ``_refine_radial_local_separation`` searches for a better angle within."""
    candidates = []
    for k in range(num_samples):
        theta = _place_in_available_intervals(intervals, (k + 0.5) / num_samples)
        if theta is not None:
            candidates.append(theta)
    return candidates


def _refine_radial_local_separation(
    placed: list[Point],
    cx: float,
    cy: float,
    angle_start_deg: float,
    angle_end_deg: float,
    exclude_zones: list[ExcludeZoneConfig],
    border_width: float,
    homography: Homography | None,
    canonical_zone_polygons: list[list[Point]] | None,
    bounds: tuple[float, float] | None,
    expected_spacing: float,
    is_valid: Callable[[Point], bool],
    passes: int = 3,
    min_acceptable_ratio: float = 0.75,
    num_candidates: int = 24,
) -> list[Point]:
    """Post-process pass, in canonical space, that nudges any point too close to its
    nearest neighbor toward a better angle -- at the SAME radius, since that's already
    correct in aggregate via the area CDF -- within whatever's actually open there.

    Why this exists: mapping each point's golden-angle-derived fraction into the available
    angular sub-interval at its own radius (see the docstring on the per-point placement
    above) correctly preserves the *aggregate* density per radius shell, but does NOT
    preserve golden angle's strong guarantee that any two points stay well-separated --
    that guarantee belongs to the raw, unmapped ``i * golden_step`` sequence specifically,
    and is not automatically inherited by a *different* sequence (the per-point available
    interval, which varies continuously with radius near any asymmetric restriction)
    derived from mapping it through a fraction. Two points at different radii, with
    different raw fractions, can still coincidentally map to nearly the same absolute angle.
    Reported as "less consistent results" once `border_width` was added on top of an
    already-asymmetric `bounds` clip: nearest-neighbor ratio swung non-monotonically between
    0.57 and 0.89 across a `border_width` sweep on the exact same (self-similarly scaled)
    clipped-circle shape -- not a gradual, predictable degradation, but a sensitive
    dependence on exactly how the fixed golden-angle phase happens to land relative to
    whatever the clipping boundary looks like at that specific `border_width`.

    For each point below ``min_acceptable_ratio`` of the placed set's OWN average
    nearest-neighbor distance (not the theoretical ``expected_spacing``, which is the
    tightest-possible-packing estimate and is typically well below what an actual
    placement achieves -- using it as the threshold here meant the check almost never
    fired, since pairs at exactly the realistic average distance were still well above that
    theoretical floor), after up to ``passes`` sweeps (stopping early once a sweep improves
    nothing): try ``num_candidates`` alternative angles spread across the available interval
    at that point's own (unchanged) radius, and keep whichever maximizes the point's distance
    to every OTHER current point. This is a bounded, deterministic local search -- not a full
    relaxation -- so it only touches points that are actually too close, and never moves a
    point's radius (which would undo the CDF's area-correctness). ``expected_spacing`` is
    still used as a floor under degenerate inputs (e.g. ``count`` of 1 or 2, where "average
    nearest-neighbor distance" is a single value with no useful spread of its own)."""
    n = len(placed)
    if n < 2:
        return placed
    for _ in range(passes):
        nn_distances = [
            min(math.hypot(placed[i][0] - placed[j][0], placed[i][1] - placed[j][1]) for j in range(n) if j != i)
            for i in range(n)
        ]
        avg_nn = sum(nn_distances) / n
        min_separation = min_acceptable_ratio * max(avg_nn, expected_spacing)
        improved_any = False
        for i in range(n):
            xi, yi = placed[i]
            best_d = min(math.hypot(xi - placed[j][0], yi - placed[j][1]) for j in range(n) if j != i)
            if best_d >= min_separation:
                continue
            r = math.hypot(xi - cx, yi - cy)
            avail = _available_angle_intervals_at_radius(
                r, cx, cy, angle_start_deg, angle_end_deg, exclude_zones, border_width,
                homography, canonical_zone_polygons, bounds,
            )
            best_theta, best_score = None, best_d
            for theta in _candidate_thetas_in_intervals(avail, num_candidates):
                cand_x = cx + r * math.cos(math.radians(theta))
                cand_y = cy + r * math.sin(math.radians(theta))
                # `avail` is approximate (a polygon zone's border is buffered by pushing
                # vertices outward from its centroid, not a true Minkowski offset) -- the
                # exact check is what catches an under-buffered edge a candidate from `avail`
                # alone could otherwise still violate.
                if not is_valid((cand_x, cand_y)):
                    continue
                score = min(
                    math.hypot(cand_x - placed[j][0], cand_y - placed[j][1]) for j in range(n) if j != i
                )
                if score > best_score:
                    best_score, best_theta = score, theta
            if best_theta is not None:
                placed[i] = (cx + r * math.cos(math.radians(best_theta)), cy + r * math.sin(math.radians(best_theta)))
                improved_any = True
        if not improved_any:
            break
    return placed


def _search_variable_density(make_points: Callable[[int], list[Point]], target_count: int, max_iters: int = 8) -> list[Point]:
    """Re-tries ``make_points(density)`` at increasing/decreasing density until its survivor
    count is as close to ``target_count`` as achievable, instead of accepting whatever a
    single density (``density == target_count``) happens to produce. Keeps the best (closest
    to target) result seen across iterations."""
    density = target_count
    best: list[Point] | None = None
    for _ in range(max_iters):
        pts = make_points(max(1, density))
        if best is None or abs(len(pts) - target_count) < abs(len(best) - target_count):
            best = pts
        if len(pts) >= target_count:
            break
        if len(pts) == 0:
            density *= 2
        else:
            density = max(density + 1, round(density * target_count / len(pts)))
    return best or []


def generate_sector_points(
    center: Point,
    inner_radius: float,
    outer_radius: float,
    angle_start_deg: float,
    angle_end_deg: float,
    count: int,
    seed: int,
    distribution: str = "random",
    exclude_zones: list[ExcludeZoneConfig] | None = None,
    homography: Homography | None = None,
    border_width: float = 0.0,
    bounds: tuple[float, float] | None = None,
    count_mode: str = "fixed",
) -> list[Point]:
    """Points filling a pie-slice/annular-sector (NOT just its boundary).

    ``center``/``inner_radius``/``outer_radius`` are in pixel space when ``homography`` is
    None (today's behavior, implicitly assuming a top-down camera), or in canonical/rectified
    space when a ``homography`` is given -- area math (sampling, ``border_width``, ``bounds``)
    is only correct in a space where circles are actually circles, which a pixel-space circle
    is not under a tilted camera. ``exclude_zones`` are always pixel-space facts about the
    image, so they're checked against the final point, after any homography mapping.

    ``distribution``: ``"random"`` (area-uniform via ``r = sqrt(uniform(r_in**2, r_out**2))``,
    the original behavior), ``"grid"`` (a near-square Cartesian lattice, see below), or
    ``"radial"`` (a deterministic Fermat/Vogel spiral, see below). `config.py` defaults YAML
    configs to ``"grid"`` (evenly spaced coverage is more legible than a random scatter for
    auditing purposes), but this function's own default stays ``"random"`` so existing direct
    callers are unaffected.

    Both ``"grid"`` and ``"radial"`` compute exactly ``count`` *ideal* positions -- never
    more, never fewer -- instead of an earlier design that over-generated a candidate pool
    and trimmed it back down to ``count``. Trimming a *population* necessarily thins some
    regions of the pattern more than others whenever rejection itself is uneven across the
    region (a lattice column near the circle's edge has fewer valid candidates to begin with
    than one through the center), and at low ``count`` (this tool's actual operating range --
    tens of points, not thousands) the resulting integer rounding is a large *relative*
    error, visible as patches that look like points are missing entirely.

    ``"grid"`` has two code paths. The common one -- a full disk, i.e. ``inner_radius == 0``
    and a 360-degree span, which is always true for the calibration tool and the typical YAML
    case -- computes each lattice column's valid y-*range* directly from the circle's chord
    at that x (intersected with ``bounds``), then -- if any ``exclude_zones`` cross that
    column -- splits the range into the sub-intervals that remain after subtracting the
    zone(s) (``_excluded_y_intervals_at_x`` + ``_subtract_intervals``), so a column straddling
    an obstacle becomes two independent ranges instead of one. ``count`` is allocated across
    every column/sub-interval proportionally to its own length (``_allocate_shares``, the
    largest-remainder apportionment method), then placed evenly spaced *within* each
    continuous range. No discrete candidates are ever generated or discarded, so there's
    nothing to trim unevenly -- and no two columns can ever collide, unlike an earlier
    version of this path (see below). Computing each column/sub-interval's TRUE available
    length up front (instead of the unobstructed length, patched by relocating individual
    points that land in a zone afterward) is what makes the whole pattern's density actually
    adapt to an obstacle: a column losing half its length to a zone gets half its previous
    share, so the points it loses are absorbed as very slightly tighter spacing everywhere
    else, not as a dense, structureless clump right at the zone's edge -- which is what the
    previous (relocate-only) design did, confirmed by checking actual output coordinates, not
    by eye. An annulus or restricted pie slice (rarer; the calibration tool never produces
    one) falls back to a near-square lattice over the bounding square with each cell
    individually relocated if invalid, via ``_relocate_if_invalid`` -- simpler to get right
    than an exact per-column annulus/sector chord formula, at the cost of a small residual
    collision risk the fast path doesn't have, and without this column-range zone-subtraction
    (a zone-induced clump is still possible in this rarer path).

    ``"radial"`` computes each spiral point's angle directly from its own index
    (``theta_i = i * golden_angle``, never colliding angle-wise), but its radius comes from
    inverting a numerically-built *available-area* CDF (``_radial_available_area_cdf`` /
    ``_invert_radial_cdf``) whenever ``exclude_zones`` is non-empty, rather than the plain
    closed-form ``r = sqrt(effective_inner**2 + (effective_outer**2-effective_inner**2)*t)``
    used when there's nothing to avoid. The CDF integrates, over a few hundred radius
    samples, the angular span actually open at each radius (full span minus whatever
    ``_circle_zone_angle_block``/``_polygon_angle_block`` find blocked there) -- so a radius
    band that's mostly or entirely inside a central zone is assigned proportionally fewer (or
    zero) points up front, instead of being assigned the unobstructed share and then having
    every one of those points individually relocated outward along its own angle. That
    relocate-only behavior was a real, reported regression: with a central zone, every point
    whose unobstructed ideal radius fell inside it relocated to approximately the same
    radius (just past the zone's edge), producing a dense ring there -- 13 of 60 points within
    0.6 units of the boundary in one measured case -- instead of spreading across the whole
    remaining disk. Each computed (r, theta) pair still goes through the same
    ``_relocate_if_invalid`` as a safety net (e.g. an off-center zone can still block only
    part of a given radius's angular span, which the CDF -- a function of radius alone --
    doesn't capture), but it's now rarely needed and rarely has to move a point far.

    ``_relocate_if_invalid`` tries a closed-form clamp first (radius/angle/``bounds``, see
    ``_clamp_to_region`` -- the ``bounds`` step specifically shrinks along the point's own
    ray from ``center`` rather than clamping x/y independently, since an axis clamp collapses
    every point whose ideal x exceeds the bound onto the exact same vertical line regardless
    of its own y), then falls back to a local random search only for an ``exclude_zones`` cut
    (no simple closed-form projection exists for an arbitrary polygon). ``_occupancy_guard``
    additionally rejects a relocation landing on a point already placed by an earlier index --
    needed because the annulus/sector fallback path (and, in principle, any two points that
    happen to share an angle from ``center``) can otherwise have two different ideal positions
    shrink to the literal same nearest boundary point.

    ``border_width``: a margin (in the same space as ``center``) within which points never
    spawn, applied to the inner/outer radius, ``bounds``, and ``exclude_zones``. When a
    ``homography`` is active, ``exclude_zones`` (always pixel-space) are first approximated as
    polygons in *canonical* space (via the inverse homography -- a circle becomes a 32-gon),
    so ``border_width`` -- itself a canonical-space distance whenever a homography is active
    -- applies to them in the same units as the circle/bounds margin, rather than being
    skipped for them entirely.

    ``bounds``, if given as ``(width, height)``, makes the valid region the *intersection* of
    the sector and the rectangle ``[0,width]x[0,height]`` -- not the sector alone. The sector
    can legitimately extend beyond that rectangle (e.g. a fitted reach-circle bigger than the
    marked workspace), but every generated point must still land somewhere the camera
    actually shows.
    """
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    if inner_radius < 0:
        raise ValueError(f"inner_radius must be >= 0, got {inner_radius}")
    if outer_radius <= inner_radius:
        raise ValueError(f"outer_radius ({outer_radius}) must be > inner_radius ({inner_radius})")
    if distribution not in ("random", "grid", "radial"):
        raise ValueError(
            f"Unknown distribution: {distribution!r} (allowed: 'random', 'grid', 'radial')"
        )
    if count_mode not in ("fixed", "variable"):
        raise ValueError(f"Unknown count_mode: {count_mode!r} (allowed: 'fixed', 'variable')")
    exclude_zones = exclude_zones or []
    cx, cy = center
    effective_inner = inner_radius + border_width
    effective_outer = outer_radius - border_width
    if effective_outer <= effective_inner:
        raise ValueError(
            f"border_width ({border_width}) leaves no valid area between inner_radius "
            f"({inner_radius}) and outer_radius ({outer_radius})"
        )
    # The scale `_relocate_if_invalid`'s fallback jitter search grows from -- the expected
    # spacing between neighboring points for a roughly area-uniform disk packing of `count`
    # points (area/count under a circle's pi factor, square-rooted). Using the *whole* disk's
    # radius instead (the original choice) made the jitter search start far too small: it
    # would settle for the very first valid spot, which for two points relocated off the same
    # ray (see `_occupancy_guard`) is right next to each other -- a near-duplicate, not a
    # point that blends into the surrounding spacing.
    expected_spacing = (effective_outer - effective_inner) / max(1.0, math.sqrt(count))

    canonical_zone_polygons: list[list[Point]] | None = None
    if homography is not None and exclude_zones:
        inverse_homography = invert_homography(homography)
        canonical_zone_polygons = [
            _zone_to_canonical_polygon(zone, inverse_homography) for zone in exclude_zones
        ]

    def is_valid(candidate: Point) -> bool:
        dx, dy = candidate[0] - cx, candidate[1] - cy
        r = math.hypot(dx, dy)
        if r < effective_inner or r > effective_outer:
            return False
        theta_deg = math.degrees(math.atan2(dy, dx)) % 360
        if not (angle_start_deg <= theta_deg <= angle_end_deg):
            return False
        if bounds is not None:
            bw, bh = bounds
            if candidate[0] < border_width or candidate[0] > bw - border_width:
                return False
            if candidate[1] < border_width or candidate[1] > bh - border_width:
                return False
        if homography is not None:
            if canonical_zone_polygons is None:
                return True
            return not any(
                _point_near_polygon(candidate, poly, border_width) for poly in canonical_zone_polygons
            )
        pixel = candidate
        return not _point_in_any_zone(pixel, exclude_zones, border_width=border_width)

    if count_mode == "variable":
        # No relocation/CDF/refinement at all -- generate IDEAL grid/radial positions (same
        # closed forms as "fixed" mode) and just drop whichever aren't valid, instead of
        # moving them to force an exact final count. `count` is the TARGET final count, not a
        # fixed resolution: `_search_variable_density` re-tries at a higher/lower density
        # until the survivor count is as close to `count` as it can get, since a naive
        # one-shot resolution can land far short of `count` whenever a lot of the candidate
        # grid/spiral falls outside the valid area (a small circle in a big bounding box, an
        # annulus, a restrictive cut, etc.).
        if distribution == "random":
            raise ValueError("count_mode='variable' only supports distribution 'grid' or 'radial'")

        def make_grid(density: int) -> list[Point]:
            x_min, x_max = cx - effective_outer, cx + effective_outer
            y_min, y_max = cy - effective_outer, cy + effective_outer
            if bounds is not None:
                bw, bh = bounds
                x_min, x_max = max(x_min, border_width), min(x_max, bw - border_width)
                y_min, y_max = max(y_min, border_width), min(y_max, bh - border_width)
            if x_max <= x_min or y_max <= y_min:
                raise ValueError("count_mode='variable': no area available for the given bounds/border_width")
            aspect = (x_max - x_min) / (y_max - y_min)
            cols = max(1, round(math.sqrt(density * aspect)))
            rows = max(1, round(density / cols))
            step_x, step_y = (x_max - x_min) / cols, (y_max - y_min) / rows
            points = []
            for i in range(cols):
                for j in range(rows):
                    p = (x_min + (i + 0.5) * step_x, y_min + (j + 0.5) * step_y)
                    if is_valid(p):
                        points.append(p)
            return points

        def make_radial(density: int) -> list[Point]:
            rng = random.Random(seed)
            angle_offset_deg = rng.uniform(0, 360)
            angle_span = angle_end_deg - angle_start_deg
            golden_step_deg = angle_span * _GOLDEN_RATIO_CONJUGATE
            points = []
            for i in range(density):
                t = (i + 0.5) / density
                r = math.sqrt(effective_inner**2 + (effective_outer**2 - effective_inner**2) * t)
                theta_deg = angle_start_deg + (angle_offset_deg + i * golden_step_deg) % angle_span
                p = (cx + r * math.cos(math.radians(theta_deg)), cy + r * math.sin(math.radians(theta_deg)))
                if is_valid(p):
                    points.append(p)
            return points

        points = _search_variable_density(make_grid if distribution == "grid" else make_radial, count)
        if not points:
            raise ValueError("count_mode='variable' produced zero valid points -- the shape/bounds/exclude_zones are likely too restrictive, or count too low")
        return [apply_homography(homography, p) if homography is not None else p for p in points]

    if distribution == "random":
        rng = random.Random(seed)
        r_sq_lo = inner_radius**2
        r_sq_hi = outer_radius**2
        points: list[Point] = []
        attempts = 0
        max_attempts = count * 200
        while len(points) < count:
            attempts += 1
            if attempts > max_attempts:
                raise ValueError(
                    f"Could not sample {count} sector point(s) after {max_attempts} attempts; "
                    f"exclude_zones/border_width/bounds are likely too restrictive for the "
                    f"available sector area (center={center}, inner_radius={inner_radius}, "
                    f"outer_radius={outer_radius}, angle=[{angle_start_deg}, {angle_end_deg}])"
                )
            r = math.sqrt(rng.uniform(r_sq_lo, r_sq_hi))
            theta = math.radians(rng.uniform(angle_start_deg, angle_end_deg))
            candidate = (cx + r * math.cos(theta), cy + r * math.sin(theta))
            if not is_valid(candidate):
                continue
            pixel = apply_homography(homography, candidate) if homography is not None else candidate
            points.append(pixel)
        return points

    if distribution == "grid":
        def clamp(p: Point) -> Point:
            return _clamp_to_region(
                p, cx, cy, effective_inner, effective_outer,
                angle_start_deg, angle_end_deg, bounds, border_width,
            )

        jitter_rng = random.Random(0)  # fixed, NOT `seed` -- see docstring above
        guarded_is_valid, mark_placed = _occupancy_guard(is_valid, 0.5 * expected_spacing)

        if inner_radius == 0 and angle_end_deg - angle_start_deg >= 360:
            # Full-disk fast path -- always true for the calibration tool, and the common
            # YAML case too. Each lattice column gets its OWN continuously-computed valid
            # y-range (the circle's chord at that x, intersected with `bounds`), and places
            # its proportional share of `count` evenly spaced within that range -- instead of
            # the general path below, which shares a single global row grid across every
            # column. Sharing a row grid meant any two columns landing on the exact row that
            # passes through `center`'s own y were co-radial (angle 0 or 180): when `bounds`
            # clipped that side, "shrink along this point's own ray" found the literal same
            # nearest point for both. Per-column placement can't collide this way, since no
            # two columns ever share candidate positions to begin with -- found by checking
            # actual output coordinates for near-duplicates under realistic clipping, not by
            # eye, after the general path below still showed them.
            # Columns are spaced over the ACTUAL available x-range, not the full circle
            # diameter -- and `cols` itself is chosen from that range's aspect ratio, not a
            # bare sqrt(count). An earlier version spaced `cols` columns over the full
            # diameter, then dropped whichever fell outside `bounds`: the survivors stayed at
            # their original (full-diameter) spacing while every dropped column's share of
            # `count` piled into the survivors' own y-ranges, visibly stretching x-spacing far
            # wider than y-spacing once `bounds` clipped a meaningful fraction of the circle
            # away (found by comparing actual x-gaps against y-gaps within a column, not by
            # eye -- x-gaps came out ~70% wider for a circle bounds-clipped to 70% width).
            x_min, x_max = cx - effective_outer, cx + effective_outer
            y_min, y_max = cy - effective_outer, cy + effective_outer
            if bounds is not None:
                bw, bh = bounds
                x_min, x_max = max(x_min, border_width), min(x_max, bw - border_width)
                y_min, y_max = max(y_min, border_width), min(y_max, bh - border_width)
            if x_max <= x_min or y_max <= y_min:
                raise ValueError(
                    f"Could not place {count} grid point(s) -- exclude_zones/border_width/"
                    f"bounds are likely too restrictive for the available sector area"
                )
            aspect = (x_max - x_min) / (y_max - y_min)
            cols = max(1, round(math.sqrt(count * aspect)))
            step_x = (x_max - x_min) / cols
            # Each column's chord is further split around any exclude_zones it crosses --
            # instead of allocating each column ONE range and patching individual points
            # that land in a zone afterward (the previous design), every zone-free
            # sub-interval gets its own entry here and its own proportional share below.
            # This is what makes the whole pattern's density actually adapt when a zone
            # (or a larger border_width) eats into a column's range: a column split in half
            # by a central obstacle gets a share based on its OWN remaining length, the same
            # as every other sub-interval, so the area lost to the obstacle is compensated by
            # very slightly tighter spacing everywhere else -- not by bunching displaced
            # points into a dense ring right at the obstacle's edge, which is what relocating
            # individual out-of-range points along a ray from `center` was found to do (a
            # real, reported regression: a central exclude_zone collapsed a third of a
            # 50-point pattern into a thin ring at the zone's boundary instead of spreading
            # them across the whole remaining disk).
            col_ranges: list[tuple[float, float, float]] = []
            for i in range(cols):
                x = x_min + (i + 0.5) * step_x
                half_chord_sq = effective_outer**2 - (x - cx) ** 2
                if half_chord_sq <= 0:
                    continue
                half_chord = math.sqrt(half_chord_sq)
                y_lo, y_hi = max(cy - half_chord, y_min), min(cy + half_chord, y_max)
                if y_hi <= y_lo:
                    continue
                excluded = (
                    _excluded_y_intervals_at_x(x, exclude_zones, border_width, homography, canonical_zone_polygons)
                    if exclude_zones
                    else []
                )
                sub_ranges = _subtract_intervals(y_lo, y_hi, excluded) if excluded else [(y_lo, y_hi)]
                col_ranges.extend((x, sub_lo, sub_hi) for sub_lo, sub_hi in sub_ranges if sub_hi > sub_lo)
            if not col_ranges:
                raise ValueError(
                    f"Could not place {count} grid point(s) -- exclude_zones/border_width/"
                    f"bounds are likely too restrictive for the available sector area"
                )
            shares = _allocate_shares([hi - lo for _, lo, hi in col_ranges], count)
            ideal = []
            for (x, y_lo, y_hi), share in zip(col_ranges, shares):
                if share <= 0:
                    continue
                step_y = (y_hi - y_lo) / share
                ideal.extend((x, y_lo + (k + 0.5) * step_y) for k in range(share))
        else:
            # General path (an annulus and/or a restricted pie slice): no per-column chord
            # formula, so fall back to a near-square lattice over the bounding square with
            # each cell individually relocated if invalid. Rarer in practice (the calibration
            # tool never restricts inner_radius/angle), so the residual, much smaller risk of
            # a co-radial collision (mitigated by `_occupancy_guard`, just not eliminated by
            # construction the way the fast path above is) is an acceptable trade-off against
            # the complexity of an exact per-column annulus/sector-chord formula. Sized from
            # effective_outer, not the raw outer_radius -- same boundary-pileup bug as radial's
            # formula above: a bounding square sized to the raw radius puts a meaningful
            # fraction of the initial lattice outside the actual effective region whenever
            # border_width is non-trivial, and `_clamp_to_region`'s ray-shrink then collapses
            # all of them onto the same effective_outer ring instead of leaving them spread
            # across the smaller disk.
            rows = max(1, round(math.sqrt(count)))
            cols = max(1, math.ceil(count / rows))
            step_x = (2 * effective_outer) / cols
            step_y = (2 * effective_outer) / rows
            ideal = [
                (cx - effective_outer + (i + 0.5) * step_x, cy - effective_outer + (j + 0.5) * step_y)
                for i in range(cols)
                for j in range(rows)
            ]
            if len(ideal) > count:
                ideal.sort(key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
                ideal = ideal[:count]

        placed = []
        for i, p in enumerate(ideal):
            point = _relocate_if_invalid(
                p, guarded_is_valid, clamp, jitter_rng, expected_spacing, "grid", i, center, effective_outer,
            )
            mark_placed(point)
            placed.append(point)
        return [apply_homography(homography, p) if homography is not None else p for p in placed]

    # distribution == "radial": a Fermat/Vogel spiral (the "sunflower seed" arrangement) --
    # r_i = sqrt(effective_inner**2 + (effective_outer**2 - effective_inner**2) * (i+0.5)/N),
    # theta_i = i * golden_angle. Uses effective_inner/effective_outer (border_width already
    # applied), not the raw inner_radius/outer_radius -- using the raw radii here was a real,
    # reported bug: ideal positions were computed over the FULL disk, so any point whose ideal
    # r fell within border_width of outer_radius was invalid and got clamped, by
    # `_clamp_to_region`'s ray-shrink, onto the *exact* effective_outer boundary. With no zone
    # involved at all, a border_width of 2 on a radius-10 disk collapsed 45% of 60 points onto
    # that single boundary ring instead of spreading them down to fill the now-smaller disk.
    # This is the standard answer for "N points that visually and statistically look evenly
    # spread across a disk": the golden-angle increment is irrational relative to a full
    # turn, so no two points ever align radially or angularly -- there is no visible row,
    # column, or spoke structure at any N. Each of the N ideal positions is computed directly
    # from its own index (never more, never fewer), then individually relocated if invalid --
    # `seed` rotates the whole spiral by a random starting angle (one rng draw) and seeds the
    # same rng instance's relocation fallback, so the whole thing stays a deterministic
    # function of `seed`.
    rng = random.Random(seed)
    angle_offset_deg = rng.uniform(0, 360)
    angle_span = angle_end_deg - angle_start_deg
    # The step is the golden-angle ratio scaled to THIS span, not the full-circle golden
    # angle (360 * _GOLDEN_RATIO_CONJUGATE = 137.50776...) folded down via modulo afterward
    # -- a real, reported bug for any restricted span (e.g. a semicircle pattern): folding a
    # sequence that's low-discrepancy over 360 degrees into a narrower range via
    # `% angle_span` produces a *different* ratio (`step/angle_span`) than the golden
    # ratio's conjugate, and that different ratio has no guarantee of being similarly hard
    # to approximate by simple fractions -- so the folded sequence can clump and leave gaps,
    # visibly worst right at the span's own boundary (reported as "sparse approaching the
    # [flat] diameter line" of a semicircle). Scaling the step to `angle_span` directly
    # keeps the same well-distributed ratio regardless of span width, and is an exact no-op
    # for the common ``angle_span == 360`` case (`360 * _GOLDEN_RATIO_CONJUGATE` is the same
    # 137.50776...-degree step as before), so the unrestricted disk's behavior is completely
    # unchanged. Verified empirically: nearest-neighbor ratio for a 60-point semicircle went
    # from 0.80 (old fold-based formula) to 0.97 (this fix).
    golden_step_deg = angle_span * _GOLDEN_RATIO_CONJUGATE

    def clamp(p: Point) -> Point:
        return _clamp_to_region(
            p, cx, cy, effective_inner, effective_outer,
            angle_start_deg, angle_end_deg, bounds, border_width,
        )

    guarded_is_valid, mark_placed = _occupancy_guard(is_valid, 0.5 * expected_spacing)
    # When exclude_zones and/or bounds restrict the disk, the radius itself is drawn from the
    # TRUE available-area CDF (see `_radial_available_area_cdf`) instead of the unobstructed
    # closed form -- this is what stops every point whose unobstructed radius would have
    # fallen inside a zone, or outside the workspace rectangle, from being individually
    # relocated onto a thin ring right at that boundary. With neither, skip the (otherwise
    # harmless, just unnecessary) numerical integration and keep the exact closed form, so the
    # common case is unaffected.
    radial_cdf = (
        _radial_available_area_cdf(
            cx, cy, effective_inner, effective_outer, angle_start_deg, angle_end_deg,
            exclude_zones, border_width, homography, canonical_zone_polygons, bounds,
        )
        if exclude_zones or bounds is not None
        else None
    )
    placed = []
    for i in range(count):
        t = (i + 0.5) / count
        if radial_cdf is not None:
            r = _invert_radial_cdf(radial_cdf[0], radial_cdf[1], t)
        else:
            r = math.sqrt(effective_inner**2 + (effective_outer**2 - effective_inner**2) * t)
        theta_deg = angle_start_deg + (angle_offset_deg + i * golden_step_deg) % angle_span
        if radial_cdf is not None:
            # The blind golden-angle theta can land in a blocked sector at THIS point's own
            # radius even though the CDF above already accounted for blocking when deciding
            # how many points this radius shell gets overall -- and `_relocate_if_invalid`'s
            # `clamp` step, for a `bounds` violation specifically, shrinks the point INWARD
            # along this same angle until back in bounds, silently overriding the CDF's
            # carefully chosen radius rather than preserving it. So instead of leaving this to
            # relocate, place the angle directly within whatever's actually open at r: map the
            # golden-angle fraction onto the available sub-intervals at this exact radius
            # (`_available_angle_intervals_at_radius` + `_place_in_available_intervals`), the
            # same idea as the grid fast path placing a point within a column's available
            # sub-interval, just in angle instead of y. Found from a real, reported
            # regression: without this, a workspace rectangle clipping a third of the circle
            # gave radial a 0.37 nearest-neighbor ratio (vs. grid's 0.97 under the identical
            # clipping) even with the CDF radius fix already in place, because so many points'
            # radii were still being silently collapsed back down by the bounds clamp.
            avail = _available_angle_intervals_at_radius(
                r, cx, cy, angle_start_deg, angle_end_deg, exclude_zones, border_width,
                homography, canonical_zone_polygons, bounds,
            )
            placed_theta = _place_in_available_intervals(
                avail, (theta_deg - angle_start_deg) / angle_span if angle_span > 0 else 0.0
            )
            if placed_theta is not None:
                theta_deg = placed_theta
        ideal = (cx + r * math.cos(math.radians(theta_deg)), cy + r * math.sin(math.radians(theta_deg)))
        point = _relocate_if_invalid(
            ideal, guarded_is_valid, clamp, rng, expected_spacing, "radial", i, center, effective_outer,
        )
        mark_placed(point)
        placed.append(point)
    if radial_cdf is not None:
        placed = _refine_radial_local_separation(
            placed, cx, cy, angle_start_deg, angle_end_deg, exclude_zones, border_width,
            homography, canonical_zone_polygons, bounds, expected_spacing, is_valid,
        )
    return [apply_homography(homography, p) if homography is not None else p for p in placed]


def generate_union_points(
    circles: list[tuple[Point, float]],
    count: int,
    seed: int,
    distribution: str = "random",
    exclude_zones: list[ExcludeZoneConfig] | None = None,
    homography: Homography | None = None,
    border_width: float = 0.0,
    bounds: tuple[float, float] | None = None,
    count_mode: str = "fixed",
) -> list[Point]:
    """Points filling the UNION of N full disks (``circles``, each ``(center, radius)``) --
    built for a bimanual (or any multi-region) setup where two arms' reach circles overlap,
    and treating each circle as an independently-sampled region would sample the overlap at
    roughly DOUBLE the density of the non-overlapping parts (each circle's own pattern adds
    its own points there, unaware the other circle already covers it). Scoped to full disks
    only (no ``inner_radius``/angle restriction per circle) -- this is exactly what the
    calibration tool's fitted reach-circles always are, so there's no need for the extra
    generality `generate_sector_points` carries for a single restricted pie-slice.

    ``circles``/``bounds`` are in canonical space when ``homography`` is given (mirroring
    `generate_sector_points`'s convention), pixel space otherwise. ``exclude_zones`` are
    always pixel-space facts about the image, checked after any homography mapping, exactly
    as in `generate_sector_points`.

    ``"random"``: rejection-sampled uniformly over the union's bounding box (NOT "pick a
    circle weighted by area, then sample within it" -- that approach sounds right but isn't:
    a point in the overlap is reachable by sampling from EITHER circle, so it would get
    accepted roughly twice as often as a non-overlapping point of the same area, silently
    re-introducing the exact double-density problem this function exists to avoid).
    Bounding-box rejection has no such bias -- standard, correct area-uniform sampling for an
    arbitrary region, just less efficient than a tighter envelope when circles are far apart
    relative to their radii (same ``count * 200`` attempt cap as every other distribution
    here; if that's ever a real problem for a genuinely sparse arm layout, revisit then).

    ``"grid"``: a direct generalization of `generate_sector_points`'s full-disk fast path --
    each lattice column's valid y-range is the UNION of every circle's chord at that x
    (`_merge_intervals`, the same interval algebra already used there for combining
    exclude_zone subtractions), not the chord of just one circle. Merging is exactly what
    prevents overlap double-counting: a column straddling two circles' chords gets ONE merged
    range covering both, with `count` allocated to it proportionally to that merged range's
    own length (`_allocate_shares`), not to each circle's chord length separately (which would
    double-count the overlapping portion). `exclude_zones` subtraction and `bounds` clipping
    apply per-column exactly as in the single-circle case, just (where a column has multiple
    disjoint merged sub-ranges, e.g. two circles not yet touching at that x) applied to each
    piece independently.

    ``"radial"`` is NOT YET SUPPORTED here (raises clearly) -- the Fermat/Vogel spiral is
    inherently single-center, and correctly generalizing its area-CDF machinery to an
    arbitrary union of differently-centered circles needs its own derivation (reusing
    `_circle_zone_angle_block`'s closed form to compute, at a given radius from a chosen
    origin, the angular extent INSIDE each union member instead of inside an exclude zone, then
    merging those as an inclusion constraint rather than subtracting them as an exclusion --
    plausible, but deliberately deferred to its own pass rather than guessed at here).

    Each circle degrades to exactly `generate_sector_points`'s own output when ``circles`` has
    only one entry -- verified by a regression test asserting byte-identical points between
    the two functions for the same single circle/seed/distribution.

    Relocation (`_relocate_if_invalid`) needs a single ``center``/``effective_outer`` to ray-
    search from, which doesn't exist for a union -- resolved by delegating each invalid point
    to whichever circle's center it's closest to (by signed distance to that circle's own
    boundary), then clamping/searching using THAT circle's own geometry. This is a per-point,
    not a per-pattern, choice: two different points needing relocation can delegate to two
    different circles.
    """
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    if not circles:
        raise ValueError("circles must be non-empty")
    if distribution not in ("random", "grid", "radial"):
        raise ValueError(
            f"Unknown distribution: {distribution!r} (allowed: 'random', 'grid', 'radial')"
        )
    if distribution == "radial":
        raise ValueError(
            "distribution='radial' is not yet supported for a union of circles (only "
            "'random' and 'grid' are) -- the Fermat/Vogel spiral is inherently single-center; "
            "use 'grid' (the default for sector patterns) or 'random' for a union pattern"
        )
    if count_mode not in ("fixed", "variable"):
        raise ValueError(f"Unknown count_mode: {count_mode!r} (allowed: 'fixed', 'variable')")
    if count_mode == "variable" and distribution != "grid":
        raise ValueError("count_mode='variable' only supports distribution 'grid' for a union pattern")
    exclude_zones = exclude_zones or []
    effective_circles: list[tuple[float, float, float]] = []
    for (cx, cy), radius in circles:
        eff_r = radius - border_width
        if eff_r > 0:
            effective_circles.append((cx, cy, eff_r))
    if not effective_circles:
        raise ValueError(
            f"border_width ({border_width}) leaves no circle with positive effective radius"
        )

    canonical_zone_polygons: list[list[Point]] | None = None
    if homography is not None and exclude_zones:
        inverse_homography = invert_homography(homography)
        canonical_zone_polygons = [
            _zone_to_canonical_polygon(zone, inverse_homography) for zone in exclude_zones
        ]

    def is_valid(candidate: Point) -> bool:
        px, py = candidate
        if not any(
            (px - cx) ** 2 + (py - cy) ** 2 <= eff_r ** 2 for cx, cy, eff_r in effective_circles
        ):
            return False
        if bounds is not None:
            bw, bh = bounds
            if px < border_width or px > bw - border_width:
                return False
            if py < border_width or py > bh - border_width:
                return False
        if homography is not None:
            if canonical_zone_polygons is None:
                return True
            return not any(
                _point_near_polygon(candidate, poly, border_width) for poly in canonical_zone_polygons
            )
        return not _point_in_any_zone(candidate, exclude_zones, border_width=border_width)

    x_min = min(cx - r for cx, cy, r in effective_circles)
    x_max = max(cx + r for cx, cy, r in effective_circles)
    y_min = min(cy - r for cx, cy, r in effective_circles)
    y_max = max(cy + r for cx, cy, r in effective_circles)
    if bounds is not None:
        bw, bh = bounds
        x_min, x_max = max(x_min, border_width), min(x_max, bw - border_width)
        y_min, y_max = max(y_min, border_width), min(y_max, bh - border_width)
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(
            f"Could not place {count} union point(s) -- bounds/border_width leave no area "
            f"at all for the given circles"
        )
    # Based on total circle area (ignoring overlap -- fine for a search-scale estimate, not a
    # correctness-critical value), NOT the bounding box's diagonal. A bbox-based estimate is a
    # poor proxy here specifically: two very differently-sized circles (or circles separated
    # by empty space) share one wide bbox, so a bbox-derived spacing systematically
    # overestimates how loosely packed the SMALLER circle's own points actually are once
    # `_allocate_shares` gives it its proportionally smaller share of `count` -- found as a
    # real test failure: `_occupancy_guard`'s `min_separation` (half of this value) ended up
    # tighter than the smaller circle's true point-to-point spacing, so legitimate grid
    # points were rejected as "too close" and the relocation fallback exhausted its 200
    # attempts trying to find room that didn't exist within that circle's own area.
    total_circle_area = sum(math.pi * r * r for _, _, r in effective_circles)
    expected_spacing = math.sqrt(total_circle_area / count) if total_circle_area > 0 else 1.0

    if count_mode == "variable":
        # Same idea as generate_sector_points' variable mode: generate IDEAL grid positions at
        # a given density and drop whichever fall outside the union/bounds/exclude_zones,
        # instead of relocating to force an exact count. `count` is a TARGET, not a fixed
        # resolution -- `_search_variable_density` retries at higher/lower density until the
        # survivor count is as close to `count` as achievable.
        def make_grid(density: int) -> list[Point]:
            aspect = (x_max - x_min) / (y_max - y_min)
            cols = max(1, round(math.sqrt(density * aspect)))
            rows = max(1, round(density / cols))
            step_x, step_y = (x_max - x_min) / cols, (y_max - y_min) / rows
            points = []
            for i in range(cols):
                for j in range(rows):
                    p = (x_min + (i + 0.5) * step_x, y_min + (j + 0.5) * step_y)
                    if is_valid(p):
                        points.append(p)
            return points

        points = _search_variable_density(make_grid, count)
        if not points:
            raise ValueError(
                "count_mode='variable' produced zero valid points -- the circles/bounds/"
                "exclude_zones are likely too restrictive, or count too low"
            )
        return [apply_homography(homography, p) if homography is not None else p for p in points]

    def nearest_circle(point: Point) -> tuple[float, float, float]:
        px, py = point
        return min(effective_circles, key=lambda c: math.hypot(px - c[0], py - c[1]) - c[2])

    def relocate_union_point(
        candidate: Point, valid_fn: Callable[[Point], bool], rng: random.Random, label: str, index: int
    ) -> Point:
        cx, cy, eff_r = nearest_circle(candidate)

        def clamp(p: Point) -> Point:
            return _clamp_to_region(p, cx, cy, 0.0, eff_r, 0.0, 360.0, bounds, border_width)

        return _relocate_if_invalid(candidate, valid_fn, clamp, rng, expected_spacing, label, index, (cx, cy), eff_r)

    if distribution == "random":
        rng = random.Random(seed)
        points: list[Point] = []
        attempts = 0
        max_attempts = count * 200
        while len(points) < count:
            attempts += 1
            if attempts > max_attempts:
                raise ValueError(
                    f"Could not sample {count} union point(s) after {max_attempts} attempts; "
                    f"exclude_zones/border_width/bounds are likely too restrictive, or the "
                    f"circles are too sparse relative to their union's bounding box"
                )
            candidate = (rng.uniform(x_min, x_max), rng.uniform(y_min, y_max))
            if not is_valid(candidate):
                continue
            points.append(apply_homography(homography, candidate) if homography is not None else candidate)
        return points

    # distribution == "grid"
    jitter_rng = random.Random(0)  # fixed, NOT `seed` -- mirrors generate_sector_points
    guarded_is_valid, mark_placed = _occupancy_guard(is_valid, 0.5 * expected_spacing)

    aspect = (x_max - x_min) / (y_max - y_min)
    cols = max(1, round(math.sqrt(count * aspect)))
    step_x = (x_max - x_min) / cols

    col_ranges: list[tuple[float, float, float]] = []
    for i in range(cols):
        x = x_min + (i + 0.5) * step_x
        raw_chords = []
        for cx, cy, eff_r in effective_circles:
            dx = x - cx
            if abs(dx) > eff_r:
                continue
            half_chord = math.sqrt(eff_r * eff_r - dx * dx)
            lo, hi = max(cy - half_chord, y_min), min(cy + half_chord, y_max)
            if hi > lo:
                raw_chords.append((lo, hi))
        merged_chords = _merge_intervals(raw_chords)
        if not merged_chords:
            continue
        if exclude_zones:
            excluded = _excluded_y_intervals_at_x(x, exclude_zones, border_width, homography, canonical_zone_polygons)
            for lo, hi in merged_chords:
                col_ranges.extend(
                    (x, sub_lo, sub_hi) for sub_lo, sub_hi in _subtract_intervals(lo, hi, excluded) if sub_hi > sub_lo
                )
        else:
            col_ranges.extend((x, lo, hi) for lo, hi in merged_chords)

    if not col_ranges:
        raise ValueError(
            f"Could not place {count} union grid point(s) -- exclude_zones/border_width/"
            f"bounds are likely too restrictive for the available union area"
        )
    shares = _allocate_shares([hi - lo for _, lo, hi in col_ranges], count)
    ideal = []
    for (x, y_lo, y_hi), share in zip(col_ranges, shares):
        if share <= 0:
            continue
        step_y = (y_hi - y_lo) / share
        ideal.extend((x, y_lo + (k + 0.5) * step_y) for k in range(share))

    placed = []
    for i, p in enumerate(ideal):
        point = relocate_union_point(p, guarded_is_valid, jitter_rng, "union grid", i)
        mark_placed(point)
        placed.append(point)
    return [apply_homography(homography, p) if homography is not None else p for p in placed]


def build_pattern(
    pattern_config: PatternConfig,
    count: int,
    exclude_zones: list[ExcludeZoneConfig] | None = None,
    homography: Homography | None = None,
    bounds: tuple[float, float] | None = None,
) -> list[Point]:
    """Dispatch on ``pattern_config.shape``. The sole extension point for a future shape."""
    exclude_zones = exclude_zones or []
    if pattern_config.shape == "arc":
        points = generate_arc_points(
            center=pattern_config.center,
            radius=pattern_config.radius,
            angle_start_deg=pattern_config.angle_start_deg,
            angle_end_deg=pattern_config.angle_end_deg,
            count=count,
        )
        _validate_no_excluded_points(points, exclude_zones, shape_label="arc")
        return points
    if pattern_config.shape == "line":
        points = generate_line_points(
            start=pattern_config.start,
            end=pattern_config.end,
            count=count,
        )
        _validate_no_excluded_points(points, exclude_zones, shape_label="line")
        return points
    if pattern_config.shape == "sector":
        return generate_sector_points(
            center=pattern_config.center,
            inner_radius=pattern_config.inner_radius,
            outer_radius=pattern_config.radius,
            angle_start_deg=pattern_config.angle_start_deg,
            angle_end_deg=pattern_config.angle_end_deg,
            count=count,
            seed=pattern_config.seed,
            distribution=pattern_config.distribution or "random",
            exclude_zones=exclude_zones,
            homography=homography,
            border_width=pattern_config.border_width or 0.0,
            bounds=bounds,
            count_mode=pattern_config.count_mode or "fixed",
        )
    if pattern_config.shape == "union":
        return generate_union_points(
            circles=pattern_config.circles,
            count=count,
            seed=pattern_config.seed,
            distribution=pattern_config.distribution or "random",
            exclude_zones=exclude_zones,
            homography=homography,
            border_width=pattern_config.border_width or 0.0,
            bounds=bounds,
            count_mode=pattern_config.count_mode or "fixed",
        )
    raise ValueError(f"Unknown pattern shape: {pattern_config.shape!r}")


def target_for_episode(points: list[T], episode_index: int) -> T:
    """Target value for ``episode_index``, wrapping if the session runs longer than the
    configured pattern length. Generic over the element type so this same modulo-cycling
    logic serves both position points (``Point``) and rotation angles (``float``) -- they
    are independent sequences that can have different lengths and therefore desync over a
    long-enough session, which only adds coverage diversity, never removes it."""
    if not points:
        raise ValueError("points must be non-empty")
    return points[episode_index % len(points)]


def generate_rotation_angles(
    count: int,
    method: str = "uniform",
    angle_start_deg: float = 0.0,
    angle_end_deg: float = 360.0,
    seed: int = 0,
    initial_angle_deg: float = 0.0,
) -> list[float]:
    """``count`` target orientation angles (degrees), in whatever space the paired position
    pattern already produces its points in (pixel space, or canonical/rectified space when a
    homography is active -- see ``orientation_arrow_points``, which is what actually projects
    an angle from one space to the other for display). Cycled per-episode the same way
    position points are, via ``target_for_episode``, with its OWN independent length: a
    different ``count`` than the position pattern's own ``count`` is intentional, not a
    mismatch to reconcile -- e.g. lengths 5 and 4 only repeat the exact same
    position/rotation pairing every 20 episodes, so decoupling the two sequences only adds
    coverage diversity over a session, never removes it.

    ``"uniform"``: ``count`` angles evenly spaced across ``[angle_start_deg, angle_end_deg)``,
    each placed at the center of its own equal sub-slice
    (``angle_start_deg + (i + 0.5) * span / count``) rather than at both inclusive endpoints
    the way ``generate_arc_points`` does for a deliberately partial arc. This matters for the
    common full-rotation default (``0`` to ``360``): 0 and 360 are the same physical angle, so
    an endpoint-inclusive scheme would duplicate it. Mirrors the midpoint convention
    ``generate_sector_points``'s ``"radial"`` distribution already uses for its own
    evenly-spaced parameter ``t``.

    ``"random"``: ``count`` independent ``random.Random(seed).uniform(angle_start_deg,
    angle_end_deg)`` draws -- deterministic given ``seed``, like every other random
    distribution in this module.
    """
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    if method not in ("uniform", "random"):
        raise ValueError(f"Unknown rotation method: {method!r} (allowed: 'uniform', 'random')")
    # angle_start_deg/angle_end_deg are RELATIVE to initial_angle_deg, not absolute -- e.g.
    # initial_angle_deg=90, start=-45, end=45 spreads +-45 degrees around direction 90,
    # without the user having to compute absolute angles by hand. Negative values are fine.
    start = initial_angle_deg + angle_start_deg
    end = initial_angle_deg + angle_end_deg
    span = end - start
    if method == "uniform":
        if count == 1:
            return [start]
        # Full wrap (start/end are the same physical angle): divide by count so the last
        # point doesn't duplicate the first. Otherwise divide by count-1 so the angles always
        # cover the full [start, end] range regardless of count -- dividing by count alone
        # left a gap that shrank toward (but never reached) `end` as count grew, making the
        # spread look like it depended on count instead of being directly set by start/end.
        divisor = count if span % 360 == 0 else count - 1
        return [start + i * span / divisor for i in range(count)]
    rng = random.Random(seed)
    return [rng.uniform(start, end) for _ in range(count)]


def orientation_arrow_points(
    position: Point,
    angle_deg: float,
    length: float,
    homography: Homography | None = None,
) -> tuple[Point, Point]:
    """The ``(tail, tip)`` pixel-space points of an arrow showing a target orientation of
    ``angle_deg`` at ``position`` (already pixel space -- the output of
    ``target_for_episode``/``build_pattern``), with the rotation itself applied in
    CANONICAL space when a homography is active, not pixel space.

    This is the one subtlety: a homography does not preserve angles (it maps lines to lines,
    but not the angles between them), so rotating a vector directly in pixel space would not
    show what a real object at ``angle_deg`` in the actual workspace would look like under a
    tilted camera -- the same reason ``sector``'s area-uniform sampling already has to happen
    in canonical space, not pixel space. Since ``position`` only exists in pixel space by the
    time an episode's target is selected (``build_pattern`` already applied the forward
    homography internally), this recovers its canonical-space coordinates via the inverse
    homography, applies the rotation there, and forward-maps the result back -- rather than
    threading a second, canonical-space copy of every pattern through ``session.py`` just for
    this. Exact up to floating point, since ``apply_homography``/``invert_homography`` are
    exact inverses of one another.

    With no homography (the common, uncalibrated case), canonical space and pixel space are
    the same thing, so the rotation is applied directly with no conversion -- consistent with
    ``generate_arc_points``'s angle convention (0 along +x, increasing toward +y).
    """
    theta = math.radians(angle_deg)
    if homography is None:
        px, py = position
        return position, (px + length * math.cos(theta), py + length * math.sin(theta))
    cx, cy = apply_homography(invert_homography(homography), position)
    tip_canonical = (cx + length * math.cos(theta), cy + length * math.sin(theta))
    return position, apply_homography(homography, tip_canonical)
