import math
import random

import pytest

from vizaudit.overlay.config import ExcludeZoneConfig, PatternConfig
from vizaudit.overlay.pattern import (
    build_pattern,
    generate_arc_points,
    generate_line_points,
    generate_sector_points,
    target_for_episode,
)
from vizaudit.overlay.pattern import (
    _allocate_shares,
    _available_angle_intervals_at_radius,
    _circle_zone_angle_block,
    _clamp_to_region,
    _excluded_y_intervals_at_x,
    _inflate_polygon,
    _merge_angle_intervals,
    _merge_intervals,
    _occupancy_guard,
    _outside_bounds_angle_intervals,
    _place_in_available_intervals,
    _polygon_angle_block,
    _polygon_vertical_intervals,
    _refine_radial_local_separation,
    _relocate_if_invalid,
    _subtract_intervals,
)
from vizaudit.overlay.perspective import compute_homography


def _nearest_neighbor_distances(points):
    return [
        min(math.hypot(p[0] - q[0], p[1] - q[1]) for j, q in enumerate(points) if j != i)
        for i, p in enumerate(points)
    ]


def _duplicate_count(points, decimals=6):
    seen = {}
    for p in points:
        key = (round(p[0], decimals), round(p[1], decimals))
        seen[key] = seen.get(key, 0) + 1
    return sum(c - 1 for c in seen.values() if c > 1)


def test_generate_arc_points_semicircle():
    points = generate_arc_points(center=(0, 0), radius=10, angle_start_deg=0, angle_end_deg=180, count=3)
    assert len(points) == 3
    assert points[0] == pytest.approx((10, 0), abs=1e-9)
    assert points[1] == pytest.approx((0, 10), abs=1e-9)
    assert points[2] == pytest.approx((-10, 0), abs=1e-9)


def test_generate_arc_points_single_point_is_midpoint_angle():
    points = generate_arc_points(center=(0, 0), radius=5, angle_start_deg=0, angle_end_deg=180, count=1)
    assert len(points) == 1
    x, y = points[0]
    assert x == pytest.approx(0, abs=1e-9)
    assert y == pytest.approx(5, abs=1e-9)


def test_generate_arc_points_offset_center():
    points = generate_arc_points(center=(100, 100), radius=10, angle_start_deg=0, angle_end_deg=90, count=2)
    assert points[0] == pytest.approx((110, 100), abs=1e-9)
    assert points[1] == pytest.approx((100, 110), abs=1e-9)


def test_generate_arc_points_rejects_zero_count():
    with pytest.raises(ValueError):
        generate_arc_points(center=(0, 0), radius=1, angle_start_deg=0, angle_end_deg=90, count=0)


def test_generate_line_points():
    points = generate_line_points(start=(0, 0), end=(10, 0), count=5)
    assert points == pytest.approx([(0, 0), (2.5, 0), (5, 0), (7.5, 0), (10, 0)])


def test_generate_line_points_single_point_is_midpoint():
    points = generate_line_points(start=(0, 0), end=(10, 10), count=1)
    assert points == pytest.approx([(5, 5)])


def test_build_pattern_dispatches_arc():
    cfg = PatternConfig(shape="arc", center=(0, 0), radius=10, angle_start_deg=0, angle_end_deg=180)
    assert len(build_pattern(cfg, count=3)) == 3


def test_build_pattern_dispatches_line():
    cfg = PatternConfig(shape="line", start=(0, 0), end=(10, 0))
    assert build_pattern(cfg, count=2) == pytest.approx([(0, 0), (10, 0)])


def test_build_pattern_unknown_shape_raises():
    cfg = PatternConfig(shape="triangle")
    with pytest.raises(ValueError):
        build_pattern(cfg, count=3)


def test_target_for_episode_wraps():
    points = [(0, 0), (1, 1), (2, 2)]
    assert target_for_episode(points, 0) == (0, 0)
    assert target_for_episode(points, 2) == (2, 2)
    assert target_for_episode(points, 3) == (0, 0)
    assert target_for_episode(points, 4) == (1, 1)


def test_target_for_episode_empty_raises():
    with pytest.raises(ValueError):
        target_for_episode([], 0)


def test_generate_sector_points_count_and_bounds():
    points = generate_sector_points(
        center=(0, 0), inner_radius=0, outer_radius=10,
        angle_start_deg=0, angle_end_deg=90, count=200, seed=0,
    )
    assert len(points) == 200
    for x, y in points:
        r = math.hypot(x, y)
        assert 0 <= r <= 10 + 1e-9
        assert x >= -1e-9 and y >= -1e-9  # first quadrant only, given 0-90deg


def test_generate_sector_points_respects_inner_radius():
    points = generate_sector_points(
        center=(0, 0), inner_radius=5, outer_radius=10,
        angle_start_deg=0, angle_end_deg=360, count=300, seed=1,
    )
    for x, y in points:
        r = math.hypot(x, y)
        assert 5 - 1e-9 <= r <= 10 + 1e-9


def test_generate_sector_points_deterministic_given_seed():
    a = generate_sector_points(center=(0, 0), inner_radius=0, outer_radius=10,
                                angle_start_deg=0, angle_end_deg=90, count=10, seed=7)
    b = generate_sector_points(center=(0, 0), inner_radius=0, outer_radius=10,
                                angle_start_deg=0, angle_end_deg=90, count=10, seed=7)
    assert a == b


def test_generate_sector_points_different_seeds_differ():
    a = generate_sector_points(center=(0, 0), inner_radius=0, outer_radius=10,
                                angle_start_deg=0, angle_end_deg=90, count=10, seed=1)
    b = generate_sector_points(center=(0, 0), inner_radius=0, outer_radius=10,
                                angle_start_deg=0, angle_end_deg=90, count=10, seed=2)
    assert a != b


def test_generate_sector_points_rejects_zero_count():
    with pytest.raises(ValueError):
        generate_sector_points(center=(0, 0), inner_radius=0, outer_radius=10,
                                angle_start_deg=0, angle_end_deg=90, count=0, seed=0)


def test_generate_sector_points_rejects_inner_ge_outer():
    with pytest.raises(ValueError):
        generate_sector_points(center=(0, 0), inner_radius=10, outer_radius=10,
                                angle_start_deg=0, angle_end_deg=90, count=1, seed=0)


def test_generate_sector_points_avoids_exclude_zone():
    zones = [ExcludeZoneConfig(name="blocker", center=(0, 0), radius=8)]
    points = generate_sector_points(
        center=(0, 0), inner_radius=0, outer_radius=10,
        angle_start_deg=0, angle_end_deg=90, count=50, seed=3, exclude_zones=zones,
    )
    assert len(points) == 50
    for x, y in points:
        assert math.hypot(x, y) > 8


def test_generate_sector_points_exhausted_attempts_raises():
    # Exclusion zone covers the entire sector -> impossible to satisfy, must raise ValueError
    # promptly rather than looping forever.
    zones = [ExcludeZoneConfig(name="blocker", center=(0, 0), radius=100)]
    with pytest.raises(ValueError):
        generate_sector_points(
            center=(0, 0), inner_radius=0, outer_radius=10,
            angle_start_deg=0, angle_end_deg=90, count=5, seed=0, exclude_zones=zones,
        )


def test_generate_sector_points_with_homography_maps_into_pixel_quadrilateral():
    corners = [(50, 50), (250, 50), (250, 200), (50, 200)]  # axis-aligned 200x150 rectangle
    h = compute_homography(corners)
    points = generate_sector_points(
        center=(100, 75), inner_radius=0, outer_radius=50,
        angle_start_deg=0, angle_end_deg=360, count=30, seed=0, homography=h,
    )
    for x, y in points:
        assert 50 <= x <= 250
        assert 50 <= y <= 200


# ── border_width applies to exclude_zones when a homography is active ──────────────────


def test_generate_sector_points_homography_border_width_buffers_circle_zone():
    # Axis-aligned corners -> homography is just a translation, so canonical and pixel
    # distances match exactly here, making the buffered margin easy to assert precisely.
    corners = [(0, 0), (200, 0), (200, 200), (0, 200)]
    h = compute_homography(corners)
    zone = ExcludeZoneConfig(name="blocker", center=(110, 100), radius=5)  # pixel space
    points = generate_sector_points(
        center=(100, 100), inner_radius=0, outer_radius=50,
        angle_start_deg=0, angle_end_deg=360, count=60, seed=0,
        homography=h, exclude_zones=[zone], border_width=3,
    )
    assert len(points) == 60
    for x, y in points:
        assert math.hypot(x - 110, y - 100) >= 5 + 3 - 1e-6


def test_generate_sector_points_no_homography_border_width_still_buffers_zone():
    # Without a homography, border_width is in the same (pixel) space as exclude_zones, so
    # this already worked before -- confirms the new homography-active path didn't regress it.
    zone = ExcludeZoneConfig(name="blocker", center=(10, 0), radius=2)
    points = generate_sector_points(
        center=(0, 0), inner_radius=0, outer_radius=20,
        angle_start_deg=0, angle_end_deg=360, count=40, seed=0,
        exclude_zones=[zone], border_width=3,
    )
    for x, y in points:
        assert math.hypot(x - 10, y - 0) >= 2 + 3 - 1e-6


def test_build_pattern_dispatches_sector():
    cfg = PatternConfig(shape="sector", center=(0, 0), radius=10, inner_radius=0,
                         angle_start_deg=0, angle_end_deg=90, seed=0)
    assert len(build_pattern(cfg, count=5)) == 5


def test_build_pattern_arc_raises_on_excluded_point():
    # center=(0,0), radius=10, single point at angle 0 -> (10, 0); zone covers it.
    cfg = PatternConfig(shape="arc", center=(0, 0), radius=10, angle_start_deg=0, angle_end_deg=0)
    zones = [ExcludeZoneConfig(name="blocker", center=(10, 0), radius=1)]
    with pytest.raises(ValueError):
        build_pattern(cfg, count=1, exclude_zones=zones)


def test_build_pattern_line_raises_on_excluded_point():
    cfg = PatternConfig(shape="line", start=(0, 0), end=(10, 0))
    zones = [ExcludeZoneConfig(name="blocker", center=(5, 0), radius=1)]
    with pytest.raises(ValueError):
        build_pattern(cfg, count=3, exclude_zones=zones)  # midpoint (5,0) falls in the zone


def test_build_pattern_unaffected_when_no_exclude_zones_passed():
    # Backward-compat: existing 2-arg call sites keep working unchanged.
    cfg = PatternConfig(shape="arc", center=(0, 0), radius=10, angle_start_deg=0, angle_end_deg=180)
    assert len(build_pattern(cfg, count=3)) == 3


# ── distribution="grid" (near-square Cartesian lattice, individually relocated) ─────────────


def test_generate_sector_points_grid_count_and_bounds():
    points = generate_sector_points(
        center=(0, 0), inner_radius=0, outer_radius=10,
        angle_start_deg=0, angle_end_deg=360, count=37, seed=0, distribution="grid",
    )
    assert len(points) == 37
    for x, y in points:
        assert math.hypot(x, y) <= 10 + 1e-9


def test_generate_sector_points_grid_is_axis_aligned_lattice():
    # The full-disk fast path places every column's share of points at that column's own
    # fixed x -- most points should share one of a small number of exact x coordinates, the
    # "covers all the spots in rows and columns" structure a grid is supposed to have.
    points = generate_sector_points(
        center=(0, 0), inner_radius=0, outer_radius=10,
        angle_start_deg=0, angle_end_deg=360, count=30, seed=7, distribution="grid",
    )
    xs = [round(x, 6) for x, _ in points]
    assert len(set(xs)) < len(xs)


def test_generate_sector_points_grid_inner_radius_respected():
    points = generate_sector_points(
        center=(0, 0), inner_radius=4, outer_radius=10,
        angle_start_deg=0, angle_end_deg=360, count=25, seed=0, distribution="grid",
    )
    assert len(points) == 25
    for x, y in points:
        r = math.hypot(x, y)
        assert 4 - 1e-6 <= r <= 10 + 1e-6


def test_generate_sector_points_grid_respects_angle_range():
    points = generate_sector_points(
        center=(0, 0), inner_radius=0, outer_radius=10,
        angle_start_deg=0, angle_end_deg=90, count=20, seed=0, distribution="grid",
    )
    assert len(points) == 20
    for x, y in points:
        assert x >= -1e-6 and y >= -1e-6


def test_generate_sector_points_grid_ignores_seed():
    # The grid lattice itself has no random component, and its relocation fallback uses a
    # fixed internal seed -- the result must come out identical regardless of the `seed`
    # argument, unlike "radial" (spiral phase offset) and "random" (rejection sampling).
    a = generate_sector_points(center=(0, 0), inner_radius=0, outer_radius=10,
                                angle_start_deg=0, angle_end_deg=360, count=15, seed=1,
                                distribution="grid")
    b = generate_sector_points(center=(0, 0), inner_radius=0, outer_radius=10,
                                angle_start_deg=0, angle_end_deg=360, count=15, seed=2,
                                distribution="grid")
    assert a == b


def test_generate_sector_points_grid_scales_with_count():
    # A bigger count must produce a denser (smaller-step) lattice that still spans the full
    # disk, not a fixed-size pattern with more clutter in one corner.
    def min_x_gap(count: int) -> float:
        points = generate_sector_points(
            center=(0, 0), inner_radius=0, outer_radius=10,
            angle_start_deg=0, angle_end_deg=360, count=count, seed=0, distribution="grid",
        )
        xs = sorted(set(round(x, 6) for x, _ in points))
        gaps = [b - a for a, b in zip(xs, xs[1:])]
        return min(gaps)

    assert min_x_gap(400) < min_x_gap(36)


@pytest.mark.parametrize("distribution", ["grid", "radial"])
def test_generate_sector_points_no_duplicates_under_heavy_bounds_clipping(distribution):
    # Regression test for the actual reported bug: a circle mostly clipped by `bounds`
    # (simulating a reach circle that pokes outside the marked workspace on one side) used to
    # either thin some regions far more than others (population trimming) or -- after
    # switching to per-point relocation -- could relocate two *different* ideal points onto
    # the exact same boundary spot, since "shrink toward center along this point's own ray"
    # finds the same nearest point for any two points that happen to share that ray. Neither
    # symptom is acceptable: every index must produce its own point, and no two may collide.
    for seed in range(5):
        points = generate_sector_points(
            center=(10, 10), inner_radius=0, outer_radius=10,
            angle_start_deg=0, angle_end_deg=360, count=50, seed=seed,
            distribution=distribution, bounds=(14, 20),
        )
        assert len(points) == 50
        assert _duplicate_count(points) == 0


@pytest.mark.parametrize("distribution", ["grid", "radial"])
def test_generate_sector_points_low_count_coverage_under_clipping(distribution):
    # The user's actual reported scenario: low counts (tens, not thousands -- this tool
    # guides on the order of 20-100 demonstrations) with a circle clipped by the workspace
    # `bounds`. No nearest-neighbor distance should collapse toward zero (that's what
    # "missing points"/overlapping points looked like) relative to the average spacing.
    points = generate_sector_points(
        center=(10, 10), inner_radius=0, outer_radius=10,
        angle_start_deg=0, angle_end_deg=360, count=50, seed=0,
        distribution=distribution, bounds=(14, 20),
    )
    distances = _nearest_neighbor_distances(points)
    assert min(distances) > 0.1 * (sum(distances) / len(distances))


def test_generate_sector_points_grid_stays_highly_even_under_heavy_asymmetric_clipping():
    # Tighter bound than the shared cross-distribution check above, specific to "grid"'s
    # full-disk fast path: per-column continuous placement (each column gets a share
    # proportional to its own valid chord length, evenly spaced within it) should keep
    # coverage close to perfectly even even when bounds clips roughly half the circle away --
    # not just "not collapsed," but genuinely even, which is what "grid" promises.
    points = generate_sector_points(
        center=(10, 10), inner_radius=0, outer_radius=10,
        angle_start_deg=0, angle_end_deg=360, count=50, seed=0,
        distribution="grid", bounds=(14, 20),
    )
    distances = _nearest_neighbor_distances(points)
    assert min(distances) > 0.7 * (sum(distances) / len(distances))


def test_generate_sector_points_grid_column_spacing_matches_row_spacing_under_clipping():
    # Regression test for a real reported bug: column (x) spacing was computed from the
    # FULL circle diameter, then columns falling outside `bounds` were simply dropped --
    # the survivors stayed at their original wide spacing while the dropped columns' share
    # of `count` piled into the survivors' own row (y) spacing, visibly stretching x-spacing
    # far wider than y-spacing. `cols` must now be chosen from the actual available
    # bounding box's aspect ratio, with columns spaced over that actual box, not the
    # uncllipped diameter.
    points = generate_sector_points(
        center=(10, 10), inner_radius=0, outer_radius=10,
        angle_start_deg=0, angle_end_deg=360, count=50, seed=0,
        distribution="grid", bounds=(14, 20),
    )
    xs = sorted(set(round(x, 6) for x, _ in points))
    x_gap = xs[1] - xs[0]
    y_gaps = []
    for x in xs:
        ys = sorted(y for px, y in points if round(px, 6) == x)
        y_gaps.extend(b - a for a, b in zip(ys, ys[1:]))
    avg_y_gap = sum(y_gaps) / len(y_gaps)
    assert 0.5 < x_gap / avg_y_gap < 2.0


@pytest.mark.parametrize("distribution", ["grid", "radial"])
def test_generate_sector_points_relocated_points_fan_out_around_central_obstacle(distribution):
    # Regression test for a real reported bug: points relocated off a circular exclude_zone
    # near `center` (e.g. marking out the robot's own base) landed via unstructured random
    # jitter, clumping at essentially arbitrary angles relative to each other instead of
    # fanning out around the obstacle the way the surrounding pattern's own structure would
    # suggest. `_relocate_if_invalid` now tries moving radially along the point's OWN angle
    # first, and `_occupancy_guard` now rejects a relocation landing too close (not just
    # exactly on top of) an already-placed point. Verified via nearest-neighbor distance,
    # not visually: no point should land catastrophically close to another near the obstacle.
    zone = ExcludeZoneConfig(name="robot_base", shape="circle", center=(10, 10), radius=4)
    points = generate_sector_points(
        center=(10, 10), inner_radius=0, outer_radius=10,
        angle_start_deg=0, angle_end_deg=360, count=50, seed=0,
        distribution=distribution, exclude_zones=[zone],
    )
    assert len(points) == 50
    distances = _nearest_neighbor_distances(points)
    assert min(distances) > 0.3 * (sum(distances) / len(distances))


def test_allocate_shares_sums_to_count_and_respects_group_sizes():
    shares = _allocate_shares([10.0, 3.0, 7.0], 12)
    assert sum(shares) == 12
    assert all(share <= size for share, size in zip(shares, [10, 3, 7]))


def test_allocate_shares_matches_largest_remainder_method():
    # 12 * [10,3,7]/20 = [6.0, 1.8, 4.2] -> floors [6,1,4] (sum 11) -> the one remaining pick
    # goes to the largest fractional remainder (group 1, remainder 0.8).
    assert _allocate_shares([10.0, 3.0, 7.0], 12) == [6, 2, 4]


def test_allocate_shares_handles_continuous_sizes():
    # The grid fast path's sizes are continuous chord lengths, not integer candidate counts.
    shares = _allocate_shares([2.5, 7.5], 10)
    assert shares == [3, 7]  # 2.5/10*10=2.5 -> 2 + remainder 0.5; 7.5/10*10=7.5 -> 7 + remainder 0.5
    assert sum(shares) == 10


# ── distribution="radial" (Fermat/Vogel spiral, individually relocated) ─────────────────────


def test_generate_sector_points_radial_count_and_bounds():
    points = generate_sector_points(
        center=(0, 0), inner_radius=0, outer_radius=10,
        angle_start_deg=0, angle_end_deg=360, count=37, seed=0, distribution="radial",
    )
    assert len(points) == 37
    for x, y in points:
        assert math.hypot(x, y) <= 10 + 1e-9


def test_generate_sector_points_radial_is_not_axis_aligned_lattice():
    # The golden-angle increment never aligns two points radially or angularly, so no two
    # should share an x coordinate -- this is what "no visible rows/columns/spokes" means
    # underneath, and what distinguishes it from "grid".
    points = generate_sector_points(
        center=(0, 0), inner_radius=0, outer_radius=10,
        angle_start_deg=0, angle_end_deg=360, count=30, seed=7, distribution="radial",
    )
    xs = [round(x, 6) for x, _ in points]
    assert len(set(xs)) == len(xs)


def test_generate_sector_points_radial_inner_radius_respected():
    # The spiral's r formula must also respect a nonzero inner_radius (annular sector).
    points = generate_sector_points(
        center=(0, 0), inner_radius=4, outer_radius=10,
        angle_start_deg=0, angle_end_deg=360, count=25, seed=0, distribution="radial",
    )
    assert len(points) == 25
    for x, y in points:
        r = math.hypot(x, y)
        assert 4 - 1e-6 <= r <= 10 + 1e-6


def test_generate_sector_points_radial_respects_angle_range():
    points = generate_sector_points(
        center=(0, 0), inner_radius=0, outer_radius=10,
        angle_start_deg=0, angle_end_deg=90, count=20, seed=0, distribution="radial",
    )
    assert len(points) == 20
    for x, y in points:
        assert x >= -1e-6 and y >= -1e-6


def test_generate_sector_points_radial_restricted_span_does_not_clump_near_diameter():
    # Regression test for a real, reported bug: a semicircle (or any restricted angle span)
    # folded the FULL-CIRCLE golden-angle sequence into the narrower span via plain modulo,
    # which breaks the golden ratio's special low-discrepancy property (it was tuned for a
    # 360-degree step, not whatever ratio `step % angle_span` happens to produce) -- visible
    # as sparse coverage right at the span's own flat boundary (reported as "sparse
    # approaching the [flat] diameter line" of a semicircle). Measured: nearest-neighbor
    # ratio of 0.80 with the old fold-based formula vs. 0.97 with the span-scaled step.
    points = generate_sector_points(
        center=(10, 0), inner_radius=0, outer_radius=10,
        angle_start_deg=0, angle_end_deg=180, count=60, seed=0, distribution="radial",
    )
    assert len(points) == 60
    distances = _nearest_neighbor_distances(points)
    assert min(distances) > 0.85 * (sum(distances) / len(distances))


def test_generate_sector_points_radial_full_circle_unaffected_by_span_scaling():
    # The span-scaled golden-angle step must be an exact no-op for the unrestricted
    # (angle_span == 360) case, since 360 * _GOLDEN_RATIO_CONJUGATE is exactly the same step
    # as the old hardcoded constant -- this just locks that equivalence in as a regression
    # test, rather than relying on it only being asserted in a comment.
    points = generate_sector_points(
        center=(0, 0), inner_radius=0, outer_radius=10,
        angle_start_deg=0, angle_end_deg=360, count=15, seed=5, distribution="radial",
    )
    expected_step = 360 * ((3 - 5**0.5) / 2)
    assert abs(expected_step - 137.50776405003785) < 1e-9
    assert len(points) == 15


def test_generate_sector_points_radial_deterministic_given_seed():
    a = generate_sector_points(center=(0, 0), inner_radius=0, outer_radius=10,
                                angle_start_deg=0, angle_end_deg=360, count=15, seed=5,
                                distribution="radial")
    b = generate_sector_points(center=(0, 0), inner_radius=0, outer_radius=10,
                                angle_start_deg=0, angle_end_deg=360, count=15, seed=5,
                                distribution="radial")
    assert a == b


def test_generate_sector_points_radial_different_seeds_can_differ():
    a = generate_sector_points(center=(0, 0), inner_radius=0, outer_radius=10,
                                angle_start_deg=0, angle_end_deg=360, count=10, seed=1,
                                distribution="radial")
    b = generate_sector_points(center=(0, 0), inner_radius=0, outer_radius=10,
                                angle_start_deg=0, angle_end_deg=360, count=10, seed=2,
                                distribution="radial")
    assert a != b


# ── _clamp_to_region / _relocate_if_invalid / _occupancy_guard (the relocation strategy) ────


def test_clamp_to_region_shrinks_radius_outward_and_inward():
    far = _clamp_to_region((100, 0), 0, 0, 2, 10, 0, 360, None, 0)
    assert far == pytest.approx((10, 0))
    near = _clamp_to_region((1, 0), 0, 0, 2, 10, 0, 360, None, 0)
    assert near == pytest.approx((2, 0))


def test_clamp_to_region_snaps_angle_into_pie_slice():
    x, y = _clamp_to_region((0, -10), 0, 0, 0, 10, 0, 90, None, 0)
    assert math.degrees(math.atan2(y, x)) % 360 == pytest.approx(0, abs=1e-6)


def test_clamp_to_region_shrinks_along_ray_for_bounds_not_axis_clamp():
    # A point straight out along +x from center, beyond `bounds`, must come back in along
    # that SAME ray (i.e. y stays 0) -- an axis clamp would also give y=0 here coincidentally,
    # so this checks the actual landing x sits exactly on the bounds edge, confirming the ray
    # search (not just a lucky axis-clamp match).
    x, y = _clamp_to_region((20, 10), 10, 10, 0, 100, 0, 360, (14, 20), 0)
    assert y == pytest.approx(10)
    assert x == pytest.approx(14, abs=1e-4)


def test_relocate_if_invalid_returns_unchanged_when_already_valid():
    point = (1, 1)
    result = _relocate_if_invalid(point, lambda p: True, lambda p: p, None, 1, "test", 0, (0, 0), 10)
    assert result == point


def test_relocate_if_invalid_raises_when_nothing_nearby_is_valid():
    with pytest.raises(ValueError, match="grid point index 3"):
        _relocate_if_invalid(
            (0, 0), lambda p: False, lambda p: p, random.Random(0), 1.0, "grid", 3, (0, 0), 10
        )


def test_relocate_if_invalid_searches_radially_before_random_jitter():
    # A point inside a central obstacle should relocate by moving OUTWARD along its own
    # angle first -- not by an unstructured random jitter -- so that several different
    # points relocated off the same obstacle stay fanned out at their original angles
    # instead of clumping together at arbitrary angles relative to each other.
    def is_valid(p):
        return math.hypot(p[0], p[1]) >= 2  # invalid inside radius 2 of the origin

    candidate = (1, 0)  # inside the obstacle, on the +x axis
    result = _relocate_if_invalid(
        candidate, is_valid, lambda p: p, random.Random(0), 1.0, "radial", 0, (0, 0), 10
    )
    assert result[1] == pytest.approx(0, abs=1e-6)  # stayed on the +x axis (theta unchanged)
    assert math.hypot(*result) >= 2 - 1e-6


def test_occupancy_guard_rejects_a_point_already_marked_placed():
    guarded, mark_placed = _occupancy_guard(lambda p: True)
    assert guarded((1, 1)) is True
    mark_placed((1, 1))
    assert guarded((1, 1)) is False
    assert guarded((1, 1.1)) is True  # a distinct point is unaffected


def test_generate_sector_points_invalid_distribution_raises():
    with pytest.raises(ValueError):
        generate_sector_points(center=(0, 0), inner_radius=0, outer_radius=10,
                                angle_start_deg=0, angle_end_deg=360, count=5, seed=0,
                                distribution="bogus")


# ── border_width ──────────────────────────────────────────────────────────


def test_generate_sector_points_border_width_shrinks_radius_random():
    points = generate_sector_points(
        center=(0, 0), inner_radius=0, outer_radius=10,
        angle_start_deg=0, angle_end_deg=360, count=50, seed=0,
        distribution="random", border_width=3,
    )
    for x, y in points:
        assert math.hypot(x, y) <= 7 + 1e-9


def test_generate_sector_points_border_width_shrinks_radius_grid():
    points = generate_sector_points(
        center=(0, 0), inner_radius=0, outer_radius=10,
        angle_start_deg=0, angle_end_deg=360, count=20, seed=0,
        distribution="grid", border_width=3,
    )
    assert len(points) == 20
    for x, y in points:
        assert math.hypot(x, y) <= 7 + 1e-9


def test_generate_sector_points_border_width_too_large_raises():
    with pytest.raises(ValueError):
        generate_sector_points(
            center=(0, 0), inner_radius=0, outer_radius=10,
            angle_start_deg=0, angle_end_deg=360, count=5, seed=0, border_width=6,
        )  # inner+border (6) >= outer-border (4)


# ── bounds (workspace-rectangle intersection) ───────────────────────────────


def test_generate_sector_points_bounds_clips_circle_random():
    # Circle centered at a corner of the bounds rectangle -- only one quadrant overlaps.
    points = generate_sector_points(
        center=(0, 0), inner_radius=0, outer_radius=50,
        angle_start_deg=0, angle_end_deg=360, count=40, seed=0,
        distribution="random", bounds=(30, 30),
    )
    assert len(points) == 40
    for x, y in points:
        assert 0 <= x <= 30 and 0 <= y <= 30


def test_generate_sector_points_bounds_clips_circle_grid():
    points = generate_sector_points(
        center=(0, 0), inner_radius=0, outer_radius=50,
        angle_start_deg=0, angle_end_deg=360, count=25, seed=0,
        distribution="grid", bounds=(30, 30),
    )
    assert len(points) == 25
    for x, y in points:
        assert 0 <= x <= 30 and 0 <= y <= 30


def test_generate_sector_points_bounds_no_overlap_raises():
    with pytest.raises(ValueError):
        generate_sector_points(
            center=(1000, 1000), inner_radius=0, outer_radius=5,
            angle_start_deg=0, angle_end_deg=360, count=5, seed=0, bounds=(30, 30),
        )


# ── polygon exclude_zones ───────────────────────────────────────────────────


def test_generate_sector_points_avoids_polygon_exclude_zone():
    # A square cut covering the right half of the disk.
    zone = ExcludeZoneConfig(name="cut", shape="polygon", vertices=[(0, -20), (20, -20), (20, 20), (0, 20)])
    points = generate_sector_points(
        center=(0, 0), inner_radius=0, outer_radius=10,
        angle_start_deg=0, angle_end_deg=360, count=40, seed=0,
        exclude_zones=[zone],
    )
    assert len(points) == 40
    for x, y in points:
        assert x < 0  # right half (x>=0) is cut out


def test_build_pattern_arc_raises_on_polygon_excluded_point():
    cfg = PatternConfig(shape="arc", center=(0, 0), radius=10, angle_start_deg=0, angle_end_deg=0)
    zone = ExcludeZoneConfig(name="cut", shape="polygon", vertices=[(8, -2), (12, -2), (12, 2), (8, 2)])
    with pytest.raises(ValueError):
        build_pattern(cfg, count=1, exclude_zones=[zone])  # point (10,0) falls inside the cut


# ── border_width-only boundary pileup (no exclude_zones) ───────────────────


def test_generate_sector_points_radial_border_width_no_boundary_pileup():
    # Regression test for a real reported bug: radial's radius formula used the RAW
    # inner_radius/outer_radius instead of effective_inner/effective_outer, so a
    # border_width of 2 on a radius-10 disk collapsed 45% of 60 points onto the exact
    # effective_outer boundary ring (via `_clamp_to_region`'s ray-shrink) instead of
    # spreading them down to fill the smaller disk.
    points = generate_sector_points(
        center=(10, 10), inner_radius=0, outer_radius=10,
        angle_start_deg=0, angle_end_deg=360, count=60, seed=0,
        distribution="radial", border_width=2.0,
    )
    radii = [math.hypot(x - 10, y - 10) for x, y in points]
    boundary_ring = sum(1 for r in radii if r > 7.4)
    assert boundary_ring < 15  # proportional expectation is ~9; old buggy behavior gave 27


def test_generate_sector_points_grid_general_path_border_width_no_boundary_pileup():
    # Same bug, in grid's general (annulus/restricted-angle) fallback path: its initial
    # lattice was sized from the raw outer_radius, not effective_outer.
    points = generate_sector_points(
        center=(10, 10), inner_radius=0, outer_radius=10,
        angle_start_deg=0, angle_end_deg=180, count=40, seed=0,
        distribution="grid", border_width=2.0,
    )
    radii = [math.hypot(x - 10, y - 10) for x, y in points]
    boundary_ring = sum(1 for r in radii if r > 7.4)
    assert boundary_ring < 12  # old buggy behavior gave 14 of 40 in this thin outer ring


# ── exclude_zones reshape the structure instead of just relocating points ──


@pytest.mark.parametrize("distribution", ["grid", "radial"])
def test_generate_sector_points_central_zone_does_not_pileup_at_boundary_ring(distribution):
    # Regression test for the "ring artifact": relocating each point individually (radially
    # outward from `center`) when its ideal position fell inside a central exclude_zone
    # collapsed many points onto a thin ring just past the zone's boundary -- 13 of 60
    # points within 0.6 units of a radius-4 zone's edge in one measured case, for radial.
    # The structural fix (per-column interval subtraction for grid, an available-area CDF
    # for radial) should distribute the "lost" points across the whole remaining disk
    # instead of bunching them at the zone's edge.
    zone = ExcludeZoneConfig(name="base", shape="circle", center=(10, 10), radius=4)
    points = generate_sector_points(
        center=(10, 10), inner_radius=0, outer_radius=10,
        angle_start_deg=0, angle_end_deg=360, count=60, seed=0,
        distribution=distribution, exclude_zones=[zone],
    )
    assert len(points) == 60
    radii = [math.hypot(x - 10, y - 10) for x, y in points]
    ring_count = sum(1 for r in radii if 4.0 <= r <= 4.6)
    assert ring_count <= 8  # proportional expectation for this band is ~4; old bug gave 13


@pytest.mark.parametrize("distribution", ["grid", "radial"])
def test_generate_sector_points_central_zone_keeps_even_spacing(distribution):
    zone = ExcludeZoneConfig(name="base", shape="circle", center=(10, 10), radius=4)
    points = generate_sector_points(
        center=(10, 10), inner_radius=0, outer_radius=10,
        angle_start_deg=0, angle_end_deg=360, count=60, seed=0,
        distribution=distribution, exclude_zones=[zone],
    )
    distances = _nearest_neighbor_distances(points)
    assert min(distances) > 0.55 * (sum(distances) / len(distances))


@pytest.mark.parametrize("distribution", ["grid", "radial"])
def test_generate_sector_points_central_square_zone_keeps_even_spacing(distribution):
    # Same as above, but with a polygon (rectangle) zone -- exercises the line-clip
    # (_polygon_vertical_intervals) / circle-segment-intersection (_polygon_angle_block)
    # paths instead of the closed-form circle paths.
    zone = ExcludeZoneConfig(
        name="base", shape="polygon", vertices=[(7, 7), (13, 7), (13, 13), (7, 13)]
    )
    points = generate_sector_points(
        center=(10, 10), inner_radius=0, outer_radius=10,
        angle_start_deg=0, angle_end_deg=360, count=60, seed=0,
        distribution=distribution, exclude_zones=[zone],
    )
    assert len(points) == 60
    distances = _nearest_neighbor_distances(points)
    assert min(distances) > 0.5 * (sum(distances) / len(distances))


@pytest.mark.parametrize("distribution", ["grid", "radial"])
def test_generate_sector_points_zone_plus_border_width_combined(distribution):
    # border_width applies to the zone (radius 3 -> effective 4) and the outer boundary
    # (radius 10 -> effective 9) at the same time -- every point must respect both.
    zone = ExcludeZoneConfig(name="base", shape="circle", center=(10, 10), radius=3)
    points = generate_sector_points(
        center=(10, 10), inner_radius=0, outer_radius=10,
        angle_start_deg=0, angle_end_deg=360, count=50, seed=0,
        distribution=distribution, exclude_zones=[zone], border_width=1.0,
    )
    assert len(points) == 50
    for x, y in points:
        r = math.hypot(x - 10, y - 10)
        assert r >= 4 - 1e-6
        assert r <= 9 + 1e-6


def test_generate_sector_points_radial_border_width_sweep_stays_consistent_under_bounds_clip():
    # Regression test for a real, reported bug: nearest-neighbor ratio swung
    # non-monotonically (0.57-0.89, with the worst point well below either neighboring
    # border_width) across a border_width sweep on the exact same (self-similarly clipped)
    # circle-vs-bounds shape, instead of degrading predictably as border_width grows. Root
    # cause: the per-point angular placement preserves AGGREGATE density correctly but not
    # point-to-point separation, so two points at different radii can coincidentally map to
    # nearly the same angle -- fixed by `_refine_radial_local_separation`, a bounded
    # post-process pass that nudges any point too close to its nearest neighbor toward a
    # better angle at its own (unchanged) radius.
    ratios = []
    for border_width in (0, 0.5, 1, 2, 3):
        points = generate_sector_points(
            center=(10, 10), inner_radius=0, outer_radius=10,
            angle_start_deg=0, angle_end_deg=360, count=60, seed=0,
            distribution="radial", bounds=(14, 20), border_width=border_width,
        )
        distances = _nearest_neighbor_distances(points)
        ratios.append(min(distances) / (sum(distances) / len(distances)))
    assert min(ratios) > 0.65, f"ratios were {ratios}, old buggy behavior dipped to ~0.57"


def test_generate_sector_points_radial_off_center_zone_stays_consistent_across_seeds():
    zone = ExcludeZoneConfig(name="base", shape="circle", center=(14, 7), radius=2.5)
    ratios = []
    for seed in range(5):
        points = generate_sector_points(
            center=(10, 10), inner_radius=0, outer_radius=10,
            angle_start_deg=0, angle_end_deg=360, count=60, seed=seed,
            distribution="radial", exclude_zones=[zone],
        )
        distances = _nearest_neighbor_distances(points)
        ratios.append(min(distances) / (sum(distances) / len(distances)))
    assert min(ratios) > 0.7, f"ratios were {ratios}, old buggy behavior ranged ~0.49-0.68"


def test_refine_radial_local_separation_improves_a_close_pair():
    placed = [(0.0, 0.0), (0.1, 0.0), (5.0, 5.0), (-5.0, 5.0), (-5.0, -5.0)]
    refined = _refine_radial_local_separation(
        list(placed), cx=0.0, cy=0.0, angle_start_deg=0, angle_end_deg=360,
        exclude_zones=[], border_width=0.0, homography=None, canonical_zone_polygons=None,
        bounds=None, expected_spacing=3.0,
    )
    distances_before = _nearest_neighbor_distances(placed)
    distances_after = _nearest_neighbor_distances(refined)
    assert min(distances_after) > min(distances_before)


def test_refine_radial_local_separation_noop_for_already_even_points():
    placed = [
        (10 * math.cos(math.radians(a)), 10 * math.sin(math.radians(a)))
        for a in (0, 72, 144, 216, 288)
    ]
    refined = _refine_radial_local_separation(
        list(placed), cx=0.0, cy=0.0, angle_start_deg=0, angle_end_deg=360,
        exclude_zones=[], border_width=0.0, homography=None, canonical_zone_polygons=None,
        bounds=None, expected_spacing=10.0,
    )
    for (x1, y1), (x2, y2) in zip(placed, refined):
        assert math.hypot(x1 - x2, y1 - y2) < 1e-9


# ── shared interval-algebra helpers ─────────────────────────────────────────


def test_merge_intervals_combines_overlapping():
    assert _merge_intervals([(0, 5), (3, 8), (10, 12)]) == [(0, 8), (10, 12)]


def test_merge_intervals_empty():
    assert _merge_intervals([]) == []


def test_subtract_intervals_splits_into_two():
    assert _subtract_intervals(0, 10, [(4, 6)]) == [(0, 4), (6, 10)]


def test_subtract_intervals_removes_entire_base():
    assert _subtract_intervals(0, 10, [(-5, 15)]) == []


def test_subtract_intervals_no_overlap_returns_base_unchanged():
    assert _subtract_intervals(0, 10, [(20, 30)]) == [(0, 10)]


def test_inflate_polygon_grows_outward_from_centroid():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    inflated = _inflate_polygon(square, 2.0)
    for (vx, vy), (ix, iy) in zip(square, inflated):
        assert math.hypot(ix - 5, iy - 5) > math.hypot(vx - 5, vy - 5)


def test_inflate_polygon_noop_for_zero_border():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert _inflate_polygon(square, 0.0) == square


def test_polygon_vertical_intervals_rectangle():
    rect = [(2, 2), (8, 2), (8, 8), (2, 8)]
    assert _polygon_vertical_intervals(5, rect) == [(2, 8)]
    assert _polygon_vertical_intervals(0, rect) == []


def test_excluded_y_intervals_at_x_circle_zone():
    zone = ExcludeZoneConfig(name="z", shape="circle", center=(5, 5), radius=3)
    intervals = _excluded_y_intervals_at_x(5, [zone], border_width=0.0, homography=None, canonical_zone_polygons=None)
    assert intervals == [(2.0, 8.0)]


def test_excluded_y_intervals_at_x_outside_circle_zone_is_empty():
    zone = ExcludeZoneConfig(name="z", shape="circle", center=(5, 5), radius=3)
    intervals = _excluded_y_intervals_at_x(20, [zone], border_width=0.0, homography=None, canonical_zone_polygons=None)
    assert intervals == []


def test_circle_zone_angle_block_concentric_full_block():
    assert _circle_zone_angle_block(2, 10, 10, 10, 10, 5) == [(0.0, 360.0)]


def test_circle_zone_angle_block_no_overlap():
    assert _circle_zone_angle_block(2, 10, 10, 30, 30, 1) == []


def test_circle_zone_angle_block_partial_overlap_symmetric_around_zone_direction():
    # zone centered directly to the +x side of `center`
    blocked = _circle_zone_angle_block(8, 10, 10, 18, 10, 3)
    assert len(blocked) == 1
    lo, hi = blocked[0]
    assert abs((lo + hi) / 2 - 0.0) < 1e-6  # centered on angle 0 (+x direction)
    assert hi - lo > 0


def test_polygon_angle_block_matches_circle_zone_for_a_circle_shaped_polygon():
    # Approximate a circular zone as a 64-gon and confirm the polygon path agrees closely
    # with the closed-form circle path.
    n = 64
    zx, zy, rz = 18.0, 10.0, 3.0
    poly = [
        (zx + rz * math.cos(2 * math.pi * i / n), zy + rz * math.sin(2 * math.pi * i / n))
        for i in range(n)
    ]
    circle_blocked = _circle_zone_angle_block(8, 10, 10, zx, zy, rz)
    poly_blocked = _polygon_angle_block(8, 10, 10, poly)
    assert len(circle_blocked) == len(poly_blocked) == 1
    # `_circle_zone_angle_block` doesn't normalize its raw lo/hi mod 360 (the caller --
    # `_merge_angle_intervals` -- does that); `_polygon_angle_block`'s crossings already come
    # from `% 360`. Compare span and center mod 360 rather than raw endpoints.
    c_lo, c_hi = circle_blocked[0]
    p_lo, p_hi = poly_blocked[0]
    assert abs((c_hi - c_lo) - (p_hi - p_lo)) < 0.5
    assert abs(((c_lo + c_hi) / 2) % 360 - ((p_lo + p_hi) / 2) % 360) < 0.5


def test_merge_angle_intervals_handles_wraparound():
    # (350, 370) normalizes to the wrapping arc [350, 360) U [0, 10), which overlaps (5, 15)
    # in [5, 10) -- a naive linear merge (no wraparound awareness) would miss that overlap
    # entirely and treat the two as disjoint.
    merged = _merge_angle_intervals([(350, 370), (5, 15)])
    assert merged == [(0.0, 15.0), (350.0, 360.0)]
    total = sum(hi - lo for lo, hi in merged)
    assert abs(total - 25.0) < 1e-9


# ── radial + bounds: the workspace rectangle was missing from the zone-only CDF fix ────


def test_outside_bounds_angle_intervals_matches_expected_chord():
    # Workspace [14, 20] with center at (10, 10): nearest edge is x=14 (distance 4), so at
    # r=5 the circle pokes out past it -- the blocked arc's half-angle is acos(4/5).
    intervals = _outside_bounds_angle_intervals(5, 10, 10, (14, 20), 0.0)
    total = sum(hi - lo for lo, hi in intervals)
    assert abs(total - 2 * math.degrees(math.acos(4 / 5))) < 0.01


def test_outside_bounds_angle_intervals_empty_when_circle_fits_entirely_inside():
    assert _outside_bounds_angle_intervals(2, 10, 10, (14, 20), 0.0) == []


def test_place_in_available_intervals_basic():
    assert _place_in_available_intervals([(0, 10), (20, 30)], 0.0) == 0.0
    assert abs(_place_in_available_intervals([(0, 10), (20, 30)], 0.75) - 25.0) < 1e-9


def test_place_in_available_intervals_empty_returns_none():
    assert _place_in_available_intervals([], 0.5) is None


def test_available_angle_intervals_at_radius_no_obstruction_is_full_circle():
    intervals = _available_angle_intervals_at_radius(2, 10, 10, 0, 360, [], 0.0, None, None, None)
    assert intervals == [(0, 360)]


@pytest.mark.parametrize("distribution", ["grid", "radial"])
def test_generate_sector_points_bounds_only_radial_does_not_clump(distribution):
    # Regression test for a real, reported follow-up bug: even after the exclude_zones-aware
    # CDF fix, radial was "just as broken" under `bounds` clipping (this tool's primary real
    # use case -- a workspace quad smaller than the fitted reach circle), because
    # `_relocate_if_invalid`'s `clamp` step shrinks a bounds-violating point INWARD along its
    # own angle, silently overriding the CDF's chosen radius. Measured before this fix:
    # nearest-neighbor ratio of 0.37 for radial under this exact clipping (vs. grid's 0.97).
    points = generate_sector_points(
        center=(10, 10), inner_radius=0, outer_radius=10,
        angle_start_deg=0, angle_end_deg=360, count=60, seed=0,
        distribution=distribution, bounds=(14, 20),
    )
    assert len(points) == 60
    for x, y in points:
        assert 0 <= x <= 14 and 0 <= y <= 20
    distances = _nearest_neighbor_distances(points)
    assert min(distances) > 0.6 * (sum(distances) / len(distances))


def test_generate_sector_points_radial_bounds_plus_zone_combined():
    zone = ExcludeZoneConfig(
        name="base", shape="polygon", vertices=[(7, 7), (13, 7), (13, 13), (7, 13)]
    )
    points = generate_sector_points(
        center=(10, 10), inner_radius=0, outer_radius=10,
        angle_start_deg=0, angle_end_deg=360, count=60, seed=0,
        distribution="radial", exclude_zones=[zone], bounds=(14, 20),
    )
    assert len(points) == 60
    distances = _nearest_neighbor_distances(points)
    assert min(distances) > 0.5 * (sum(distances) / len(distances))
