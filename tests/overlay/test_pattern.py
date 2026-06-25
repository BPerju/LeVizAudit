import math
import random

import pytest

from vizaudit.overlay.config import ExcludeZoneConfig, PatternConfig
from vizaudit.overlay.pattern import (
    build_pattern,
    combination_index,
    combination_period,
    generate_arc_points,
    generate_line_points,
    generate_rotation_angles,
    generate_sector_points,
    generate_union_points,
    level_order,
    occupied_sites,
    orientation_arrow_points,
    resolve_combination_indices,
    resolve_keep_apart,
    sequence_index,
    target_for_episode,
)
from vizaudit.overlay.pattern import (
    _allocate_shares,
    _available_angle_intervals_at_radius,
    _cartesian_indices,
    _circle_zone_angle_block,
    _clamp_to_region,
    _combine_seed,
    _coprime_stride_for,
    _excluded_y_intervals_at_x,
    _gcd,
    _inflate_polygon,
    _merge_angle_intervals,
    _merge_intervals,
    _mulberry32,
    _occupancy_guard,
    _outside_bounds_angle_intervals,
    _place_in_available_intervals,
    _polygon_angle_block,
    _polygon_vertical_intervals,
    _point_near_polygon,
    _refine_radial_local_separation,
    _relocate_if_invalid,
    _seeded_permutation,
    _subtract_intervals,
)
from vizaudit.overlay.perspective import apply_homography, compute_homography, invert_homography


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
        bounds=None, expected_spacing=3.0, is_valid=lambda p: True,
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
        bounds=None, expected_spacing=10.0, is_valid=lambda p: True,
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


def test_generate_sector_points_radial_refinement_respects_border_width_near_off_center_cut():
    # Regression: _refine_radial_local_separation could move a point to a candidate that
    # violated border_width near an off-center polygon cut, because it only checked the
    # cut's APPROXIMATE inflated boundary, not the exact one.
    zone = ExcludeZoneConfig(
        name="r", shape="polygon", vertices=[(150, 100), (250, 100), (250, 140), (150, 140)]
    )
    points = generate_sector_points(
        center=(200, 150), inner_radius=0, outer_radius=100,
        angle_start_deg=0, angle_end_deg=360, count=40, seed=2,
        distribution="radial", exclude_zones=[zone], border_width=10,
    )
    for p in points:
        assert not _point_near_polygon(p, zone.vertices, 10)


# ── generate_union_points: bimanual (or any multi-region) reach-circle union ──────────────


def _circle_lens_area(r1, r2, d):
    """Closed-form intersection (lens) area of two circles of radii r1/r2 whose centers are
    distance d apart -- the standard two-circular-segment formula. Used as ground truth to
    check generate_union_points doesn't double-sample the overlap, independent of the
    implementation under test."""
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2
    d1 = (d * d + r1 * r1 - r2 * r2) / (2 * d)
    d2 = d - d1
    return (
        r1 * r1 * math.acos(max(-1.0, min(1.0, d1 / r1))) - d1 * math.sqrt(max(0.0, r1 * r1 - d1 * d1))
        + r2 * r2 * math.acos(max(-1.0, min(1.0, d2 / r2))) - d2 * math.sqrt(max(0.0, r2 * r2 - d2 * d2))
    )


@pytest.mark.parametrize("distribution", ["random", "grid"])
def test_generate_union_points_returns_exactly_count(distribution):
    circles = [((0.0, 0.0), 10.0), ((15.0, 0.0), 10.0)]  # overlapping
    points = generate_union_points(circles, count=50, seed=0, distribution=distribution)
    assert len(points) == 50


@pytest.mark.parametrize("distribution", ["random", "grid"])
def test_generate_union_points_every_point_inside_some_circle(distribution):
    circles = [((0.0, 0.0), 10.0), ((25.0, 0.0), 8.0)]  # non-overlapping
    points = generate_union_points(circles, count=80, seed=1, distribution=distribution)
    for x, y in points:
        assert any(math.hypot(x - cx, y - cy) <= r + 1e-6 for (cx, cy), r in circles)


def test_generate_union_points_single_circle_grid_matches_generate_sector_points():
    # Degenerate case: a "union" of exactly one circle's "grid" output must be byte-identical
    # to sampling that circle directly with generate_sector_points -- the per-column chord
    # logic is structurally the same algorithm either way (merging a list of exactly one
    # interval is a no-op), so there's no reason for them to differ.
    union_points = generate_union_points(
        [((10.0, 10.0), 10.0)], count=40, seed=7, distribution="grid",
    )
    sector_points = generate_sector_points(
        center=(10.0, 10.0), inner_radius=0.0, outer_radius=10.0,
        angle_start_deg=0.0, angle_end_deg=360.0, count=40, seed=7, distribution="grid",
    )
    assert union_points == sector_points


def test_generate_union_points_single_circle_random_is_valid_but_not_byte_identical():
    # "random" is NOT expected to match generate_sector_points's output for a single circle:
    # generate_union_points rejection-samples over the bounding BOX (x, y both uniform), while
    # generate_sector_points samples in POLAR form (r via sqrt(uniform(r^2)), theta uniform) --
    # both are legitimate area-uniform algorithms, but they consume the same seeded RNG in a
    # different order/shape, so their outputs differ even for the identical circle/seed. Just
    # check the result is still a valid, correctly-sized sample of that one circle.
    points = generate_union_points([((10.0, 10.0), 10.0)], count=40, seed=7, distribution="random")
    assert len(points) == 40
    for x, y in points:
        assert math.hypot(x - 10.0, y - 10.0) <= 10.0 + 1e-9


def test_generate_union_points_grid_does_not_double_sample_the_overlap():
    # The core bug this function exists to avoid: treating each arm's circle as an
    # independent pattern would sample the overlap region at roughly DOUBLE its fair
    # area-proportional share (once from each circle's own pattern). Two equal circles,
    # overlapping by a known amount, let the expected overlap share be computed analytically
    # (ground truth independent of the implementation) and compared against the actual count.
    r, d = 10.0, 12.0  # meaningful, but not total, overlap
    circles = [((0.0, 0.0), r), ((d, 0.0), r)]
    lens_area = _circle_lens_area(r, r, d)
    circle_area = math.pi * r * r
    union_area = 2 * circle_area - lens_area
    expected_overlap_share = lens_area / union_area

    count = 200
    points = generate_union_points(circles, count=count, seed=3, distribution="grid")
    assert len(points) == count
    overlap_count = sum(
        1 for x, y in points
        if math.hypot(x - 0.0, y - 0.0) <= r and math.hypot(x - d, y - 0.0) <= r
    )
    actual_overlap_share = overlap_count / count

    # Double-sampling (the bug) would produce a share close to 2x the analytic expectation;
    # correct union-aware sampling should land close to the analytic expectation itself.
    assert actual_overlap_share < 1.5 * expected_overlap_share, (
        f"overlap share {actual_overlap_share:.3f} looks double-sampled relative to the "
        f"analytic expectation {expected_overlap_share:.3f} (lens_area={lens_area:.2f}, "
        f"union_area={union_area:.2f})"
    )
    assert abs(actual_overlap_share - expected_overlap_share) < 0.1


def test_generate_union_points_grid_respects_bounds_and_exclude_zones():
    circles = [((10.0, 10.0), 8.0), ((20.0, 10.0), 8.0)]
    zone = ExcludeZoneConfig(name="obstacle", shape="circle", center=(15.0, 10.0), radius=2.0)
    points = generate_union_points(
        circles, count=60, seed=0, distribution="grid", exclude_zones=[zone], bounds=(28.0, 20.0),
    )
    assert len(points) == 60
    for x, y in points:
        assert 0 <= x <= 28.0 and 0 <= y <= 20.0
        assert math.hypot(x - 15.0, y - 10.0) > 2.0


def test_generate_union_points_radial_raises_clear_not_yet_supported_error():
    with pytest.raises(ValueError, match="not yet supported"):
        generate_union_points([((0.0, 0.0), 10.0)], count=10, seed=0, distribution="radial")


def test_generate_union_points_variable_count_mode_searches_for_closest_count():
    circles = [((0.0, 0.0), 10.0), ((15.0, 0.0), 10.0)]
    points = generate_union_points(
        circles, count=60, seed=0, distribution="grid", count_mode="variable",
    )
    assert abs(len(points) - 60) < 15
    for x, y in points:
        assert any(math.hypot(x - cx, y - cy) <= r for (cx, cy), r in circles)


def test_generate_union_points_variable_count_mode_rejects_non_grid_distribution():
    with pytest.raises(ValueError, match="only supports distribution 'grid'"):
        generate_union_points(
            [((0.0, 0.0), 10.0)], count=10, seed=0, distribution="random", count_mode="variable",
        )


def test_generate_union_points_invalid_count_mode_raises():
    with pytest.raises(ValueError, match="Unknown count_mode"):
        generate_union_points([((0.0, 0.0), 10.0)], count=10, seed=0, count_mode="bogus")


def test_generate_union_points_empty_circles_raises():
    with pytest.raises(ValueError, match="non-empty"):
        generate_union_points([], count=10, seed=0)


def test_generate_union_points_border_width_too_large_for_every_circle_raises():
    with pytest.raises(ValueError, match="effective radius"):
        generate_union_points([((0.0, 0.0), 5.0)], count=10, seed=0, border_width=10.0)


def test_generate_union_points_border_width_drops_only_the_too_small_circles():
    # A border_width that eliminates the SMALLER circle but leaves the larger one valid
    # should still succeed, sampling only from the surviving circle.
    points = generate_union_points(
        [((0.0, 0.0), 2.0), ((20.0, 0.0), 10.0)], count=30, seed=0, border_width=3.0,
    )
    assert len(points) == 30
    for x, y in points:
        assert math.hypot(x - 20.0, y - 0.0) <= 10.0 - 3.0 + 1e-6


def test_build_pattern_dispatches_union_shape():
    pattern_config = PatternConfig(
        shape="union", circles=[((0.0, 0.0), 10.0), ((15.0, 0.0), 10.0)], seed=0,
        distribution="grid", border_width=0.0,
    )
    points = build_pattern(pattern_config, count=25)
    assert len(points) == 25


@pytest.mark.parametrize("distribution", ["grid", "radial"])
def test_generate_sector_points_variable_count_mode_returns_only_valid_points(distribution):
    points = generate_sector_points(
        (0, 0), 0, 10, 0, 360, count=30, seed=1, distribution=distribution, count_mode="variable",
    )
    assert len(points) > 0
    for x, y in points:
        assert math.hypot(x, y) <= 10 + 1e-6


def test_generate_sector_points_variable_count_mode_searches_for_closest_count():
    # A tiny inner circle relative to a huge bounding area: a single-shot resolution (density
    # == target) would land far short of 400 -- the density search should get much closer.
    points = generate_sector_points(
        (0, 0), 0, 2, 0, 360, count=400, seed=1, distribution="grid", count_mode="variable",
    )
    assert abs(len(points) - 400) < 20


def test_generate_sector_points_variable_count_mode_rejects_random():
    with pytest.raises(ValueError, match="count_mode"):
        generate_sector_points((0, 0), 0, 10, 0, 360, count=10, seed=1, distribution="random", count_mode="variable")


def test_generate_sector_points_invalid_count_mode_raises():
    with pytest.raises(ValueError, match="count_mode"):
        generate_sector_points((0, 0), 0, 10, 0, 360, count=10, seed=1, count_mode="bogus")


@pytest.mark.parametrize("method", ["uniform", "random"])
def test_generate_rotation_angles_returns_exactly_count(method):
    angles = generate_rotation_angles(7, method=method, seed=1)
    assert len(angles) == 7


def test_generate_rotation_angles_uniform_starts_at_user_angle_start():
    angles = generate_rotation_angles(4, method="uniform")
    assert angles == pytest.approx([0.0, 90.0, 180.0, 270.0])


def test_generate_rotation_angles_start_end_are_relative_to_initial_angle():
    angles = generate_rotation_angles(4, method="uniform", angle_start_deg=-45, angle_end_deg=45, initial_angle_deg=90)
    assert angles == pytest.approx([45.0, 75.0, 105.0, 135.0])


def test_generate_rotation_angles_uniform_spans_full_range_regardless_of_count():
    # The achieved spread shouldn't shrink/grow with count -- it should always cover the
    # literal [start, end] range exactly, for any non-wraparound span.
    for count in (2, 3, 5, 10):
        angles = generate_rotation_angles(count, method="uniform", angle_start_deg=0, angle_end_deg=90)
        assert angles[0] == pytest.approx(0.0)
        assert angles[-1] == pytest.approx(90.0)


def test_generate_rotation_angles_random_relative_to_initial_angle_stays_in_range():
    angles = generate_rotation_angles(20, method="random", angle_start_deg=-30, angle_end_deg=30, seed=1, initial_angle_deg=180)
    assert all(150.0 <= a <= 210.0 for a in angles)


def test_generate_rotation_angles_uniform_does_not_duplicate_the_wraparound_endpoint():
    # 0 and 360 are the same physical angle for a full rotation -- starting AT 0 is correct
    # (the user's chosen start), but 360 (the same point, redundantly) must not also appear.
    angles = generate_rotation_angles(6, method="uniform")
    assert 360.0 not in angles


def test_generate_rotation_angles_uniform_respects_partial_range():
    angles = generate_rotation_angles(3, method="uniform", angle_start_deg=0.0, angle_end_deg=90.0)
    for a in angles:
        assert 0.0 <= a <= 90.0


def test_generate_rotation_angles_random_within_range_and_deterministic():
    angles_a = generate_rotation_angles(10, method="random", angle_start_deg=10.0, angle_end_deg=50.0, seed=3)
    angles_b = generate_rotation_angles(10, method="random", angle_start_deg=10.0, angle_end_deg=50.0, seed=3)
    assert angles_a == angles_b
    for a in angles_a:
        assert 10.0 <= a <= 50.0


def test_generate_rotation_angles_different_seeds_differ():
    angles_a = generate_rotation_angles(10, method="random", seed=1)
    angles_b = generate_rotation_angles(10, method="random", seed=2)
    assert angles_a != angles_b


def test_generate_rotation_angles_invalid_count_raises():
    with pytest.raises(ValueError, match="count"):
        generate_rotation_angles(0)


def test_generate_rotation_angles_invalid_method_raises():
    with pytest.raises(ValueError, match="method"):
        generate_rotation_angles(4, method="spiral")


def test_target_for_episode_generalizes_to_floats():
    angles = [10.0, 20.0, 30.0]
    assert target_for_episode(angles, 0) == 10.0
    assert target_for_episode(angles, 4) == 20.0  # wraps: 4 % 3 == 1


def test_orientation_arrow_points_tail_is_always_the_given_position():
    position = (100.0, 50.0)
    tail, _ = orientation_arrow_points(position, angle_deg=37.0, length=20.0)
    assert tail == position


def test_orientation_arrow_points_no_homography_rotates_in_pixel_space_directly():
    position = (100.0, 50.0)
    # angle 0 points along +x (no homography == canonical space == pixel space).
    _, tip = orientation_arrow_points(position, angle_deg=0.0, length=10.0)
    assert tip == pytest.approx((110.0, 50.0))
    # angle 90 rotates toward +y (image y grows downward) -- same convention as
    # generate_arc_points.
    _, tip90 = orientation_arrow_points(position, angle_deg=90.0, length=10.0)
    assert tip90 == pytest.approx((100.0, 60.0))


def test_orientation_arrow_points_identity_homography_matches_no_homography_case():
    # Corners that exactly match a canonical square produce an (approximately) identity
    # homography, so projecting through it should reproduce the no-homography result.
    corners = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    homography = compute_homography(corners)
    position = (40.0, 60.0)
    tail_h, tip_h = orientation_arrow_points(position, angle_deg=30.0, length=15.0, homography=homography)
    tail_plain, tip_plain = orientation_arrow_points(position, angle_deg=30.0, length=15.0)
    assert tail_h == pytest.approx(tail_plain)
    assert tip_h == pytest.approx(tip_plain, abs=1e-6)


def test_orientation_arrow_points_homography_rotates_in_canonical_space_not_pixel_space():
    # A genuinely tilted/non-square set of corners makes canonical and pixel space disagree
    # about angles -- verify the tip is computed by rotating in CANONICAL space (recovered
    # via the inverse homography) and then forward-mapped, by recomputing that independently.
    corners = [(50.0, 20.0), (300.0, 10.0), (320.0, 200.0), (30.0, 210.0)]
    homography = compute_homography(corners)
    position = (150.0, 100.0)
    angle_deg, length = 65.0, 25.0

    _, tip = orientation_arrow_points(position, angle_deg, length, homography=homography)

    inverse = invert_homography(homography)
    cx, cy = apply_homography(inverse, position)
    theta = math.radians(angle_deg)
    expected_tip_canonical = (cx + length * math.cos(theta), cy + length * math.sin(theta))
    expected_tip = apply_homography(homography, expected_tip_canonical)
    assert tip == pytest.approx(expected_tip)

    # And it must NOT equal the naive (wrong) pixel-space rotation, confirming the function
    # is actually doing the canonical-space round trip rather than ignoring the homography.
    px, py = position
    naive_tip = (px + length * math.cos(theta), py + length * math.sin(theta))
    assert tip != pytest.approx(naive_tip, abs=1e-6)


# ===== Multi-object behavioral randomization (mulberry32, sequencing, rotation groups, levels) =====


def test_mulberry32_deterministic_given_seed():
    a = _mulberry32(42)
    b = _mulberry32(42)
    assert [a() for _ in range(10)] == [b() for _ in range(10)]


def test_mulberry32_different_seeds_differ():
    a = _mulberry32(1)
    b = _mulberry32(2)
    assert [a() for _ in range(5)] != [b() for _ in range(5)]


def test_mulberry32_draws_in_unit_interval():
    rng = _mulberry32(7)
    draws = [rng() for _ in range(1000)]
    assert all(0.0 <= d < 1.0 for d in draws)


def test_mulberry32_known_first_values_for_seed_zero():
    # Pins the exact algorithm (a literal port target for calibrate.js's mulberry32) against
    # regressions -- if this ever changes, the JS port must change identically or parity breaks.
    rng = _mulberry32(0)
    first, second = rng(), rng()
    assert first == pytest.approx(0.26642920868471265, abs=1e-12)
    assert second == pytest.approx(0.0003297457005828619, abs=1e-12)


def test_combine_seed_deterministic_and_order_sensitive():
    assert _combine_seed(1, 2, 3) == _combine_seed(1, 2, 3)
    assert _combine_seed(1, 2, 3) != _combine_seed(3, 2, 1)


def test_seeded_permutation_is_a_permutation():
    perm = _seeded_permutation(20, seed=5)
    assert sorted(perm) == list(range(20))


def test_seeded_permutation_deterministic_given_seed():
    assert _seeded_permutation(20, seed=5) == _seeded_permutation(20, seed=5)


def test_sequence_index_lockstep_matches_target_for_episode():
    for i in range(10):
        assert sequence_index(i, length=4, mode="lockstep") == i % 4


def test_sequence_index_unknown_mode_raises():
    with pytest.raises(ValueError):
        sequence_index(0, length=4, mode="bogus")


def test_sequence_index_rejects_zero_length():
    with pytest.raises(ValueError):
        sequence_index(0, length=0)


@pytest.mark.parametrize("mode", ["lockstep", "shuffled"])
def test_sequence_index_visits_every_point_evenly_over_one_period(mode):
    length = 7
    visited = [
        sequence_index(i, length, mode=mode, object_ordinal=2, num_objects=3, seed=11)
        for i in range(length)
    ]
    assert sorted(visited) == list(range(length))


def test_sequence_index_shuffled_differs_from_lockstep_for_some_seed():
    length = 9
    lockstep = [sequence_index(i, length, mode="lockstep") for i in range(length)]
    shuffled = [sequence_index(i, length, mode="shuffled", object_ordinal=1, seed=3) for i in range(length)]
    assert shuffled != lockstep


def test_combination_index_rejects_zero_length():
    with pytest.raises(ValueError):
        combination_index(0, length=0, object_ordinal=0, num_combinations=5)


def test_combination_index_rejects_zero_num_combinations():
    with pytest.raises(ValueError):
        combination_index(0, length=4, object_ordinal=0, num_combinations=0)


def test_combination_index_rejects_a_joint_only_mode():
    # "cartesian"/"lcm" need every object's length jointly -- only resolve_combination_indices
    # handles them, never this per-object function.
    with pytest.raises(ValueError):
        combination_index(0, length=4, object_ordinal=0, num_combinations=5, mode="cartesian")
    with pytest.raises(ValueError):
        combination_index(0, length=4, object_ordinal=0, num_combinations=5, mode="lcm")


@pytest.mark.parametrize("mode", ["systematic", "random", "coprime"])
def test_combination_index_always_in_range(mode):
    length, num_combinations = 5, 17
    for i in range(num_combinations):
        for ordinal in range(3):
            idx = combination_index(i, length, ordinal, num_combinations, mode=mode, seed=11)
            assert 0 <= idx < length


def test_combination_index_systematic_undersampling_spreads_evenly_no_duplicates():
    # num_combinations <= length: each combination should land on a DIFFERENT point, evenly
    # spread across the larger pool (skipping some) -- the proportional formula's intended use.
    length, num_combinations = 20, 10
    indices = [combination_index(i, length, object_ordinal=0, num_combinations=num_combinations) for i in range(num_combinations)]
    assert len(set(indices)) == num_combinations  # all distinct
    assert indices == sorted(indices)  # monotonically increasing -- evenly spread, not shuffled


def test_combination_index_systematic_oversampling_never_stalls_on_consecutive_episodes():
    # The reported bug: setting num_combinations higher than an object's own pattern length
    # made consecutive episodes repeat the SAME point for several episodes in a row (the
    # proportional formula's floor-division naturally clusters when the step size is < 1)
    # instead of cycling. Fixed via a plain-modulo branch once num_combinations > length.
    length, num_combinations = 4, 20
    indices = [combination_index(i, length, object_ordinal=0, num_combinations=num_combinations) for i in range(num_combinations)]
    for a, b in zip(indices, indices[1:]):
        assert a != b, f"consecutive episodes repeated index {a} -- the stalling bug is back"


def test_combination_index_systematic_oversampling_still_evenly_counts_each_point():
    # The bug fix shouldn't sacrifice aggregate fairness -- still num_combinations/length visits
    # per point when num_combinations is an exact multiple of length.
    length, num_combinations = 4, 20
    counts = [0] * length
    for i in range(num_combinations):
        counts[combination_index(i, length, object_ordinal=0, num_combinations=num_combinations)] += 1
    assert all(c == num_combinations // length for c in counts)


def test_combination_index_systematic_decouples_combination_count_from_length():
    # The whole point: num_combinations can exceed (or be less than) length -- unlike
    # sequence_index's lockstep, which is hard-capped at `length` distinct values per period.
    length = 5
    values_at_20 = {combination_index(i, length, 0, 20) for i in range(20)}
    assert values_at_20 == set(range(length))  # all 5 points visited well before 20 combos


def test_combination_index_systematic_phase_shifts_equal_length_objects_apart():
    # Two objects with the SAME length, no phase shift, would always land on the same
    # proportional point at the same combination -- the phase (object_ordinal) decorrelates
    # them so they don't always pair the same way.
    length, num_combinations = 6, 6
    ordinal0 = [combination_index(i, length, 0, num_combinations) for i in range(num_combinations)]
    ordinal1 = [combination_index(i, length, 1, num_combinations) for i in range(num_combinations)]
    assert ordinal0 != ordinal1


def test_combination_index_systematic_undersampling_pure_integer_arithmetic():
    # Within the undersampling (num_combinations <= length) branch, combination_i * length is
    # exactly divisible by num_combinations here (an exact half-integer boundary would be the
    # classic round()/Math.round() divergence point) -- confirms floor division alone (no
    # round()) is used, by checking an exact value.
    assert combination_index(1, length=8, object_ordinal=0, num_combinations=4) == 2  # 1*8//4 = 2
    assert combination_index(2, length=8, object_ordinal=0, num_combinations=4) == 4  # 2*8//4 = 4


def test_combination_index_random_deterministic_given_seed():
    a = [combination_index(i, 7, 0, 1, mode="random", seed=42) for i in range(30)]
    b = [combination_index(i, 7, 0, 1, mode="random", seed=42) for i in range(30)]
    assert a == b


def test_combination_index_random_covers_full_range_given_enough_draws():
    length = 5
    draws = {combination_index(i, length, object_ordinal=0, num_combinations=1, mode="random", seed=3) for i in range(200)}
    assert draws == set(range(length))


def test_combination_index_random_differs_by_seed():
    a = [combination_index(i, 6, 0, 1, mode="random", seed=1) for i in range(20)]
    b = [combination_index(i, 6, 0, 1, mode="random", seed=2) for i in range(20)]
    assert a != b


def test_gcd_basic():
    assert _gcd(12, 8) == 4
    assert _gcd(7, 5) == 1
    assert _gcd(0, 5) == 5


@pytest.mark.parametrize("length", [2, 3, 4, 5, 7, 11, 12, 20])
@pytest.mark.parametrize("ordinal", [0, 1, 2, 5])
def test_coprime_stride_for_is_always_actually_coprime(length, ordinal):
    stride = _coprime_stride_for(length, ordinal)
    assert _gcd(stride, length) == 1


def test_combination_index_coprime_is_a_permutation_within_one_period():
    length = 9
    values = [combination_index(i, length, object_ordinal=0, num_combinations=1, mode="coprime") for i in range(length)]
    assert sorted(values) == list(range(length))


def test_combination_index_coprime_is_fully_deterministic_no_randomness_involved():
    length, num_combinations = 7, 7
    a = [combination_index(i, length, 2, num_combinations, mode="coprime") for i in range(length)]
    b = [combination_index(i, length, 2, num_combinations, mode="coprime") for i in range(length)]
    assert a == b


def test_combination_index_coprime_differs_by_ordinal_typically():
    length = 8
    ordinal0 = [combination_index(i, length, 0, 1, mode="coprime") for i in range(length)]
    ordinal1 = [combination_index(i, length, 1, 1, mode="coprime") for i in range(length)]
    assert ordinal0 != ordinal1


def test_cartesian_indices_enumerates_full_product_exactly_once():
    lengths = [2, 3]
    seen = set()
    for i in range(2 * 3):
        seen.add(tuple(_cartesian_indices(i, lengths)))
    assert len(seen) == 6
    assert seen == {(a, b) for a in range(2) for b in range(3)}


def test_cartesian_indices_wraps_after_full_product():
    lengths = [2, 3]
    assert _cartesian_indices(0, lengths) == _cartesian_indices(6, lengths)


def test_resolve_combination_indices_cartesian_matches_cartesian_indices_directly():
    lengths = [3, 4, 2]
    for i in range(24):
        assert resolve_combination_indices(i, lengths, mode="cartesian") == _cartesian_indices(i, lengths)


def test_resolve_combination_indices_lcm_is_plain_modulo_no_phase():
    lengths = [4, 6]
    for i in range(12):
        assert resolve_combination_indices(i, lengths, mode="lcm") == [i % 4, i % 6]


def test_resolve_combination_indices_lcm_completes_every_lockstep_pairing_once():
    # lcm(4, 6) = 12 -- over exactly 12 combinations, every (i%4, i%6) pair that plain lockstep
    # naturally produces should appear, with no repeat before the 12th.
    lengths = [4, 6]
    pairs = [tuple(resolve_combination_indices(i, lengths, mode="lcm")) for i in range(12)]
    assert len(set(pairs)) == 12


def test_resolve_combination_indices_systematic_delegates_to_combination_index():
    lengths = [5, 7]
    for i in range(10):
        expected = [combination_index(i, length, ordinal, 10, mode="systematic") for ordinal, length in enumerate(lengths)]
        assert resolve_combination_indices(i, lengths, mode="systematic", num_combinations=10) == expected


def test_resolve_combination_indices_requires_count_for_systematic_random_coprime():
    with pytest.raises(ValueError):
        resolve_combination_indices(0, [4, 5], mode="systematic", num_combinations=None)
    with pytest.raises(ValueError):
        resolve_combination_indices(0, [4, 5], mode="random", num_combinations=None)
    with pytest.raises(ValueError):
        resolve_combination_indices(0, [4, 5], mode="coprime", num_combinations=None)


def test_combination_period_unknown_mode_raises():
    with pytest.raises(ValueError):
        combination_period([4, 5], mode="bogus")


def test_combination_period_systematic_random_coprime_require_count():
    for mode in ("systematic", "random", "coprime"):
        with pytest.raises(ValueError):
            combination_period([4, 5], mode=mode, combination_count=None)
        assert combination_period([4, 5], mode=mode, combination_count=50) == 50


def test_combination_period_cartesian_natural_is_product_of_lengths():
    assert combination_period([4, 5, 2], mode="cartesian", combination_count=None) == 40


def test_combination_period_cartesian_capped_by_combination_count():
    assert combination_period([4, 5, 2], mode="cartesian", combination_count=10) == 10
    # cap above the natural product is a no-op, not an error
    assert combination_period([4, 5, 2], mode="cartesian", combination_count=1000) == 40


def test_combination_period_lcm_natural_is_lcm_of_lengths():
    assert combination_period([4, 6], mode="lcm", combination_count=None) == 12


def test_combination_period_lcm_capped_by_combination_count():
    assert combination_period([4, 6], mode="lcm", combination_count=5) == 5


def test_occupied_sites_disjoint_objects_one_site_each():
    sites = occupied_sites([[(1.0, 1.0)], [(2.0, 2.0)]])
    assert sites == [((1.0, 1.0), [0]), ((2.0, 2.0), [1])]


def test_occupied_sites_shared_point_forms_a_guaranteed_multi_member_site():
    sites = occupied_sites([[(5.0, 5.0)], [(5.0, 5.0)]])
    assert sites == [((5.0, 5.0), [0, 1])]


def test_occupied_sites_orders_by_earliest_object_ordinal_then_point_index():
    # B (ordinal 1) is declared after A (ordinal 0), so A's points sort first; within A's own
    # points, the earlier point_index sorts first.
    sites = occupied_sites([[(10.0, 10.0), (20.0, 20.0)], [(30.0, 30.0)]])
    assert [p for p, _ in sites] == [(10.0, 10.0), (20.0, 20.0), (30.0, 30.0)]


def test_occupied_sites_a_shared_point_is_ordered_by_its_earliest_member():
    # The shared point (50,50) appears at A's index 0 and B's index 0 -- its order key is
    # min((0, 0), (1, 0)) = (0, 0), so it sorts before A's OTHER, later point (5,5) at (0, 1).
    sites = occupied_sites([[(50.0, 50.0), (5.0, 5.0)], [(50.0, 50.0)]])
    assert [p for p, _ in sites] == [(50.0, 50.0), (5.0, 5.0)]
    assert sites[0][1] == [0, 1]


def test_occupied_sites_empty_when_no_object_has_any_point():
    assert occupied_sites([[], []]) == []


def test_resolve_keep_apart_disjoint_sets_untouched():
    natural = [0, 0]  # would collide if these were the same list, but they aren't
    assigned = [[0, 1], [0, 1]]
    resolved, residual = resolve_keep_apart(natural, assigned, episode_index=0)
    # First object (declared first) keeps its natural point; second collides and must move.
    assert resolved[0] == 0
    assert resolved[1] != 0
    assert residual == []


def test_resolve_keep_apart_truly_disjoint_assigned_sets_never_perturbed():
    natural = [0, 2]  # no collision at all -- different points entirely
    assigned = [[0, 1], [2, 3]]
    resolved, residual = resolve_keep_apart(natural, assigned, episode_index=5)
    assert resolved == [0, 2]
    assert residual == []


def test_resolve_keep_apart_moves_second_object_to_its_own_free_point():
    natural = [0, 0]
    assigned = [[0, 1], [0, 2]]
    resolved, residual = resolve_keep_apart(natural, assigned, episode_index=0)
    assert resolved[0] == 0
    assert resolved[1] == 2
    assert residual == []


def test_resolve_keep_apart_residual_when_no_alternative_exists():
    # Both objects have ONLY point 0 -- there's nowhere for the second one to go.
    natural = [0, 0]
    assigned = [[0], [0]]
    resolved, residual = resolve_keep_apart(natural, assigned, episode_index=0)
    assert resolved == [0, 0]
    assert residual == [1]


def test_resolve_keep_apart_skips_absent_objects():
    natural = [0, None, 0]
    assigned = [[0, 1], [5], [0, 2]]
    resolved, residual = resolve_keep_apart(natural, assigned, episode_index=0)
    assert resolved[1] is None
    assert resolved[0] == 0
    assert resolved[2] != 0
    assert residual == []


def test_resolve_keep_apart_deterministic_given_episode_index():
    natural = [0, 0, 0]
    assigned = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
    a = resolve_keep_apart(natural, assigned, episode_index=3)
    b = resolve_keep_apart(natural, assigned, episode_index=3)
    assert a == b


def test_level_order_fixed_is_identity():
    assert level_order(4, episode_index=5, strategy="fixed") == [0, 1, 2, 3]


def test_level_order_unknown_strategy_raises():
    with pytest.raises(ValueError):
        level_order(3, 0, strategy="bogus")


def test_level_order_rejects_zero_stack_size():
    with pytest.raises(ValueError):
        level_order(0, 0)


def test_level_order_shuffle_is_a_permutation():
    order = level_order(5, episode_index=3, strategy="shuffle", seed=2)
    assert sorted(order) == list(range(5))


@pytest.mark.parametrize("strategy", ["cycle", "balanced"])
def test_level_order_cycle_and_balanced_are_latin_squares(strategy):
    stack_size = 4
    # For a fixed slot, the levels it visits across one full period must all be distinct.
    for slot in range(stack_size):
        levels_for_slot = {
            level_order(stack_size, episode_index=e, strategy=strategy, seed=7)[slot]
            for e in range(stack_size)
        }
        assert levels_for_slot == set(range(stack_size))


def test_level_order_balanced_seeded_start_differs_from_cycle():
    stack_size = 5
    cycle = [level_order(stack_size, e, strategy="cycle") for e in range(stack_size)]
    balanced = [level_order(stack_size, e, strategy="balanced", seed=13) for e in range(stack_size)]
    assert balanced != cycle


def test_build_pattern_dispatches_points_shape_returns_as_is():
    pts = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
    cfg = PatternConfig(shape="points", points=pts)
    assert build_pattern(cfg, count=3) == pts
