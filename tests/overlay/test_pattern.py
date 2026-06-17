import pytest

from vizaudit.overlay.config import PatternConfig
from vizaudit.overlay.pattern import (
    build_pattern,
    generate_arc_points,
    generate_line_points,
    target_for_episode,
)


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
