import math
from unittest.mock import patch

from vizaudit.overlay.config import MarkerConfig
from vizaudit.overlay.rerun_client import STACK_FAN_RADIUS, _stack_offset_position, clear_target, log_target

CAMERA_KEY = "observation.images.top"
MARKER = MarkerConfig(radius_px=8.0, color_rgba=(1, 2, 3, 4), label=True)


def test_stack_offset_position_is_a_noop_for_a_lone_marker():
    assert _stack_offset_position((10.0, 20.0), level=0, stack_size=1) == (10.0, 20.0)


def test_stack_offset_position_fans_around_point_by_level():
    point = (0.0, 0.0)
    a = _stack_offset_position(point, level=0, stack_size=2)
    b = _stack_offset_position(point, level=1, stack_size=2)
    assert a != b
    assert math.hypot(*a) == math.hypot(*b) == STACK_FAN_RADIUS


def test_log_target_default_level_and_stack_size_apply_no_offset():
    with patch("vizaudit.overlay.rerun_client.rr") as mock_rr:
        log_target(CAMERA_KEY, (10.0, 20.0), "cube", MARKER)
        path, points2d_kwargs = mock_rr.log.call_args[0][0], mock_rr.Points2D.call_args.kwargs
        assert path == f"{CAMERA_KEY}/target/cube"
        assert points2d_kwargs["positions"] == [(10.0, 20.0)]
        assert points2d_kwargs["labels"] == ["cube"]


def test_log_target_stacked_offsets_position_and_labels_with_level():
    with patch("vizaudit.overlay.rerun_client.rr") as mock_rr:
        log_target(CAMERA_KEY, (10.0, 20.0), "cube", MARKER, level=1, stack_size=2)
        points2d_kwargs = mock_rr.Points2D.call_args.kwargs
        expected_point = _stack_offset_position((10.0, 20.0), 1, 2)
        assert points2d_kwargs["positions"] == [expected_point]
        assert points2d_kwargs["labels"] == ["cube L1"]


def test_log_target_no_label_when_marker_label_false():
    unlabeled = MarkerConfig(radius_px=8.0, color_rgba=(1, 2, 3, 4), label=False)
    with patch("vizaudit.overlay.rerun_client.rr") as mock_rr:
        log_target(CAMERA_KEY, (1.0, 2.0), "cube", unlabeled, level=1, stack_size=2)
        assert mock_rr.Points2D.call_args.kwargs["labels"] is None


def test_log_target_orientation_arrow_translated_by_same_fan_offset():
    point = (10.0, 20.0)
    tip = (15.0, 20.0)
    with patch("vizaudit.overlay.rerun_client.rr") as mock_rr:
        log_target(CAMERA_KEY, point, "cube", MARKER, orientation_tip=tip, level=1, stack_size=2)
        arrows_kwargs = mock_rr.Arrows2D.call_args.kwargs
        expected_origin = _stack_offset_position(point, 1, 2)
        assert arrows_kwargs["origins"] == [expected_origin]
        assert arrows_kwargs["vectors"] == [(tip[0] - point[0], tip[1] - point[1])]


def test_log_target_no_orientation_call_when_tip_omitted():
    with patch("vizaudit.overlay.rerun_client.rr") as mock_rr:
        log_target(CAMERA_KEY, (1.0, 2.0), "cube", MARKER)
        mock_rr.Arrows2D.assert_not_called()


def test_log_target_extra_placements_logs_all_positions_in_one_call():
    primary = (1.0, 1.0)
    extra = [(2.0, 2.0), (3.0, 3.0)]
    with patch("vizaudit.overlay.rerun_client.rr") as mock_rr:
        log_target(
            CAMERA_KEY, primary, "cube", MARKER,
            extra_placements=[(extra[0], None, 0, 1), (extra[1], None, 0, 1)],
        )
        positions = mock_rr.Points2D.call_args.kwargs["positions"]
        assert positions == [primary, extra[0], extra[1]]
        mock_rr.log.assert_called_once()  # ONE entity, not 3 separate log calls


def test_log_target_extra_placements_each_get_their_own_label_and_fan_offset():
    with patch("vizaudit.overlay.rerun_client.rr") as mock_rr:
        log_target(
            CAMERA_KEY, (0.0, 0.0), "cube", MARKER, level=0, stack_size=2,
            extra_placements=[((10.0, 10.0), None, 1, 2)],
        )
        kwargs = mock_rr.Points2D.call_args.kwargs
        assert kwargs["labels"] == ["cube L0", "cube L1"]
        assert kwargs["positions"][0] == _stack_offset_position((0.0, 0.0), 0, 2)
        assert kwargs["positions"][1] == _stack_offset_position((10.0, 10.0), 1, 2)


def test_log_target_extra_placements_orientation_arrows_skip_missing_tips():
    with patch("vizaudit.overlay.rerun_client.rr") as mock_rr:
        log_target(
            CAMERA_KEY, (0.0, 0.0), "cube", MARKER, orientation_tip=(5.0, 0.0),
            extra_placements=[((10.0, 10.0), None, 0, 1), ((20.0, 20.0), (25.0, 20.0), 0, 1)],
        )
        arrows_kwargs = mock_rr.Arrows2D.call_args.kwargs
        # Only the 2 placements WITH a tip (primary + the third extra) get an arrow.
        assert len(arrows_kwargs["origins"]) == 2
        assert arrows_kwargs["vectors"] == [(5.0, 0.0), (5.0, 0.0)]


def test_log_target_no_extra_placements_is_byte_for_byte_unchanged():
    # The default (omitted extra_placements) call shape, byte-for-byte against a plain single
    # placement -- confirms the multi-placement mechanism is a pure additive capability.
    with patch("vizaudit.overlay.rerun_client.rr") as mock_rr1:
        log_target(CAMERA_KEY, (1.0, 2.0), "cube", MARKER, level=1, stack_size=2)
        kwargs1 = dict(mock_rr1.Points2D.call_args.kwargs)
    with patch("vizaudit.overlay.rerun_client.rr") as mock_rr2:
        log_target(CAMERA_KEY, (1.0, 2.0), "cube", MARKER, level=1, stack_size=2, extra_placements=None)
        kwargs2 = dict(mock_rr2.Points2D.call_args.kwargs)
    assert kwargs1 == kwargs2


def test_clear_target_logs_recursive_clear_at_object_path():
    with patch("vizaudit.overlay.rerun_client.rr") as mock_rr:
        clear_target(CAMERA_KEY, "cube")
        path = mock_rr.log.call_args[0][0]
        assert path == f"{CAMERA_KEY}/target/cube"
        mock_rr.Clear.assert_called_once_with(recursive=True)
