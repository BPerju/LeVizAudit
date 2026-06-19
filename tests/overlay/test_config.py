from pathlib import Path

import yaml

import pytest

from vizaudit.overlay.config import ConfigError, load_config

VALID = {
    "camera_key": "observation.images.top",
    "objects": [
        {
            "name": "cube",
            "count": 12,
            "variable": True,
            "pattern": {
                "shape": "arc",
                "center": [320, 240],
                "radius": 150,
                "angle_start_deg": 0,
                "angle_end_deg": 180,
            },
        },
        {"name": "distractor", "count": 1, "variable": False, "pattern": None},
    ],
    "marker": {"radius_px": 10, "color_rgba": [255, 64, 64, 255], "label": True},
}


def write_config(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_load_valid_config(tmp_path):
    config = load_config(write_config(tmp_path, VALID))
    assert config.camera_key == "observation.images.top"
    assert len(config.objects) == 2

    cube = config.objects[0]
    assert cube.name == "cube"
    assert cube.variable is True
    assert cube.pattern.shape == "arc"
    assert cube.pattern.center == (320.0, 240.0)

    distractor = config.objects[1]
    assert distractor.variable is False
    assert distractor.pattern is None


def test_load_config_defaults_marker_when_omitted(tmp_path):
    data = {k: v for k, v in VALID.items() if k != "marker"}
    config = load_config(write_config(tmp_path, data))
    assert config.marker.radius_px == 10.0
    assert config.marker.color_rgba == (255, 64, 64, 255)


def test_object_without_marker_override_inherits_the_global_marker(tmp_path):
    config = load_config(write_config(tmp_path, VALID))
    cube = config.objects[0]
    assert cube.marker == config.marker


def test_object_marker_override_only_changes_specified_fields(tmp_path):
    # A bimanual setup's two objects typically only need to differ in color_rgba -- radius_px
    # and label should keep tracking the top-level default unless an object overrides those
    # too, so an operator doesn't have to repeat the whole marker block per object.
    data = {
        **VALID,
        "objects": [
            {**VALID["objects"][0], "marker": {"color_rgba": [0, 0, 255, 255]}},
            VALID["objects"][1],
        ],
    }
    config = load_config(write_config(tmp_path, data))
    cube = config.objects[0]
    assert cube.marker.color_rgba == (0, 0, 255, 255)
    assert cube.marker.radius_px == config.marker.radius_px
    assert cube.marker.label == config.marker.label
    # The second (variable: false) object never had a marker override -- still inherits the
    # global default, same as before this feature existed.
    assert config.objects[1].marker == config.marker


def test_object_marker_override_with_no_top_level_marker_falls_back_to_hardcoded_default(tmp_path):
    data = {k: v for k, v in VALID.items() if k != "marker"}
    data["objects"] = [
        {**VALID["objects"][0], "marker": {"radius_px": 6}},
        VALID["objects"][1],
    ]
    config = load_config(write_config(tmp_path, data))
    cube = config.objects[0]
    assert cube.marker.radius_px == 6.0
    assert cube.marker.color_rgba == (255, 64, 64, 255)  # MarkerConfig()'s hardcoded default


def test_object_marker_unknown_field_raises(tmp_path):
    data = {
        **VALID,
        "objects": [
            {**VALID["objects"][0], "marker": {"colour_rgba": [0, 0, 255, 255]}},  # typo'd
            VALID["objects"][1],
        ],
    }
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, data))


def test_variable_object_missing_pattern_raises(tmp_path):
    data = {
        "camera_key": "observation.images.top",
        "objects": [{"name": "cube", "count": 1, "variable": True}],
    }
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, data))


def test_variable_false_with_pattern_raises(tmp_path):
    data = {
        "camera_key": "observation.images.top",
        "objects": [
            {
                "name": "cube",
                "count": 1,
                "variable": False,
                "pattern": {
                    "shape": "arc",
                    "center": [0, 0],
                    "radius": 1,
                    "angle_start_deg": 0,
                    "angle_end_deg": 90,
                },
            }
        ],
    }
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, data))


def test_unknown_shape_field_raises(tmp_path):
    data = {
        "camera_key": "observation.images.top",
        "objects": [
            {
                "name": "cube",
                "count": 1,
                "variable": True,
                "pattern": {
                    "shape": "arc",
                    "center": [0, 0],
                    "radius": 1,
                    "angle_start_deg": 0,
                    "angle_end_deg": 90,
                    "radius_range": [1, 2],  # typo'd unknown field
                },
            }
        ],
    }
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, data))


def test_duplicate_object_names_raises(tmp_path):
    data = {
        "camera_key": "observation.images.top",
        "objects": [
            {"name": "cube", "count": 1, "variable": False, "pattern": None},
            {"name": "cube", "count": 1, "variable": False, "pattern": None},
        ],
    }
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, data))


def test_unknown_pattern_shape_raises(tmp_path):
    data = {
        "camera_key": "observation.images.top",
        "objects": [
            {"name": "cube", "count": 1, "variable": True, "pattern": {"shape": "triangle"}}
        ],
    }
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, data))


def test_load_config_with_sector_pattern(tmp_path):
    data = {
        "camera_key": "observation.images.top",
        "objects": [
            {
                "name": "marble", "count": 5, "variable": True,
                "pattern": {
                    "shape": "sector", "center": [100, 100], "radius": 50,
                    "inner_radius": 10, "angle_start_deg": 0, "angle_end_deg": 90, "seed": 7,
                },
            }
        ],
    }
    config = load_config(write_config(tmp_path, data))
    pattern = config.objects[0].pattern
    assert pattern.shape == "sector"
    assert pattern.inner_radius == 10.0
    assert pattern.seed == 7


def test_sector_pattern_defaults_inner_radius_and_seed(tmp_path):
    data = {
        "camera_key": "observation.images.top",
        "objects": [
            {
                "name": "marble", "count": 5, "variable": True,
                "pattern": {"shape": "sector", "center": [0, 0], "radius": 50,
                            "angle_start_deg": 0, "angle_end_deg": 90},
            }
        ],
    }
    config = load_config(write_config(tmp_path, data))
    pattern = config.objects[0].pattern
    assert pattern.inner_radius == 0.0
    assert pattern.seed == 0


def test_sector_inner_radius_ge_radius_raises(tmp_path):
    data = {
        "camera_key": "observation.images.top",
        "objects": [
            {
                "name": "marble", "count": 5, "variable": True,
                "pattern": {"shape": "sector", "center": [0, 0], "radius": 50,
                            "inner_radius": 50, "angle_start_deg": 0, "angle_end_deg": 90},
            }
        ],
    }
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, data))


def test_sector_unknown_field_raises(tmp_path):
    data = {
        "camera_key": "observation.images.top",
        "objects": [
            {
                "name": "marble", "count": 5, "variable": True,
                "pattern": {"shape": "sector", "center": [0, 0], "radius": 50,
                            "angle_start_deg": 0, "angle_end_deg": 90, "bogus": 1},
            }
        ],
    }
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, data))


def test_load_config_with_exclude_zones(tmp_path):
    data = dict(VALID)
    data["exclude_zones"] = [
        {"name": "robot_base", "center": [10, 10], "radius": 5},
        {"name": "cup", "center": [50, 50], "radius": 3},
    ]
    config = load_config(write_config(tmp_path, data))
    assert len(config.exclude_zones) == 2
    assert config.exclude_zones[0].name == "robot_base"
    assert config.exclude_zones[0].radius == 5.0


def test_exclude_zones_defaults_empty_when_omitted(tmp_path):
    config = load_config(write_config(tmp_path, VALID))
    assert config.exclude_zones == []


def test_exclude_zones_duplicate_names_raises(tmp_path):
    data = dict(VALID)
    data["exclude_zones"] = [
        {"name": "dup", "center": [0, 0], "radius": 1},
        {"name": "dup", "center": [1, 1], "radius": 1},
    ]
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, data))


def test_exclude_zone_non_positive_radius_raises(tmp_path):
    data = dict(VALID)
    data["exclude_zones"] = [{"name": "zero", "center": [0, 0], "radius": 0}]
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, data))


def test_exclude_zone_unknown_field_raises(tmp_path):
    data = dict(VALID)
    data["exclude_zones"] = [{"name": "z", "center": [0, 0], "radius": 1, "bogus": 1}]
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, data))


def test_exclude_zone_polygon(tmp_path):
    data = dict(VALID)
    data["exclude_zones"] = [
        {"name": "robot_base", "shape": "polygon", "vertices": [[0, 0], [10, 0], [10, 10], [0, 10]]}
    ]
    config = load_config(write_config(tmp_path, data))
    zone = config.exclude_zones[0]
    assert zone.shape == "polygon"
    assert zone.vertices == [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    assert zone.center is None
    assert zone.radius is None


def test_exclude_zone_polygon_too_few_vertices_raises(tmp_path):
    data = dict(VALID)
    data["exclude_zones"] = [{"name": "z", "shape": "polygon", "vertices": [[0, 0], [10, 0]]}]
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, data))


def test_exclude_zone_polygon_with_center_raises(tmp_path):
    data = dict(VALID)
    data["exclude_zones"] = [
        {
            "name": "z", "shape": "polygon",
            "vertices": [[0, 0], [10, 0], [10, 10]], "center": [5, 5],
        }
    ]
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, data))


def test_exclude_zone_unknown_shape_raises(tmp_path):
    data = dict(VALID)
    data["exclude_zones"] = [{"name": "z", "shape": "hexagon", "center": [0, 0], "radius": 1}]
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, data))


def test_load_config_with_sector_distribution_and_border_width(tmp_path):
    data = {
        "camera_key": "observation.images.top",
        "objects": [
            {
                "name": "marble", "count": 5, "variable": True,
                "pattern": {
                    "shape": "sector", "center": [0, 0], "radius": 50,
                    "angle_start_deg": 0, "angle_end_deg": 90,
                    "distribution": "random", "border_width": 5,
                },
            }
        ],
    }
    config = load_config(write_config(tmp_path, data))
    pattern = config.objects[0].pattern
    assert pattern.distribution == "random"
    assert pattern.border_width == 5.0
    assert pattern.count_mode == "fixed"


def test_sector_count_mode_variable_with_grid(tmp_path):
    data = dict(VALID)
    data["objects"] = [dict(VALID["objects"][0], pattern={
        "shape": "sector", "center": [0, 0], "radius": 50,
        "angle_start_deg": 0, "angle_end_deg": 90, "distribution": "grid", "count_mode": "variable",
    }), VALID["objects"][1]]
    config = load_config(write_config(tmp_path, data))
    assert config.objects[0].pattern.count_mode == "variable"


def test_sector_count_mode_variable_with_random_raises(tmp_path):
    data = dict(VALID)
    data["objects"] = [dict(VALID["objects"][0], pattern={
        "shape": "sector", "center": [0, 0], "radius": 50,
        "angle_start_deg": 0, "angle_end_deg": 90, "distribution": "random", "count_mode": "variable",
    }), VALID["objects"][1]]
    with pytest.raises(ConfigError, match="count_mode"):
        load_config(write_config(tmp_path, data))


def test_sector_invalid_count_mode_raises(tmp_path):
    data = dict(VALID)
    data["objects"] = [dict(VALID["objects"][0], pattern={
        "shape": "sector", "center": [0, 0], "radius": 50,
        "angle_start_deg": 0, "angle_end_deg": 90, "count_mode": "bogus",
    }), VALID["objects"][1]]
    with pytest.raises(ConfigError, match="count_mode"):
        load_config(write_config(tmp_path, data))


def test_union_count_mode_variable_with_grid(tmp_path):
    data = dict(VALID)
    data["objects"] = [dict(VALID["objects"][0], pattern={
        "shape": "union", "circles": [{"center": [0, 0], "radius": 10}],
        "distribution": "grid", "count_mode": "variable",
    }), VALID["objects"][1]]
    config = load_config(write_config(tmp_path, data))
    assert config.objects[0].pattern.count_mode == "variable"


def test_union_count_mode_variable_with_random_raises(tmp_path):
    data = dict(VALID)
    data["objects"] = [dict(VALID["objects"][0], pattern={
        "shape": "union", "circles": [{"center": [0, 0], "radius": 10}],
        "distribution": "random", "count_mode": "variable",
    }), VALID["objects"][1]]
    with pytest.raises(ConfigError, match="count_mode"):
        load_config(write_config(tmp_path, data))


def test_sector_distribution_defaults_to_grid(tmp_path):
    data = {
        "camera_key": "observation.images.top",
        "objects": [
            {
                "name": "marble", "count": 5, "variable": True,
                "pattern": {"shape": "sector", "center": [0, 0], "radius": 50,
                            "angle_start_deg": 0, "angle_end_deg": 90},
            }
        ],
    }
    config = load_config(write_config(tmp_path, data))
    pattern = config.objects[0].pattern
    assert pattern.distribution == "grid"
    assert pattern.border_width == 0.0


def test_sector_invalid_distribution_raises(tmp_path):
    data = {
        "camera_key": "observation.images.top",
        "objects": [
            {
                "name": "marble", "count": 5, "variable": True,
                "pattern": {"shape": "sector", "center": [0, 0], "radius": 50,
                            "angle_start_deg": 0, "angle_end_deg": 90, "distribution": "bogus"},
            }
        ],
    }
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, data))


def test_sector_negative_border_width_raises(tmp_path):
    data = {
        "camera_key": "observation.images.top",
        "objects": [
            {
                "name": "marble", "count": 5, "variable": True,
                "pattern": {"shape": "sector", "center": [0, 0], "radius": 50,
                            "angle_start_deg": 0, "angle_end_deg": 90, "border_width": -1},
            }
        ],
    }
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, data))


def test_load_config_with_surface_calibration(tmp_path):
    data = dict(VALID)
    data["surface_calibration"] = {
        "corners": [[10, 10], [110, 10], [110, 60], [10, 60]],
    }
    config = load_config(write_config(tmp_path, data))
    assert config.surface_calibration is not None
    assert len(config.surface_calibration.corners) == 4
    assert config.surface_calibration.aspect_ratio is None


def test_surface_calibration_defaults_to_none_when_omitted(tmp_path):
    config = load_config(write_config(tmp_path, VALID))
    assert config.surface_calibration is None


def test_surface_calibration_with_aspect_ratio(tmp_path):
    data = dict(VALID)
    data["surface_calibration"] = {
        "corners": [[10, 10], [110, 10], [110, 60], [10, 60]],
        "aspect_ratio": 1.5,
    }
    config = load_config(write_config(tmp_path, data))
    assert config.surface_calibration.aspect_ratio == 1.5


def test_surface_calibration_wrong_corner_count_raises(tmp_path):
    data = dict(VALID)
    data["surface_calibration"] = {"corners": [[10, 10], [110, 10], [110, 60]]}
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, data))


def test_surface_calibration_unknown_field_raises(tmp_path):
    data = dict(VALID)
    data["surface_calibration"] = {
        "corners": [[10, 10], [110, 10], [110, 60], [10, 60]],
        "bogus": 1,
    }
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, data))


def _with_orientation(orientation: dict) -> dict:
    data = dict(VALID)
    data["objects"] = [dict(VALID["objects"][0], orientation=orientation), VALID["objects"][1]]
    return data


def test_load_config_with_orientation(tmp_path):
    data = _with_orientation({"count": 4, "method": "random", "angle_start_deg": 0, "angle_end_deg": 90, "seed": 5, "arrow_length": 25})
    config = load_config(write_config(tmp_path, data))
    orientation = config.objects[0].orientation
    assert orientation.count == 4
    assert orientation.method == "random"
    assert orientation.angle_start_deg == 0.0
    assert orientation.angle_end_deg == 90.0
    assert orientation.seed == 5
    assert orientation.arrow_length == 25.0


def test_orientation_defaults_when_only_count_given(tmp_path):
    data = _with_orientation({"count": 3})
    config = load_config(write_config(tmp_path, data))
    orientation = config.objects[0].orientation
    assert orientation.method == "uniform"
    assert orientation.angle_start_deg == 0.0
    assert orientation.angle_end_deg == 360.0
    assert orientation.seed == 0
    assert orientation.arrow_length == 40.0


def test_orientation_omitted_defaults_to_none(tmp_path):
    config = load_config(write_config(tmp_path, VALID))
    assert config.objects[0].orientation is None


def test_orientation_missing_count_raises(tmp_path):
    data = _with_orientation({"method": "uniform"})
    with pytest.raises(ConfigError, match="count"):
        load_config(write_config(tmp_path, data))


def test_orientation_zero_count_raises(tmp_path):
    data = _with_orientation({"count": 0})
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, data))


def test_orientation_invalid_method_raises(tmp_path):
    data = _with_orientation({"count": 2, "method": "spiral"})
    with pytest.raises(ConfigError, match="method"):
        load_config(write_config(tmp_path, data))


def test_orientation_non_positive_arrow_length_raises(tmp_path):
    data = _with_orientation({"count": 2, "arrow_length": 0})
    with pytest.raises(ConfigError, match="arrow_length"):
        load_config(write_config(tmp_path, data))


def test_orientation_unknown_field_raises(tmp_path):
    data = _with_orientation({"count": 2, "bogus": 1})
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, data))


def test_orientation_on_variable_false_object_raises(tmp_path):
    data = dict(VALID)
    data["objects"] = [
        VALID["objects"][0],
        dict(VALID["objects"][1], orientation={"count": 2}),
    ]
    with pytest.raises(ConfigError, match="orientation"):
        load_config(write_config(tmp_path, data))
