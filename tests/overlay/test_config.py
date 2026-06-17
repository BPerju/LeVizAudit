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
