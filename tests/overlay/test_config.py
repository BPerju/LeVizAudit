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
