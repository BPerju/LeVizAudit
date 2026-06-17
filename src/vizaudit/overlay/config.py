"""YAML config schema for the guided overlay: object list + pixel-space patterns.

No Rerun or dataset imports — this module only parses and validates a config file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

Point = tuple[float, float]

_ARC_REQUIRED = {"center", "radius", "angle_start_deg", "angle_end_deg"}
_ARC_ALLOWED = _ARC_REQUIRED | {"shape"}
_LINE_REQUIRED = {"start", "end"}
_LINE_ALLOWED = _LINE_REQUIRED | {"shape"}


@dataclass(frozen=True)
class PatternConfig:
    shape: str  # "arc" | "line"
    center: Point | None = None
    radius: float | None = None
    angle_start_deg: float | None = None
    angle_end_deg: float | None = None
    start: Point | None = None
    end: Point | None = None


@dataclass(frozen=True)
class ObjectConfig:
    name: str
    count: int
    variable: bool
    pattern: PatternConfig | None


@dataclass(frozen=True)
class MarkerConfig:
    radius_px: float = 10.0
    color_rgba: tuple[int, int, int, int] = (255, 64, 64, 255)
    label: bool = True


@dataclass(frozen=True)
class OverlayConfig:
    camera_key: str
    objects: list[ObjectConfig]
    marker: MarkerConfig


class ConfigError(ValueError):
    pass


def _require_keys(data: dict, required: set[str], allowed: set[str], context: str) -> None:
    keys = set(data.keys())
    missing = required - keys
    if missing:
        raise ConfigError(f"{context}: missing required field(s) {sorted(missing)}")
    unknown = keys - allowed
    if unknown:
        raise ConfigError(f"{context}: unknown field(s) {sorted(unknown)} (allowed: {sorted(allowed)})")


def _parse_point(value: object, context: str) -> Point:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigError(f"{context}: expected a 2-element [x, y] pixel coordinate, got {value!r}")
    x, y = value
    if not all(isinstance(v, (int, float)) for v in (x, y)):
        raise ConfigError(f"{context}: pixel coordinates must be numbers, got {value!r}")
    if x < 0 or y < 0:
        raise ConfigError(f"{context}: pixel coordinates must be non-negative, got {value!r}")
    return (float(x), float(y))


def _parse_pattern(data: dict, context: str) -> PatternConfig:
    if "shape" not in data:
        raise ConfigError(f"{context}: pattern is missing required field 'shape'")
    shape = data["shape"]
    if shape == "arc":
        _require_keys(data, _ARC_REQUIRED, _ARC_ALLOWED, f"{context} (shape: arc)")
        radius = data["radius"]
        if not isinstance(radius, (int, float)) or radius < 0:
            raise ConfigError(f"{context}: radius must be a non-negative number, got {radius!r}")
        return PatternConfig(
            shape="arc",
            center=_parse_point(data["center"], f"{context}.center"),
            radius=float(radius),
            angle_start_deg=float(data["angle_start_deg"]),
            angle_end_deg=float(data["angle_end_deg"]),
        )
    if shape == "line":
        _require_keys(data, _LINE_REQUIRED, _LINE_ALLOWED, f"{context} (shape: line)")
        return PatternConfig(
            shape="line",
            start=_parse_point(data["start"], f"{context}.start"),
            end=_parse_point(data["end"], f"{context}.end"),
        )
    raise ConfigError(f"{context}: unknown pattern shape {shape!r} (allowed: 'arc', 'line')")


def _parse_object(data: dict, context: str) -> ObjectConfig:
    _require_keys(data, {"name", "count", "variable"}, {"name", "count", "variable", "pattern"}, context)
    name = data["name"]
    if not isinstance(name, str) or not name:
        raise ConfigError(f"{context}: 'name' must be a non-empty string")
    count = data["count"]
    if not isinstance(count, int) or count < 1:
        raise ConfigError(f"{context}: 'count' must be an integer >= 1, got {count!r}")
    variable = data["variable"]
    if not isinstance(variable, bool):
        raise ConfigError(f"{context}: 'variable' must be a boolean, got {variable!r}")
    raw_pattern = data.get("pattern")
    if variable:
        if not raw_pattern:
            raise ConfigError(f"{context}: 'pattern' is required when variable: true")
        pattern = _parse_pattern(raw_pattern, f"{context}.pattern")
    else:
        if raw_pattern:
            raise ConfigError(f"{context}: 'pattern' must be omitted/null when variable: false")
        pattern = None
    return ObjectConfig(name=name, count=count, variable=variable, pattern=pattern)


def _parse_marker(data: dict | None) -> MarkerConfig:
    if data is None:
        return MarkerConfig()
    allowed = {"radius_px", "color_rgba", "label"}
    unknown = set(data.keys()) - allowed
    if unknown:
        raise ConfigError(f"marker: unknown field(s) {sorted(unknown)} (allowed: {sorted(allowed)})")
    marker = MarkerConfig()
    radius_px = data.get("radius_px", marker.radius_px)
    color_rgba = data.get("color_rgba", marker.color_rgba)
    label = data.get("label", marker.label)
    if len(color_rgba) != 4:
        raise ConfigError(f"marker.color_rgba: expected 4 values [r, g, b, a], got {color_rgba!r}")
    return MarkerConfig(radius_px=float(radius_px), color_rgba=tuple(color_rgba), label=bool(label))


def load_config(path: str | Path) -> OverlayConfig:
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top-level YAML must be a mapping")

    camera_key = raw.get("camera_key")
    if not isinstance(camera_key, str) or not camera_key:
        raise ConfigError(f"{path}: 'camera_key' must be a non-empty string")

    raw_objects = raw.get("objects")
    if not isinstance(raw_objects, list) or not raw_objects:
        raise ConfigError(f"{path}: 'objects' must be a non-empty list")

    objects = [_parse_object(o, f"objects[{i}]") for i, o in enumerate(raw_objects)]
    names = [o.name for o in objects]
    if len(names) != len(set(names)):
        raise ConfigError(f"{path}: object names must be unique, got {names}")

    marker = _parse_marker(raw.get("marker"))
    return OverlayConfig(camera_key=camera_key, objects=objects, marker=marker)
