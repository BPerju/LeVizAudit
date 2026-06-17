"""YAML config schema for the guided overlay: object list + pixel-space patterns.

No Rerun or dataset imports — this module only parses and validates a config file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

Point = tuple[float, float]

_ARC_REQUIRED = {"center", "radius", "angle_start_deg", "angle_end_deg"}
_ARC_ALLOWED = _ARC_REQUIRED | {"shape"}
_LINE_REQUIRED = {"start", "end"}
_LINE_ALLOWED = _LINE_REQUIRED | {"shape"}
_SECTOR_REQUIRED = {"center", "radius", "angle_start_deg", "angle_end_deg"}
_SECTOR_ALLOWED = _SECTOR_REQUIRED | {"shape", "inner_radius", "seed", "distribution", "border_width"}
_DISTRIBUTIONS = {"grid", "radial", "random"}
_EXCLUDE_ZONE_CIRCLE_REQUIRED = {"name", "center", "radius"}
_EXCLUDE_ZONE_CIRCLE_ALLOWED = _EXCLUDE_ZONE_CIRCLE_REQUIRED | {"shape"}
_EXCLUDE_ZONE_POLYGON_REQUIRED = {"name", "vertices"}
_EXCLUDE_ZONE_POLYGON_ALLOWED = _EXCLUDE_ZONE_POLYGON_REQUIRED | {"shape"}
_SURFACE_CALIBRATION_REQUIRED = {"corners"}
_SURFACE_CALIBRATION_ALLOWED = _SURFACE_CALIBRATION_REQUIRED | {"aspect_ratio"}


@dataclass(frozen=True)
class PatternConfig:
    shape: str  # "arc" | "line" | "sector"
    center: Point | None = None
    radius: float | None = None  # arc: radius; sector: OUTER radius
    angle_start_deg: float | None = None  # shared by arc and sector
    angle_end_deg: float | None = None  # shared by arc and sector
    start: Point | None = None
    end: Point | None = None
    inner_radius: float | None = None  # sector only; default 0.0 (full pie-slice)
    seed: int | None = None  # sector only; default 0
    distribution: str | None = None  # sector only; "grid" | "radial" | "random"; default "grid"
    border_width: float | None = None  # sector only; default 0.0 (no margin)


@dataclass(frozen=True)
class ExcludeZoneConfig:
    name: str
    shape: str = "circle"  # "circle" | "polygon"
    center: Point | None = None  # circle only
    radius: float | None = None  # circle only
    vertices: list[Point] | None = None  # polygon only


@dataclass(frozen=True)
class SurfaceCalibrationConfig:
    corners: list[Point]  # exactly 4, clockwise from top-left, pixel space
    aspect_ratio: float | None = None


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
    exclude_zones: list[ExcludeZoneConfig] = field(default_factory=list)
    surface_calibration: SurfaceCalibrationConfig | None = None


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
    if shape == "sector":
        _require_keys(data, _SECTOR_REQUIRED, _SECTOR_ALLOWED, f"{context} (shape: sector)")
        radius = data["radius"]
        if not isinstance(radius, (int, float)) or radius < 0:
            raise ConfigError(f"{context}: radius must be a non-negative number, got {radius!r}")
        inner_radius = data.get("inner_radius", 0.0)
        if not isinstance(inner_radius, (int, float)) or inner_radius < 0:
            raise ConfigError(
                f"{context}: inner_radius must be a non-negative number, got {inner_radius!r}"
            )
        if inner_radius >= radius:
            raise ConfigError(
                f"{context}: inner_radius ({inner_radius!r}) must be strictly less than radius ({radius!r})"
            )
        seed = data.get("seed", 0)
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ConfigError(f"{context}: seed must be an integer, got {seed!r}")
        distribution = data.get("distribution", "grid")
        if distribution not in _DISTRIBUTIONS:
            raise ConfigError(
                f"{context}: distribution must be one of {sorted(_DISTRIBUTIONS)}, got {distribution!r}"
            )
        border_width = data.get("border_width", 0.0)
        if not isinstance(border_width, (int, float)) or border_width < 0:
            raise ConfigError(
                f"{context}: border_width must be a non-negative number, got {border_width!r}"
            )
        return PatternConfig(
            shape="sector",
            center=_parse_point(data["center"], f"{context}.center"),
            radius=float(radius),
            angle_start_deg=float(data["angle_start_deg"]),
            angle_end_deg=float(data["angle_end_deg"]),
            inner_radius=float(inner_radius),
            seed=int(seed),
            distribution=distribution,
            border_width=float(border_width),
        )
    raise ConfigError(f"{context}: unknown pattern shape {shape!r} (allowed: 'arc', 'line', 'sector')")


def _parse_exclude_zone(data: dict, context: str) -> ExcludeZoneConfig:
    shape = data.get("shape", "circle")
    if shape == "circle":
        _require_keys(data, _EXCLUDE_ZONE_CIRCLE_REQUIRED, _EXCLUDE_ZONE_CIRCLE_ALLOWED, context)
        name = data["name"]
        if not isinstance(name, str) or not name:
            raise ConfigError(f"{context}: 'name' must be a non-empty string")
        radius = data["radius"]
        if not isinstance(radius, (int, float)) or radius <= 0:
            raise ConfigError(f"{context}: 'radius' must be a positive number, got {radius!r}")
        return ExcludeZoneConfig(
            name=name,
            shape="circle",
            center=_parse_point(data["center"], f"{context}.center"),
            radius=float(radius),
        )
    if shape == "polygon":
        _require_keys(data, _EXCLUDE_ZONE_POLYGON_REQUIRED, _EXCLUDE_ZONE_POLYGON_ALLOWED, context)
        name = data["name"]
        if not isinstance(name, str) or not name:
            raise ConfigError(f"{context}: 'name' must be a non-empty string")
        vertices = data["vertices"]
        if not isinstance(vertices, list) or len(vertices) < 3:
            raise ConfigError(f"{context}: 'vertices' must be a list of at least 3 [x, y] points")
        parsed_vertices = [_parse_point(v, f"{context}.vertices[{i}]") for i, v in enumerate(vertices)]
        return ExcludeZoneConfig(name=name, shape="polygon", vertices=parsed_vertices)
    raise ConfigError(f"{context}: unknown exclude_zone shape {shape!r} (allowed: 'circle', 'polygon')")


def _parse_surface_calibration(data: dict, context: str) -> SurfaceCalibrationConfig:
    _require_keys(data, _SURFACE_CALIBRATION_REQUIRED, _SURFACE_CALIBRATION_ALLOWED, context)
    corners = data["corners"]
    if not isinstance(corners, list) or len(corners) != 4:
        raise ConfigError(f"{context}: 'corners' must be a list of exactly 4 [x, y] points")
    parsed_corners = [_parse_point(c, f"{context}.corners[{i}]") for i, c in enumerate(corners)]
    aspect_ratio = data.get("aspect_ratio")
    if aspect_ratio is not None:
        if not isinstance(aspect_ratio, (int, float)) or aspect_ratio <= 0:
            raise ConfigError(
                f"{context}: 'aspect_ratio' must be a positive number, got {aspect_ratio!r}"
            )
        aspect_ratio = float(aspect_ratio)
    return SurfaceCalibrationConfig(corners=parsed_corners, aspect_ratio=aspect_ratio)


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

    raw_zones = raw.get("exclude_zones", [])
    if not isinstance(raw_zones, list):
        raise ConfigError(f"{path}: 'exclude_zones' must be a list")
    exclude_zones = [_parse_exclude_zone(z, f"exclude_zones[{i}]") for i, z in enumerate(raw_zones)]
    zone_names = [z.name for z in exclude_zones]
    if len(zone_names) != len(set(zone_names)):
        raise ConfigError(f"{path}: exclude_zones names must be unique, got {zone_names}")

    raw_calibration = raw.get("surface_calibration")
    surface_calibration = (
        _parse_surface_calibration(raw_calibration, "surface_calibration")
        if raw_calibration is not None
        else None
    )

    return OverlayConfig(
        camera_key=camera_key,
        objects=objects,
        marker=marker,
        exclude_zones=exclude_zones,
        surface_calibration=surface_calibration,
    )
