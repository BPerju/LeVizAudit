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
_SECTOR_ALLOWED = _SECTOR_REQUIRED | {
    "shape", "inner_radius", "seed", "distribution", "border_width", "count_mode",
}
_COUNT_MODES = {"fixed", "variable"}
_UNION_REQUIRED = {"circles"}
_UNION_ALLOWED = _UNION_REQUIRED | {"shape", "seed", "distribution", "border_width", "count_mode"}
_UNION_CIRCLE_REQUIRED = {"center", "radius"}
_DISTRIBUTIONS = {"grid", "radial", "random"}
_POINTS_REQUIRED = {"points"}
_POINTS_ALLOWED = _POINTS_REQUIRED | {"shape", "seed"}
_ORIENTATION_REQUIRED = {"count"}
_ORIENTATION_ALLOWED = _ORIENTATION_REQUIRED | {
    "method", "angle_start_deg", "angle_end_deg", "seed", "arrow_length", "initial_angle_deg",
}
_ROTATION_METHODS = {"uniform", "random"}
_EXCLUDE_ZONE_CIRCLE_REQUIRED = {"name", "center", "radius"}
_EXCLUDE_ZONE_CIRCLE_ALLOWED = _EXCLUDE_ZONE_CIRCLE_REQUIRED | {"shape"}
_EXCLUDE_ZONE_POLYGON_REQUIRED = {"name", "vertices"}
_EXCLUDE_ZONE_POLYGON_ALLOWED = _EXCLUDE_ZONE_POLYGON_REQUIRED | {"shape"}
_SURFACE_CALIBRATION_REQUIRED = {"corners"}
_SURFACE_CALIBRATION_ALLOWED = _SURFACE_CALIBRATION_REQUIRED | {"aspect_ratio"}
_SEQUENCING_MODES = {"lockstep", "shuffled"}
_LEVEL_STRATEGIES = {"fixed", "shuffle", "cycle", "balanced"}
_CO_LOCATION_MODES = {"stack", "keep_apart"}
_EPISODE_TARGETS_MODES = {"all", "one"}
_COMBINATION_MODES = {"systematic", "random", "coprime", "cartesian", "lcm"}
_COMBINATION_MODES_REQUIRING_COUNT = {"random", "coprime"}  # "systematic" tolerates an unset
                                                              # count (it just means inert/
                                                              # legacy); "cartesian"/"lcm"
                                                              # derive their own natural period


@dataclass(frozen=True)
class PatternConfig:
    shape: str  # "arc" | "line" | "sector" | "union"
    center: Point | None = None
    radius: float | None = None  # arc: radius; sector: OUTER radius
    angle_start_deg: float | None = None  # shared by arc and sector
    angle_end_deg: float | None = None  # shared by arc and sector
    start: Point | None = None
    end: Point | None = None
    inner_radius: float | None = None  # sector only; default 0.0 (full pie-slice)
    seed: int | None = None  # sector and union; default 0
    distribution: str | None = None  # sector and union; "grid" | "radial" | "random"; default
                                       # "grid" -- "radial" isn't yet implemented for union,
                                       # see generate_union_points
    border_width: float | None = None  # sector and union; default 0.0 (no margin)
    count_mode: str | None = None  # sector and union; "fixed" (default, exactly `count` points
                                     # via relocation) | "variable" (sector: grid/radial only;
                                     # union: grid only -- generates ideal positions and drops
                                     # invalid ones instead of relocating, so the final count
                                     # may be < `count`)
    circles: list[tuple[Point, float]] | None = None  # union only -- [(center, radius), ...],
                                                         # a bimanual (or any multi-region)
                                                         # setup's reach circles, sampled as
                                                         # their UNION so an overlap between two
                                                         # arms' circles isn't double-density --
                                                         # see generate_union_points
    points: list[Point] | None = None  # "points" only -- explicit, manually-authored positions
                                         # (the calibration tool's per-object point assignment),
                                         # returned as-is by build_pattern with no sampling


@dataclass(frozen=True)
class OrientationConfig:
    count: int  # how many distinct target rotation angles to generate -- cycled per-episode
                # independently of the position pattern's own count, via target_for_episode
    method: str = "uniform"  # "uniform" | "random" -- see generate_rotation_angles
    angle_start_deg: float = 0.0
    angle_end_deg: float = 360.0
    seed: int = 0  # "random" method only
    arrow_length: float = 40.0  # same space as the object's pattern (pixel, or canonical
                                  # when surface_calibration is set) -- length of the
                                  # orientation guide arrow drawn from the target point
    initial_angle_deg: float = 0.0  # "uniform" method only -- the first arrow's angle


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
class MarkerConfig:
    radius_px: float = 10.0
    color_rgba: tuple[int, int, int, int] = (255, 64, 64, 255)
    label: bool = True


@dataclass(frozen=True)
class ObjectConfig:
    name: str
    count: int
    variable: bool
    pattern: PatternConfig | None
    marker: MarkerConfig  # always resolved (falls back to the top-level `marker:` if this
                           # object has no override) -- consumers never need an `obj.marker or
                           # config.marker` fallback dance. Distinct per-object colors are the
                           # main thing that makes a bimanual setup's two simultaneous markers
                           # visually tell-apart-able; see the `marker:` bullet in CLAUDE.md.
    orientation: OrientationConfig | None = None  # opt-in only (default: no orientation arrow
                                                    # at all, the original position-only
                                                    # behavior) -- only meaningful/allowed when
                                                    # variable: true, same as pattern
    sequencing: str = "lockstep"  # how this object cycles through its own pattern's points --
                                    # "lockstep" (default, today's exact i % len behavior) |
                                    # "shuffled" -- see pattern.sequence_index. Decorrelates two
                                    # objects of equal-length patterns from always pairing the
                                    # same way. ("stratified" and "phase", two earlier modes,
                                    # were both removed before this ever shipped in a release --
                                    # see CLAUDE.md. "phase" in particular conflated object-pair
                                    # decorrelation with the separate, scene-level question of
                                    # whether co-located objects should ever coincide at all --
                                    # that's now OverlayConfig.co_location, below. `jitter_px`
                                    # and `presence`, two other earlier per-object knobs, were
                                    # also removed -- see CLAUDE.md.)


@dataclass(frozen=True)
class OverlayConfig:
    camera_key: str
    objects: list[ObjectConfig]
    marker: MarkerConfig
    exclude_zones: list[ExcludeZoneConfig] = field(default_factory=list)
    surface_calibration: SurfaceCalibrationConfig | None = None
    # SCENE-LEVEL (not per-object): whether 2+ objects sharing a point this episode are a real
    # STACK (today's default/original behavior -- z-order via level_strategy/level_seed,
    # see pattern.level_order) or should instead be kept apart -- each nudged, where possible,
    # onto one of its OWN other assigned points instead of co-occupying (pattern.
    # resolve_keep_apart). Replaces the removed per-StackConfig authored-site design: stacking
    # is now read directly off which objects' `pattern: {shape: points}` lists happen to share
    # a coordinate, not a separately-authored block -- see CLAUDE.md.
    co_location: str = "stack"  # "stack" | "keep_apart" -- only meaningful under
                                  # episode_targets == "all"; under "one", co-location is a
                                  # static, guaranteed fact derived from occupied_sites (see
                                  # below), not something that can or can't coincide per
                                  # episode, so this setting has no effect there
    level_strategy: str = "fixed"  # "fixed" | "shuffle" | "cycle" | "balanced" -- consulted
                                     # whenever 2+ objects share a site this episode: under
                                     # "all", only when co_location == "stack" and they happen
                                     # to coincide; under "one", always, for every site with 2+
                                     # objects (every such site is a guaranteed stack)
    level_seed: int = 0
    # SCENE-LEVEL: how many SITES are visible at once each episode. A site is one occupied
    # point -- either a single object, or several objects that share that exact coordinate
    # (see pattern.occupied_sites for "one", or the per-episode coincidence grouping in
    # session.show_targets for "all"). "all" (default, today's exact behavior): every site is
    # shown simultaneously, every episode -- decoupled from any one object's own pattern length
    # via combination_count below. "one": exactly ONE site is shown per episode, round-robin in
    # declaration order over pattern.occupied_sites (the STATIC union of every object's own
    # assigned points) -- every object outside the active site is cleared. #episodes under
    # "one" therefore equals the number of distinct occupied points, not the longest object's
    # pattern length; an object with several assigned points gets one dedicated turn per point,
    # not a sweep. See CLAUDE.md for the reports that shaped this (stacks were rare/partial
    # under an earlier dynamic-coincidence design, and the JS preview separately
    # over-multiplied its episode count).
    episode_targets: str = "all"  # "all" | "one"
    # SCENE-LEVEL, "all" mode only: how many distinct simultaneous-combination episodes to
    # cycle through, via pattern.resolve_combination_indices, instead of each object
    # independently lockstep/shuffled-sweeping its own pattern (sequence_index) -- decouples
    # the visible combination count from any one object's own pattern length (reported
    # directly: "if i have 3 objects and 20 steps the combinations are limited to the nr of
    # steps"). `None` (default, with combination_mode also left at its default "systematic")
    # keeps today's exact lockstep/shuffled behavior, byte-for-byte. Ignored under
    # episode_targets == "one", which has no notion of "combination" at all.
    combination_count: int | None = None
    # SCENE-LEVEL, "all" mode only: which combination-generation strategy combination_count
    # drives -- see pattern.resolve_combination_indices/combination_period for the full
    # rationale of each. "systematic" (default): a deterministic, evenly-spread mapping,
    # capped per object at that object's own pattern length (see the note on combination_index
    # about why ANY purely-deterministic per-object function is capped there -- escaping that
    # cap needs "cartesian" or "random" below). "random": independent seeded draws per object
    # (combination_seed) -- the joint combination space is up to the PRODUCT of every object's
    # length, not capped at any single one. "coprime": a deterministic, no-RNG alternative to
    # "systematic" (a fixed coprime stride per object) -- same per-object cap as "systematic".
    # "cartesian": the full Cartesian product of every object's own points, enumerated
    # deterministically -- escapes the per-object cap exactly like "random" does, with no
    # randomness; can get very large (combination_count, if set, caps it). "lcm": plain
    # lockstep cycling, but run for exactly lcm(every object's own length) episodes instead of
    # just the longest one, so every pairing lockstep naturally produces appears once.
    # "cartesian"/"lcm" derive their own natural period when combination_count is left unset
    # (the product, or the lcm, of every object's length) -- "systematic"/"random"/"coprime"
    # have no such natural period and REQUIRE combination_count to be set.
    combination_mode: str = "systematic"  # "systematic" | "random" | "coprime" | "cartesian" | "lcm"
    combination_seed: int = 0  # "random" mode only


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
        count_mode = data.get("count_mode", "fixed")
        if count_mode not in _COUNT_MODES:
            raise ConfigError(f"{context}: count_mode must be one of {sorted(_COUNT_MODES)}, got {count_mode!r}")
        if count_mode == "variable" and distribution == "random":
            raise ConfigError(f"{context}: count_mode='variable' only supports distribution 'grid' or 'radial'")
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
            count_mode=count_mode,
        )
    if shape == "union":
        _require_keys(data, _UNION_REQUIRED, _UNION_ALLOWED, f"{context} (shape: union)")
        raw_circles = data["circles"]
        if not isinstance(raw_circles, list) or len(raw_circles) < 1:
            raise ConfigError(f"{context}: 'circles' must be a non-empty list")
        circles = [_parse_union_circle(c, f"{context}.circles[{i}]") for i, c in enumerate(raw_circles)]
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
        count_mode = data.get("count_mode", "fixed")
        if count_mode not in _COUNT_MODES:
            raise ConfigError(f"{context}: count_mode must be one of {sorted(_COUNT_MODES)}, got {count_mode!r}")
        if count_mode == "variable" and distribution != "grid":
            raise ConfigError(f"{context}: count_mode='variable' only supports distribution 'grid' for a union pattern")
        return PatternConfig(
            shape="union",
            circles=circles,
            seed=int(seed),
            distribution=distribution,
            border_width=float(border_width),
            count_mode=count_mode,
        )
    if shape == "points":
        _require_keys(data, _POINTS_REQUIRED, _POINTS_ALLOWED, f"{context} (shape: points)")
        raw_points = data["points"]
        if not isinstance(raw_points, list) or not raw_points:
            raise ConfigError(f"{context}: 'points' must be a non-empty list")
        points = [_parse_point(p, f"{context}.points[{i}]") for i, p in enumerate(raw_points)]
        # Not used for sampling (points are explicit, not generated) -- only as the seed
        # `sequence_index` draws from for this object's "shuffled" sequencing mode (see
        # ObjectConfig.sequencing).
        seed = data.get("seed", 0)
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ConfigError(f"{context}: seed must be an integer, got {seed!r}")
        return PatternConfig(shape="points", points=points, seed=int(seed))
    raise ConfigError(
        f"{context}: unknown pattern shape {shape!r} "
        "(allowed: 'arc', 'line', 'sector', 'union', 'points')"
    )


def _parse_union_circle(data: dict, context: str) -> tuple[Point, float]:
    _require_keys(data, _UNION_CIRCLE_REQUIRED, _UNION_CIRCLE_REQUIRED, context)
    radius = data["radius"]
    if not isinstance(radius, (int, float)) or radius <= 0:
        raise ConfigError(f"{context}: 'radius' must be a positive number, got {radius!r}")
    return (_parse_point(data["center"], f"{context}.center"), float(radius))


def _parse_orientation(data: dict, context: str) -> OrientationConfig:
    _require_keys(data, _ORIENTATION_REQUIRED, _ORIENTATION_ALLOWED, context)
    count = data["count"]
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ConfigError(f"{context}: 'count' must be an integer >= 1, got {count!r}")
    method = data.get("method", "uniform")
    if method not in _ROTATION_METHODS:
        raise ConfigError(
            f"{context}: method must be one of {sorted(_ROTATION_METHODS)}, got {method!r}"
        )
    angle_start_deg = data.get("angle_start_deg", 0.0)
    angle_end_deg = data.get("angle_end_deg", 360.0)
    if not isinstance(angle_start_deg, (int, float)) or not isinstance(angle_end_deg, (int, float)):
        raise ConfigError(f"{context}: angle_start_deg/angle_end_deg must be numbers")
    seed = data.get("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ConfigError(f"{context}: seed must be an integer, got {seed!r}")
    arrow_length = data.get("arrow_length", 40.0)
    if not isinstance(arrow_length, (int, float)) or arrow_length <= 0:
        raise ConfigError(f"{context}: arrow_length must be a positive number, got {arrow_length!r}")
    initial_angle_deg = data.get("initial_angle_deg", 0.0)
    if not isinstance(initial_angle_deg, (int, float)):
        raise ConfigError(f"{context}: initial_angle_deg must be a number, got {initial_angle_deg!r}")
    return OrientationConfig(
        count=count,
        method=method,
        angle_start_deg=float(angle_start_deg),
        angle_end_deg=float(angle_end_deg),
        seed=int(seed),
        arrow_length=float(arrow_length),
        initial_angle_deg=float(initial_angle_deg),
    )


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


def _parse_object(data: dict, context: str, default_marker: MarkerConfig) -> ObjectConfig:
    _require_keys(
        data,
        {"name", "count", "variable"},
        {
            "name", "count", "variable", "pattern", "marker", "orientation", "sequencing",
        },
        context,
    )
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
    # An object's `marker:` only needs to specify whatever it wants to DIFFER from the
    # top-level default (typically just color_rgba, for a bimanual setup's two simultaneous
    # markers) -- any field it omits falls back to default_marker's, not MarkerConfig()'s
    # hardcoded ones, so e.g. radius_px/label stay governed by the one top-level `marker:`
    # block unless an object explicitly overrides them too.
    marker = _parse_marker(data.get("marker"), default_marker, f"{context}.marker")
    raw_orientation = data.get("orientation")
    if variable:
        orientation = (
            _parse_orientation(raw_orientation, f"{context}.orientation") if raw_orientation else None
        )
    else:
        if raw_orientation:
            raise ConfigError(f"{context}: 'orientation' must be omitted/null when variable: false")
        orientation = None
    sequencing = data.get("sequencing", "lockstep")
    if not variable and "sequencing" in data:
        raise ConfigError(f"{context}: 'sequencing' must be omitted when variable: false")
    if sequencing not in _SEQUENCING_MODES:
        raise ConfigError(f"{context}: sequencing must be one of {sorted(_SEQUENCING_MODES)}, got {sequencing!r}")
    return ObjectConfig(
        name=name, count=count, variable=variable, pattern=pattern, marker=marker, orientation=orientation,
        sequencing=sequencing,
    )


def _parse_marker(
    data: dict | None, fallback: MarkerConfig = MarkerConfig(), context: str = "marker"
) -> MarkerConfig:
    if data is None:
        return fallback
    allowed = {"radius_px", "color_rgba", "label"}
    unknown = set(data.keys()) - allowed
    if unknown:
        raise ConfigError(f"{context}: unknown field(s) {sorted(unknown)} (allowed: {sorted(allowed)})")
    radius_px = data.get("radius_px", fallback.radius_px)
    color_rgba = data.get("color_rgba", fallback.color_rgba)
    label = data.get("label", fallback.label)
    if len(color_rgba) != 4:
        raise ConfigError(f"{context}.color_rgba: expected 4 values [r, g, b, a], got {color_rgba!r}")
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

    # Parsed before objects -- each object's own `marker:` (if any) falls back to this one.
    marker = _parse_marker(raw.get("marker"))

    raw_objects = raw.get("objects")
    if not isinstance(raw_objects, list) or not raw_objects:
        raise ConfigError(f"{path}: 'objects' must be a non-empty list")

    objects = [_parse_object(o, f"objects[{i}]", marker) for i, o in enumerate(raw_objects)]
    names = [o.name for o in objects]
    if len(names) != len(set(names)):
        raise ConfigError(f"{path}: object names must be unique, got {names}")

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

    co_location = raw.get("co_location", "stack")
    if co_location not in _CO_LOCATION_MODES:
        raise ConfigError(f"{path}: co_location must be one of {sorted(_CO_LOCATION_MODES)}, got {co_location!r}")
    level_strategy = raw.get("level_strategy", "fixed")
    if level_strategy not in _LEVEL_STRATEGIES:
        raise ConfigError(f"{path}: level_strategy must be one of {sorted(_LEVEL_STRATEGIES)}, got {level_strategy!r}")
    level_seed = raw.get("level_seed", 0)
    if not isinstance(level_seed, int) or isinstance(level_seed, bool):
        raise ConfigError(f"{path}: level_seed must be an integer, got {level_seed!r}")

    episode_targets = raw.get("episode_targets", "all")
    if episode_targets not in _EPISODE_TARGETS_MODES:
        raise ConfigError(
            f"{path}: episode_targets must be one of {sorted(_EPISODE_TARGETS_MODES)}, got {episode_targets!r}"
        )

    combination_count = raw.get("combination_count")
    if combination_count is not None:
        if not isinstance(combination_count, int) or isinstance(combination_count, bool) or combination_count < 1:
            raise ConfigError(
                f"{path}: combination_count must be a positive integer, got {combination_count!r}"
            )

    combination_mode = raw.get("combination_mode", "systematic")
    if combination_mode not in _COMBINATION_MODES:
        raise ConfigError(
            f"{path}: combination_mode must be one of {sorted(_COMBINATION_MODES)}, got {combination_mode!r}"
        )
    if combination_mode in _COMBINATION_MODES_REQUIRING_COUNT and combination_count is None:
        raise ConfigError(
            f"{path}: combination_mode={combination_mode!r} requires combination_count to be set"
        )

    combination_seed = raw.get("combination_seed", 0)
    if not isinstance(combination_seed, int) or isinstance(combination_seed, bool):
        raise ConfigError(f"{path}: combination_seed must be an integer, got {combination_seed!r}")

    return OverlayConfig(
        camera_key=camera_key,
        objects=objects,
        marker=marker,
        exclude_zones=exclude_zones,
        surface_calibration=surface_calibration,
        co_location=co_location,
        level_strategy=level_strategy,
        level_seed=int(level_seed),
        episode_targets=episode_targets,
        combination_count=combination_count,
        combination_mode=combination_mode,
        combination_seed=int(combination_seed),
    )
