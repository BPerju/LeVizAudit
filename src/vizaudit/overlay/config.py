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
_POINTS_ALLOWED = _POINTS_REQUIRED | {"shape"}
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

# ===== Scene-level model =====
#
# What Rerun actually shows each episode is just a set of {point, object, color, level} -- so
# the whole scene only needs to answer two questions: how many targets per episode
# (`per_episode`), and how the distinct episodes are enumerated (`combinations`/`order`, only
# meaningful when `per_episode == "all"`) -- plus whether 2+ objects sharing a point this
# episode pile up or get nudged apart (`stacking`), and how their z-order varies (`level`).
# This REPLACES five previously co-equal, implementation-named axes (position_mode,
# site_selection+site_count+site_order+site_seed, combination_mode+combination_count+
# combination_seed, co_location, level_strategy+level_seed) plus the deprecated
# `episode_targets` alias -- see CLAUDE.md for the full mathematical rationale and the old-to-
# new mapping. This project is pre-release, so every removed key is a hard ConfigError naming
# its replacement, not a silently-accepted alias.
_COMBINATIONS_KEYWORDS = {"synced", "shuffled", "all"}
_ORDER_VALUES = {"even", "random", "coprime"}
_STACKING_VALUES = {"stack", "keep_apart"}
_LEVEL_VALUES = {"fixed", "cycle", "shuffle", "balanced"}

# Presets are pure sugar: they set the DEFAULT value of each axis below, before any explicit
# axis in the YAML overrides it ("advanced follows the preset but can be modified") -- nothing
# downstream of `load_config` ever sees `preset` itself, only the resolved axes.
_PRESETS: dict[str, dict[str, object]] = {
    # Today's original default behavior: every object advances every episode, in lockstep,
    # objects sharing a point pile into a stack.
    "sweep": dict(per_episode="all", combinations="synced", order="even", stacking="stack", level="fixed"),
    # Same, but each object's own visit order is independently seeded-random instead of
    # lockstep -- decorrelates two equal-length objects from always pairing the same way.
    "shuffled_sweep": dict(per_episode="all", combinations="shuffled", order="even", stacking="stack", level="fixed"),
    # The common "place N objects one at a time" case: exactly one stack/episode, cycling
    # through every distinct point any object is assigned to.
    "one_at_a_time": dict(per_episode=1, combinations="synced", order="even", stacking="stack", level="fixed"),
    # Cycle through every individual (object, point) pair one at a time -- never grouping
    # coincidentally-shared points into a stack.
    "cycle_through_points": dict(per_episode=1, combinations="synced", order="even", stacking="keep_apart", level="fixed"),
    # Every fixed point any object is assigned to, shown simultaneously, forever -- no
    # rotation at all (a persistent coverage/reference view rather than a guided one-at-a-time
    # directive).
    "show_everything": dict(per_episode="static", combinations="synced", order="even", stacking="stack", level="fixed"),
    # Every simultaneous combination of every object's own points, shown all at once, the full
    # Cartesian product (deterministic, no count needed).
    "every_combination": dict(per_episode="all", combinations="all", order="even", stacking="stack", level="fixed"),
    # A sampled subset of the full combination space -- override `combinations:` to taste.
    "sample_combinations": dict(per_episode="all", combinations=100, order="even", stacking="stack", level="fixed"),
    # No preset-driven defaults beyond the same baseline `sweep` already uses -- exists so the
    # calibration tool can show "Custom" once the operator deviates from every named preset.
    "custom": dict(per_episode="all", combinations="synced", order="even", stacking="stack", level="fixed"),
}

# Every removed scene-level key, mapped to the new field(s) that replaced it -- checked first in
# load_config so a stale config gets one clear, actionable error instead of being silently
# misinterpreted or hitting an unrelated downstream error.
_REMOVED_SCENE_KEYS = {
    "position_mode": "per_episode/combinations/stacking",
    "site_selection": "per_episode",
    "site_count": "per_episode",
    "site_order": "order",
    "site_seed": "seed",
    "combination_mode": "combinations/order",
    "combination_count": "combinations",
    "combination_seed": "seed",
    "co_location": "stacking",
    "level_strategy": "level",
    "level_seed": "seed",
    "episode_targets": "per_episode",
}


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
    pattern: PatternConfig  # always required -- the old `variable: bool` flag was dropped
                              # (every object that exists is something the scene places; a
                              # non-placed "object" did nothing in the engine at all, since
                              # session.py filtered such objects out and never touched them
                              # again -- pure dead config). See CLAUDE.md.
    marker: MarkerConfig  # always resolved (falls back to the top-level `marker:` if this
                           # object has no override) -- consumers never need an `obj.marker or
                           # config.marker` fallback dance. Distinct per-object colors are the
                           # main thing that makes a bimanual setup's two simultaneous markers
                           # visually tell-apart-able; see the `marker:` bullet in CLAUDE.md.
    orientation: OrientationConfig | None = None  # opt-in only (default: no orientation arrow
                                                    # at all, the original position-only
                                                    # behavior)
    # `sequencing` (per-object lockstep/shuffled) was REMOVED and folded into the scene-level
    # `OverlayConfig.combinations`/`order` -- the two were always the same underlying question
    # (which of this thing's own assigned points to show) asked at two different scopes, so
    # keeping both was redundant. Losing per-object order granularity is a deliberate, accepted
    # tradeoff -- see CLAUDE.md. A `sequencing:` key on an object is now a hard ConfigError
    # naming the migration (this project is pre-release, so a clean break is acceptable).


@dataclass(frozen=True)
class OverlayConfig:
    camera_key: str
    objects: list[ObjectConfig]
    marker: MarkerConfig
    exclude_zones: list[ExcludeZoneConfig] = field(default_factory=list)
    surface_calibration: SurfaceCalibrationConfig | None = None
    # ===== Scene-level model -- see CLAUDE.md for the full math/rationale =====
    # Together these answer the only two things a multi-object scene needs to specify: how many
    # targets per episode (`per_episode`), and how the distinct episodes are enumerated
    # (`combinations`/`order`). `preset` is parse-time-only sugar (config.py expands it into
    # defaults for the fields below before explicit values override it) and isn't stored here.
    #
    # "all" (default): every object's own position advances every episode -- DYNAMIC sweep.
    # Objects sharing an IDENTICAL assigned point-list are grouped into one atomic stack unit
    # (pattern.group_into_units) so a stack is always shown full, never sliced. A scene with no
    # shared point-lists degenerates to one singleton unit per object, in declaration order --
    # byte-identical to the original per-object sweep.
    # "static": STATIC enumeration, no sweep at all -- every point any object is assigned to is
    # a fixed, permanent site, and EVERY one of them is shown every episode (no rotation) -- a
    # persistent coverage/reference view rather than a guided one-at-a-time directive. Useful
    # when the site count isn't known/stable up front (depends on exclude_zones/object count).
    # <positive int> (e.g. `1`): like "static", but exactly this many of those fixed sites are
    # shown per episode (pattern.select_sites, ordered by `order`). `per_episode: 1` is the
    # common "place one object/stack at a time" case.
    per_episode: int | str = "all"
    # Only meaningful when per_episode == "all" -- how the distinct (DYNAMIC) episodes are
    # enumerated. "synced" (default): plain `episode_index % length` per unit (today's exact
    # original behavior). "shuffled": each unit visits its own points in its own seeded-random
    # order. "all": every simultaneous combination of every unit's own points (the full
    # Cartesian product, deterministic). <positive int> N: sample exactly N combinations,
    # spread by `order`.
    combinations: str | int = "synced"
    # "even" (default): a deterministic, evenly-spread choice -- used both for sampling
    # `combinations: <int>` and for which sites a static `per_episode: <int>` shows each
    # episode. "random": a seeded random choice instead (uses `seed`). "coprime": a
    # deterministic, no-RNG alternative to "even" for sampling `combinations: <int>` only (not
    # valid for a static per_episode selection).
    order: str = "even"
    # Whether 2+ objects sharing a point this episode pile into a real stack (z-order via
    # `level`/`seed`, pattern.level_order) or get nudged apart instead -- under per_episode:
    # "all", onto one of their OWN other assigned points where possible (pattern.
    # resolve_keep_apart); under a static per_episode: <int>, by never merging coincidentally-
    # shared points into one site in the first place (pattern._exploded_sites), so the rotation
    # cycles through every individual point instead of every stack.
    stacking: str = "stack"  # "stack" | "keep_apart"
    # How a stack's z-order (which member renders on top) varies across episodes when 2+
    # objects share a site -- see pattern.level_order. "fixed" (default): declaration/
    # assignment order, never changes.
    level: str = "fixed"  # "fixed" | "shuffle" | "cycle" | "balanced"
    # One shared seed for every randomized aspect above (combinations: "shuffled"/sampling
    # "random"; order: "random"; level: "shuffle"/"balanced") -- internally salted per aspect
    # (pattern._SEED_SALT_*) so they don't silently correlate just because they share one
    # user-facing number.
    seed: int = 0


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
    if not isinstance(data, dict):
        raise ConfigError(f"{context}: 'pattern' must be a mapping, got {data!r}")
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
        return PatternConfig(shape="points", points=points)
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
    if "sequencing" in data:
        raise ConfigError(
            f"{context}: 'sequencing' was removed -- per-object visit order is now scene-level "
            "via 'combinations'/'order' (synced/shuffled/all/<int>, even/random/coprime), "
            "see CLAUDE.md"
        )
    if "variable" in data:
        raise ConfigError(
            f"{context}: 'variable' was removed -- every object now always has a 'pattern' "
            "(a non-placed object did nothing in the engine at all, see CLAUDE.md); just "
            "remove the 'variable' key"
        )
    _require_keys(
        data,
        {"name", "count", "pattern"},
        {"name", "count", "pattern", "marker", "orientation"},
        context,
    )
    name = data["name"]
    if not isinstance(name, str) or not name:
        raise ConfigError(f"{context}: 'name' must be a non-empty string")
    count = data["count"]
    if not isinstance(count, int) or count < 1:
        raise ConfigError(f"{context}: 'count' must be an integer >= 1, got {count!r}")
    pattern = _parse_pattern(data["pattern"], f"{context}.pattern")
    # An object's `marker:` only needs to specify whatever it wants to DIFFER from the
    # top-level default (typically just color_rgba, for a bimanual setup's two simultaneous
    # markers) -- any field it omits falls back to default_marker's, not MarkerConfig()'s
    # hardcoded ones, so e.g. radius_px/label stay governed by the one top-level `marker:`
    # block unless an object explicitly overrides them too.
    marker = _parse_marker(data.get("marker"), default_marker, f"{context}.marker")
    raw_orientation = data.get("orientation")
    orientation = (
        _parse_orientation(raw_orientation, f"{context}.orientation") if raw_orientation else None
    )
    return ObjectConfig(name=name, count=count, pattern=pattern, marker=marker, orientation=orientation)


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

    # Every removed scene-level key gets one clear, actionable error naming its replacement --
    # checked before any new-field parsing, so a stale config never gets silently
    # misinterpreted. See CLAUDE.md for the full old-to-new mapping.
    for old_key, new_key in _REMOVED_SCENE_KEYS.items():
        if old_key in raw:
            raise ConfigError(
                f"{path}: '{old_key}' was replaced by '{new_key}' -- see CLAUDE.md for the new "
                "per_episode/combinations/order/stacking/level/seed scene model"
            )

    preset = raw.get("preset", "custom")
    if preset not in _PRESETS:
        raise ConfigError(f"{path}: preset must be one of {sorted(_PRESETS)}, got {preset!r}")
    defaults = _PRESETS[preset]

    per_episode = raw.get("per_episode", defaults["per_episode"])
    if per_episode not in ("all", "static"):
        if not isinstance(per_episode, int) or isinstance(per_episode, bool) or per_episode < 1:
            raise ConfigError(
                f"{path}: per_episode must be 'all', 'static', or a positive integer, "
                f"got {per_episode!r}"
            )

    combinations = raw.get("combinations", defaults["combinations"])
    if isinstance(combinations, bool) or not (
        combinations in _COMBINATIONS_KEYWORDS
        or (isinstance(combinations, int) and combinations >= 1)
    ):
        raise ConfigError(
            f"{path}: combinations must be one of {sorted(_COMBINATIONS_KEYWORDS)} or a "
            f"positive integer, got {combinations!r}"
        )
    if per_episode != "all" and combinations != "synced":
        raise ConfigError(
            f"{path}: 'combinations' only applies when per_episode='all' (got "
            f"per_episode={per_episode!r}) -- remove 'combinations' or set per_episode: all"
        )

    order = raw.get("order", defaults["order"])
    if order not in _ORDER_VALUES:
        raise ConfigError(f"{path}: order must be one of {sorted(_ORDER_VALUES)}, got {order!r}")
    if order == "coprime" and not isinstance(combinations, int):
        raise ConfigError(
            f"{path}: order='coprime' requires 'combinations' to be a positive integer "
            "(it's a sampling strategy for combinations: <int>, not a site order)"
        )

    stacking = raw.get("stacking", defaults["stacking"])
    if stacking not in _STACKING_VALUES:
        raise ConfigError(f"{path}: stacking must be one of {sorted(_STACKING_VALUES)}, got {stacking!r}")

    level = raw.get("level", defaults["level"])
    if level not in _LEVEL_VALUES:
        raise ConfigError(f"{path}: level must be one of {sorted(_LEVEL_VALUES)}, got {level!r}")

    seed = raw.get("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ConfigError(f"{path}: seed must be an integer, got {seed!r}")

    return OverlayConfig(
        camera_key=camera_key,
        objects=objects,
        marker=marker,
        exclude_zones=exclude_zones,
        surface_calibration=surface_calibration,
        per_episode=per_episode,
        combinations=combinations,
        order=order,
        stacking=stacking,
        level=level,
        seed=int(seed),
    )
