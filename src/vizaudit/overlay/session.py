"""Thin orchestrator wiring config + pattern + dataset_watcher + rerun_client together.

Deliberately has little logic of its own -- logic belongs in the modules it calls.
"""

from __future__ import annotations

import logging
from pathlib import Path

from vizaudit.overlay.config import OverlayConfig
from vizaudit.overlay.dataset_watcher import (
    EpisodeBoundaryWatcher,
    read_dataset_info,
    wait_for_dataset_root,
)
from vizaudit.overlay.pattern import (
    Point,
    build_pattern,
    combination_period,
    generate_rotation_angles,
    level_order,
    occupied_sites,
    orientation_arrow_points,
    resolve_combination_indices,
    resolve_keep_apart,
    sequence_index,
    target_for_episode,
)
from vizaudit.overlay.perspective import canonical_rect_dims, compute_homography
from vizaudit.overlay.rerun_client import clear_target, connect, log_target

logger = logging.getLogger(__name__)


def run_session(config: OverlayConfig, dataset_root: Path, rerun_host: str, rerun_port: int) -> None:
    """Runs until interrupted (Ctrl-C). Does not attempt to infer
    `--dataset.num_episodes` from lerobot-record's own invocation -- out of scope."""
    logger.info("Waiting for dataset root %s ...", dataset_root)
    wait_for_dataset_root(dataset_root)

    info = read_dataset_info(dataset_root)
    if config.camera_key not in info.image_keys:
        raise ValueError(
            f"camera_key {config.camera_key!r} not found in dataset features; "
            f"available image keys: {info.image_keys}"
        )

    variable_objects = [obj for obj in config.objects if obj.variable]
    if not variable_objects:
        logger.warning("No variable objects configured -- nothing to guide. Exiting.")
        return
    num_objects = len(variable_objects)

    # Computed once: corrects sector sampling for a non-top-down camera, and confines it to
    # the marked workspace rectangle (not just the sector itself). arc/line/points ignore both.
    homography = None
    bounds = None
    if config.surface_calibration is not None:
        homography = compute_homography(
            config.surface_calibration.corners, config.surface_calibration.aspect_ratio
        )
        bounds = canonical_rect_dims(
            config.surface_calibration.corners, config.surface_calibration.aspect_ratio
        )

    # Patterns are static for the whole session, built once up front -- not recomputed per
    # episode.
    patterns: dict[str, list[Point]] = {
        obj.name: build_pattern(obj.pattern, obj.count, config.exclude_zones, homography, bounds)
        for obj in variable_objects
    }

    # `count` is how many distinct rotations EACH position point cycles through across its
    # repeated visits. "uniform" shares the same (user-controlled-start) angle list across
    # every point; "random" gives each point its OWN independently-seeded list (seed + point
    # index), so different points actually point in different directions.
    rotations: dict[str, list[list[float]]] = {}
    for obj in variable_objects:
        if obj.orientation is None:
            continue
        n = len(patterns[obj.name])
        if obj.orientation.method == "random":
            rotations[obj.name] = [
                generate_rotation_angles(
                    obj.orientation.count, "random", obj.orientation.angle_start_deg,
                    obj.orientation.angle_end_deg, obj.orientation.seed + p,
                    initial_angle_deg=obj.orientation.initial_angle_deg,
                )
                for p in range(n)
            ]
        else:
            shared = generate_rotation_angles(
                obj.orientation.count, "uniform", obj.orientation.angle_start_deg,
                obj.orientation.angle_end_deg, obj.orientation.seed,
                initial_angle_deg=obj.orientation.initial_angle_deg,
            )
            rotations[obj.name] = [shared] * n

    logger.info("Connecting to Rerun at %s:%s ...", rerun_host, rerun_port)
    connect(host=rerun_host, port=rerun_port)

    def show_targets(episode_index: int) -> None:
        if config.episode_targets == "one":
            # SCENE-LEVEL choice (config.episode_targets == "one"): each episode shows exactly
            # ONE occupied point's full membership -- a single object, or, since assignment to
            # the same point is now a GUARANTEED stack (never a per-episode coincidence), several
            # at once. Every other object is cleared. `sites` is the STATIC union of every
            # object's own assigned points (pattern.occupied_sites) -- co_location/sequencing
            # don't apply here, since which objects share a point is a fixed, authored fact, not
            # something that can or can't coincide episode to episode. #episodes therefore equals
            # the number of distinct occupied points, not the longest object's pattern length --
            # see CLAUDE.md for the symptom this fixes (stacks were rare/partial under an earlier
            # dynamic-coincidence design, and the calibration preview separately over-multiplied
            # its own episode count).
            sites = occupied_sites([patterns[obj.name] for obj in variable_objects])
            if not sites:
                logger.info("No object has any assigned point -- nothing to show for episode %d", episode_index)
                return
            site_point, member_ordinals = sites[episode_index % len(sites)]
            active_ordinals = set(member_ordinals)
            stack_size = len(member_ordinals)
            levels = level_order(stack_size, episode_index, config.level_strategy, config.level_seed)
            visit_number = episode_index // len(sites)
            for ordinal, obj in enumerate(variable_objects):
                if ordinal not in active_ordinals:
                    clear_target(config.camera_key, obj.name)
                    continue
                level = levels[member_ordinals.index(ordinal)] if stack_size > 1 else 0
                orientation_tip = None
                if obj.orientation is not None:
                    position_index = patterns[obj.name].index(site_point)
                    angle_deg = target_for_episode(rotations[obj.name][position_index], visit_number)
                    _, orientation_tip = orientation_arrow_points(
                        site_point, angle_deg, obj.orientation.arrow_length, homography
                    )
                log_target(
                    config.camera_key, site_point, obj.name, obj.marker,
                    orientation_tip=orientation_tip, level=level, stack_size=stack_size,
                )
            logger.info("Showing targets for episode %d", episode_index)
            return

        # config.episode_targets == "all": every site (an object, or several objects bonded
        # together because their own sweeps happen to coincide this episode) is shown
        # simultaneously, every episode -- today's original behavior. Each object's own index is
        # either one of the 5 `resolve_combination_indices` modes (when the combination
        # subsystem is active -- decoupling the visible combination count from any one object's
        # own pattern length, reported directly: "if i have 3 objects and 20 steps the
        # combinations are limited to the nr of steps") or the legacy per-object
        # `sequence_index` sweep (today's exact lockstep/shuffled behavior, unchanged when
        # inactive). Active whenever `combination_count` is set OR `combination_mode` is
        # anything other than its "systematic" default -- "random"/"coprime" have no count of
        # their own and need one set explicitly (validated eagerly in config.py); "cartesian"/
        # "lcm" derive their own natural period (the product, or the lcm, of every object's own
        # length) when left unset.
        combination_active = config.combination_count is not None or config.combination_mode != "systematic"
        if combination_active:
            lengths = [len(patterns[obj.name]) for obj in variable_objects]
            period = combination_period(lengths, config.combination_mode, config.combination_count)
            natural_indices = resolve_combination_indices(
                episode_index % period, lengths, config.combination_mode, period, config.combination_seed
            )
        else:
            natural_indices = [
                sequence_index(
                    episode_index, len(patterns[obj.name]), obj.sequencing, ordinal, num_objects,
                    obj.pattern.seed if obj.pattern.seed is not None else 0,
                )
                for ordinal, obj in enumerate(variable_objects)
            ]

        # SCENE-LEVEL choice (config.co_location): "keep_apart" nudges a colliding object onto
        # one of its OWN other assigned points where possible (resolve_keep_apart) before
        # anything else runs; "stack" (default) leaves natural indices untouched, so two
        # objects whose patterns happen to coincide this episode are grouped into a real stack
        # below.
        if config.co_location == "keep_apart":
            assigned_lists = [list(range(len(patterns[obj.name]))) for obj in variable_objects]
            resolved_indices, residual = resolve_keep_apart(natural_indices, assigned_lists, episode_index)
            if residual:
                names = ", ".join(variable_objects[o].name for o in residual)
                logger.warning(
                    "co_location: keep_apart -- could not separate %s at episode %d "
                    "(all of their own points are taken)", names, episode_index,
                )
        else:
            resolved_indices = natural_indices

        # Grouped by exact resolved-position equality -- recomputed fresh EVERY episode: a
        # "site" here is whichever object(s) actually coincide THIS episode, which can be a lone
        # object or a real stack, and can change shape episode to episode as each object's own
        # independent sweep moves.
        base_positions: dict[str, Point] = {}
        position_indices: dict[str, int] = {}
        for ordinal, obj in enumerate(variable_objects):
            idx = resolved_indices[ordinal]
            base_positions[obj.name] = patterns[obj.name][idx]
            position_indices[obj.name] = idx

        groups: dict[Point, list[str]] = {}
        for name, pos in base_positions.items():
            groups.setdefault(pos, []).append(name)
        declaration_order = {obj.name: i for i, obj in enumerate(variable_objects)}
        for members in groups.values():
            members.sort(key=lambda n: declaration_order[n])

        for obj in variable_objects:
            point = base_positions[obj.name]
            group = groups[point]
            stack_size = len(group)
            level = (
                level_order(stack_size, episode_index, config.level_strategy, config.level_seed)[group.index(obj.name)]
                if stack_size > 1 else 0
            )

            orientation_tip = None
            if obj.orientation is not None:
                position_index = position_indices[obj.name]
                num_points = len(patterns[obj.name])
                # Approximates "how many times this position has been visited so far" as
                # episode_index // num_points -- exact under "lockstep"/"shuffled" (each is a
                # clean permutation of one period), only approximate on the rare episode where
                # "keep_apart" nudges an object off its own natural cadence.
                visit_number = episode_index // num_points
                angle_deg = target_for_episode(rotations[obj.name][position_index], visit_number)
                _, orientation_tip = orientation_arrow_points(
                    point, angle_deg, obj.orientation.arrow_length, homography
                )
            log_target(
                config.camera_key, point, obj.name, obj.marker,
                orientation_tip=orientation_tip, level=level, stack_size=stack_size,
            )
        logger.info("Showing targets for episode %d", episode_index)

    # Episode 0's target must be visible before the operator starts moving -- show it
    # immediately, then let the watcher drive every subsequent episode.
    show_targets(0)

    watcher = EpisodeBoundaryWatcher(dataset_root, config.camera_key, info.fps)
    for finished_episode_index in watcher.next_episode_boundary():
        show_targets(finished_episode_index + 1)
