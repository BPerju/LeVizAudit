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
    episode_placements,
    generate_rotation_angles,
    orientation_arrow_points,
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
        for obj in config.objects
    }

    # `count` is how many distinct rotations EACH position point cycles through across its
    # repeated visits. "uniform" shares the same (user-controlled-start) angle list across
    # every point; "random" gives each point its OWN independently-seeded list (seed + point
    # index), so different points actually point in different directions.
    rotations: dict[str, list[list[float]]] = {}
    for obj in config.objects:
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
        # The entire scene-level model (per_episode/combinations/order/stacking/level) is
        # encapsulated in pattern.episode_placements -- see CLAUDE.md for the rationale.
        point_lists = [patterns[obj.name] for obj in config.objects]
        placements_per_ordinal, residual = episode_placements(
            point_lists, episode_index, config.per_episode, config.combinations,
            config.order, config.stacking, config.level, config.seed,
        )
        if residual:
            names = ", ".join(config.objects[o].name for o in residual)
            logger.warning(
                "stacking: keep_apart -- could not separate %s at episode %d "
                "(all of their own points are taken)", names, episode_index,
            )
        if not placements_per_ordinal:
            logger.info("No object has any assigned point -- nothing to show for episode %d", episode_index)
            return

        for ordinal, obj in enumerate(config.objects):
            placements = placements_per_ordinal.get(ordinal)
            if not placements:
                clear_target(config.camera_key, obj.name)
                continue

            full_placements: list[tuple[Point, Point | None, int, int]] = []
            for site_point, level, stack_size, visit_number in placements:
                orientation_tip = None
                if obj.orientation is not None:
                    position_index = point_lists[ordinal].index(site_point)
                    angle_deg = target_for_episode(rotations[obj.name][position_index], visit_number)
                    _, orientation_tip = orientation_arrow_points(
                        site_point, angle_deg, obj.orientation.arrow_length, homography
                    )
                full_placements.append((site_point, orientation_tip, level, stack_size))

            primary_point, primary_tip, primary_level, primary_stack_size = full_placements[0]
            log_target(
                config.camera_key, primary_point, obj.name, obj.marker,
                orientation_tip=primary_tip, level=primary_level, stack_size=primary_stack_size,
                extra_placements=full_placements[1:] or None,
            )
        logger.info("Showing targets for episode %d", episode_index)

    # Episode 0's target must be visible before the operator starts moving -- show it
    # immediately, then let the watcher drive every subsequent episode.
    show_targets(0)

    watcher = EpisodeBoundaryWatcher(dataset_root, config.camera_key, info.fps)
    for finished_episode_index in watcher.next_episode_boundary():
        show_targets(finished_episode_index + 1)
