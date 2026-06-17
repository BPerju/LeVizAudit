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
from vizaudit.overlay.pattern import Point, build_pattern, target_for_episode
from vizaudit.overlay.perspective import canonical_rect_dims, compute_homography
from vizaudit.overlay.rerun_client import connect, log_target

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

    # Computed once: corrects sector sampling for a non-top-down camera, and confines it to
    # the marked workspace rectangle (not just the sector itself). arc/line ignore both.
    homography = None
    bounds = None
    if config.surface_calibration is not None:
        homography = compute_homography(
            config.surface_calibration.corners, config.surface_calibration.aspect_ratio
        )
        bounds = canonical_rect_dims(
            config.surface_calibration.corners, config.surface_calibration.aspect_ratio
        )

    # Patterns are static for the whole session, built once up front -- not recomputed
    # per episode.
    patterns: dict[str, list[Point]] = {
        obj.name: build_pattern(obj.pattern, obj.count, config.exclude_zones, homography, bounds)
        for obj in variable_objects
    }

    logger.info("Connecting to Rerun at %s:%s ...", rerun_host, rerun_port)
    connect(host=rerun_host, port=rerun_port)

    def show_targets(episode_index: int) -> None:
        for obj in variable_objects:
            point = target_for_episode(patterns[obj.name], episode_index)
            log_target(config.camera_key, point, obj.name, config.marker)
        logger.info("Showing targets for episode %d", episode_index)

    # Episode 0's target must be visible before the operator starts moving -- show it
    # immediately, then let the watcher drive every subsequent episode.
    show_targets(0)

    watcher = EpisodeBoundaryWatcher(dataset_root, config.camera_key, info.fps)
    for finished_episode_index in watcher.next_episode_boundary():
        show_targets(finished_episode_index + 1)
