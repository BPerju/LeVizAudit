"""Dataset-root resolution and episode-boundary detection.

Filesystem-facing only — no Rerun imports. Deliberately re-implements (rather than
imports) the handful of lerobot internals it depends on, to keep this package's import
graph light and decoupled from lerobot-internal churn. Each re-implementation cites the
exact lerobot source it mirrors; if lerobot changes that logic, these comments are the
trip wire to notice.

Episode-boundary detection rationale (see CLAUDE.md "Key decisions"): lerobot-record's
Rerun stream carries no episode marker, and `dataset.save_episode()` fires *after* the
reset window for that transition has already elapsed — too late to use as a "show the
next target" trigger. `DatasetWriter.add_frame()` does write a PNG to disk immediately,
every frame, during recording (regardless of `streaming_encoding`), and those writes stop
the instant the reset phase begins. `EpisodeBoundaryWatcher` below polls that per-episode
image directory's file-arrival cadence and fires the moment it goes idle.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

# Mirrors lerobot/src/lerobot/datasets/utils.py:94 (`DEFAULT_IMAGE_PATH`) — only the
# per-episode directory portion is needed here, not individual frame paths.
_EPISODE_IMAGE_DIR_TEMPLATE = "images/{image_key}/episode-{episode_index:06d}"


def _hf_home() -> Path:
    return Path(os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))).expanduser()


def resolve_dataset_root(repo_id: str, root_override: str | Path | None) -> Path:
    """Mirrors lerobot/src/lerobot/utils/constants.py:68 and the root-resolution logic in
    lerobot/src/lerobot/datasets/dataset_metadata.py: an explicit root wins, otherwise it's
    `$HF_LEROBOT_HOME/<repo_id>`, defaulting to `~/.cache/huggingface/lerobot/<repo_id>`."""
    if root_override is not None:
        return Path(root_override).expanduser()
    lerobot_home = Path(os.getenv("HF_LEROBOT_HOME", str(_hf_home() / "lerobot"))).expanduser()
    return lerobot_home / repo_id


def episode_image_dir(root: Path, camera_key: str, episode_index: int) -> Path:
    return root / _EPISODE_IMAGE_DIR_TEMPLATE.format(image_key=camera_key, episode_index=episode_index)


@dataclass(frozen=True)
class DatasetInfo:
    fps: float
    image_keys: list[str]


def read_dataset_info(root: Path) -> DatasetInfo:
    """Parses `<root>/meta/info.json`, written at `LeRobotDataset.create()` time, before
    any episode is recorded — so this is readable as soon as the dataset root exists."""
    with (root / "meta" / "info.json").open() as f:
        raw = json.load(f)
    features = raw.get("features", {})
    image_keys = [
        key
        for key, feature in features.items()
        if isinstance(feature, dict) and feature.get("dtype") in ("image", "video")
    ]
    return DatasetInfo(fps=float(raw["fps"]), image_keys=image_keys)


def wait_for_dataset_root(root: Path, poll_interval_s: float = 1.0, timeout_s: float | None = None) -> None:
    """Blocks until `<root>/meta/info.json` exists, for the case where the overlay starts
    before `lerobot-record` has created the dataset."""
    info_path = root / "meta" / "info.json"
    start = time.monotonic()
    while not info_path.exists():
        if timeout_s is not None and time.monotonic() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for {info_path} to appear")
        time.sleep(poll_interval_s)


class EpisodeBoundaryWatcher:
    """Polls per-episode image-write cadence to detect recording->reset transitions.

    No `watchdog`/inotify dependency for v1 — plain polling is simple to reason about and
    test, and latency is bounded by `fps` (see CLAUDE.md's open items on this tradeoff).
    """

    def __init__(
        self,
        root: Path,
        camera_key: str,
        fps: float,
        idle_threshold_s: float | None = None,
        poll_interval_s: float | None = None,
    ) -> None:
        self._root = root
        self._camera_key = camera_key
        self._fps = fps
        # 2x the inter-frame interval: comfortably larger than normal frame-to-frame
        # gaps during active recording, small enough to fire promptly once it stops.
        self._idle_threshold_s = idle_threshold_s if idle_threshold_s is not None else 2.0 / fps
        self._poll_interval_s = poll_interval_s if poll_interval_s is not None else max(0.5 / fps, 0.05)

    def _latest_frame_mtime(self, episode_index: int) -> float | None:
        ep_dir = episode_image_dir(self._root, self._camera_key, episode_index)
        if not ep_dir.is_dir():
            return None
        mtimes = [p.stat().st_mtime for p in ep_dir.iterdir() if p.is_file()]
        return max(mtimes) if mtimes else None

    def wait_for_episode_start(self, episode_index: int) -> None:
        """Blocks until episode_index's image directory has at least one frame written."""
        while self._latest_frame_mtime(episode_index) is None:
            time.sleep(self._poll_interval_s)

    def wait_for_episode_end(self, episode_index: int) -> None:
        """Blocks until episode_index's frame writes have gone idle for the threshold."""
        while True:
            time.sleep(self._poll_interval_s)
            mtime = self._latest_frame_mtime(episode_index)
            if mtime is not None and (time.time() - mtime) > self._idle_threshold_s:
                return

    def next_episode_boundary(self) -> Iterator[int]:
        """Yields `episode_index` the moment recording-phase-`episode_index` goes idle —
        i.e. the reset window has begun and the *next* episode's target should be shown."""
        episode_index = 0
        while True:
            self.wait_for_episode_start(episode_index)
            self.wait_for_episode_end(episode_index)
            yield episode_index
            episode_index += 1
