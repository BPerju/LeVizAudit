#!/usr/bin/env python
"""Fakes a lerobot-record session, for testing vizaudit-overlay with no robot/camera/
lerobot install needed.

Mimics the two things vizaudit-overlay actually depends on from a real session:
1. Writes meta/info.json + per-frame PNGs into a fake dataset root, at the same path
   layout DatasetWriter.add_frame() uses, pausing between episodes to simulate the reset
   window -- this is what EpisodeBoundaryWatcher detects.
2. Logs a dummy image to the same Rerun entity path lerobot-record's log_rerun_data()
   uses, via the same shared (application_id, recording_id) pair vizaudit-overlay and
   scripts/vizaudit_record.py use, so both processes' data lands in one merged recording.

Usage (three terminals, all in the `vizaudit` conda env except the server doesn't need
any env):

    # 1.
    python -m rerun --serve-web --port 9876

    # 2.
    python examples/fake_lerobot_session.py --root /tmp/vizaudit_fake_dataset \
        --fps 10 --num-episodes 3 --episode-time-s 4 --reset-time-s 3 --rerun-port 9876

    # 3.
    vizaudit-overlay --config examples/pattern.example.yaml --connect 127.0.0.1:9876 \
        --dataset.root /tmp/vizaudit_fake_dataset --dataset.repo_id ignored/unused

Then open the printed web-viewer URL (default http://127.0.0.1:9090) and confirm the
target marker appears composited on the dummy image, advancing each episode.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import rerun as rr

from vizaudit.overlay.dataset_watcher import episode_image_dir
from vizaudit.overlay.rerun_client import connect


def write_info_json(root: Path, camera_key: str, fps: float) -> None:
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    info = {"fps": fps, "features": {camera_key: {"dtype": "video", "shape": [480, 640, 3]}}}
    (meta_dir / "info.json").write_text(json.dumps(info))


def dummy_frame() -> np.ndarray:
    frame = np.full((480, 640, 3), 60, dtype=np.uint8)
    frame[::20, :, :] = 120  # faint grid so it isn't a flat color in the viewer
    frame[:, ::20, :] = 120
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Fake dataset root (created/cleared).")
    parser.add_argument("--camera-key", default="observation.images.top")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--episode-time-s", type=float, default=4.0)
    parser.add_argument("--reset-time-s", type=float, default=3.0)
    parser.add_argument("--rerun-host", default="127.0.0.1")
    parser.add_argument("--rerun-port", type=int, default=9876)
    args = parser.parse_args()

    root = Path(args.root)
    if root.exists():
        shutil.rmtree(root)
    write_info_json(root, args.camera_key, args.fps)

    connect(host=args.rerun_host, port=args.rerun_port)
    frame = dummy_frame()
    frame_interval = 1.0 / args.fps

    for episode_index in range(args.num_episodes):
        print(f"Recording fake episode {episode_index} ...", flush=True)
        ep_dir = episode_image_dir(root, args.camera_key, episode_index)
        ep_dir.mkdir(parents=True, exist_ok=True)
        for frame_index in range(int(args.episode_time_s * args.fps)):
            (ep_dir / f"frame-{frame_index:06d}.png").write_bytes(b"")
            rr.log(args.camera_key, rr.Image(frame), static=True)
            time.sleep(frame_interval)
        print(f"Reset window ({args.reset_time_s}s) -- place the next target now", flush=True)
        time.sleep(args.reset_time_s)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
