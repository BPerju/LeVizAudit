import shutil
import threading
import time
from pathlib import Path

from vizaudit.overlay.dataset_watcher import (
    EpisodeBoundaryWatcher,
    episode_image_dir,
    read_dataset_info,
    resolve_dataset_root,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_resolve_dataset_root_explicit_override(tmp_path):
    root = resolve_dataset_root("someorg/somedataset", str(tmp_path))
    assert root == tmp_path


def test_resolve_dataset_root_default(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))
    root = resolve_dataset_root("someorg/somedataset", None)
    assert root == tmp_path / "someorg/somedataset"


def test_read_dataset_info(tmp_path):
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    shutil.copy(FIXTURES / "fake_info.json", meta_dir / "info.json")

    info = read_dataset_info(tmp_path)
    assert info.fps == 10.0
    assert info.image_keys == ["observation.images.top", "observation.images.wrist"]


def _write_frame(root: Path, camera_key: str, episode_index: int, frame_index: int) -> None:
    ep_dir = episode_image_dir(root, camera_key, episode_index)
    ep_dir.mkdir(parents=True, exist_ok=True)
    (ep_dir / f"frame-{frame_index:06d}.png").write_bytes(b"")


def test_episode_boundary_watcher_detects_idle_transition(tmp_path):
    camera_key = "observation.images.top"
    watcher = EpisodeBoundaryWatcher(
        tmp_path, camera_key, fps=30.0, idle_threshold_s=0.15, poll_interval_s=0.02
    )
    frame_interval = 0.02

    def writer():
        for i in range(8):
            _write_frame(tmp_path, camera_key, 0, i)
            time.sleep(frame_interval)

    t = threading.Thread(target=writer)
    t.start()
    boundaries = watcher.next_episode_boundary()
    assert next(boundaries) == 0
    t.join()


def test_episode_boundary_watcher_multiple_episodes(tmp_path):
    camera_key = "observation.images.top"
    watcher = EpisodeBoundaryWatcher(
        tmp_path, camera_key, fps=30.0, idle_threshold_s=0.15, poll_interval_s=0.02
    )
    frame_interval = 0.02
    reset_gap = 0.3

    def writer():
        for ep in range(2):
            for i in range(8):
                _write_frame(tmp_path, camera_key, ep, i)
                time.sleep(frame_interval)
            time.sleep(reset_gap)

    t = threading.Thread(target=writer)
    t.start()
    boundaries = watcher.next_episode_boundary()
    assert next(boundaries) == 0
    assert next(boundaries) == 1
    t.join()
