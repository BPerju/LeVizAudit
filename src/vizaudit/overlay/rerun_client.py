"""Rerun-facing logging for the guided overlay's target markers.

Filesystem/dataset-facing code does not belong here — see dataset_watcher.py for that.
"""

from __future__ import annotations

import rerun as rr

from vizaudit.overlay.config import MarkerConfig

# Rerun's actual recording identity is the (application_id, recording_id) PAIR, not
# recording_id alone -- empirically verified via the dataframe API (`rr.dataframe.
# load_archive(...).all_recordings()`): two processes sharing only recording_id still
# produced 2 distinct recordings; matching application_id too collapsed it to 1. Neither
# the rr.init() docstring nor the viewer UI make this obvious (the viewer just silently
# showed the marker in a second, image-less view instead of erroring). So both processes
# must use the IDENTICAL application_id, not just the identical recording_id.
#
# SHARED_APPLICATION_ID is "recording" specifically because that's the literal hardcoded
# session_name lerobot_record.py's record() passes to init_rerun() -- so
# scripts/vizaudit_record.py's monkeypatch only needs to inject recording_id; application_id
# already matches for free. Both constants MUST stay in sync with the copies in
# scripts/vizaudit_record.py (cross-referenced there too); a mismatch silently breaks
# compositing with no visible error -- the marker just lands in its own empty view.
#
# NOTE: pinning multiprocessing.current_process().authkey to a shared value does NOT
# achieve any of this, despite rr.init()'s docstring suggesting it would -- empirically
# verified separately (two independent processes with the same pinned authkey still get
# different default recording_ids; only true multiprocessing-spawned children, which
# inherit more than just the authkey value via fork/spawn, share the default). Passing
# recording_id= explicitly is the actual supported mechanism for that part. See CLAUDE.md's
# "Key decisions" for the full story of both bugs found while building this.
SHARED_APPLICATION_ID = "recording"
SHARED_RECORDING_ID = "8122e1bc-273e-4f00-a0ee-ab0b15c44107"


def connect(host: str, port: int) -> None:
    """Connects to the shared Rerun gRPC server using the same URL scheme lerobot's own
    `init_rerun()` uses (`rerun+http://host:port/proxy`), pinned to the shared recording."""
    rr.init(SHARED_APPLICATION_ID, recording_id=SHARED_RECORDING_ID)
    rr.connect_grpc(url=f"rerun+http://{host}:{port}/proxy")


def log_target(
    camera_key: str,
    point: tuple[float, float],
    object_name: str,
    marker: MarkerConfig,
    orientation_tip: tuple[float, float] | None = None,
) -> None:
    """Logs a target marker as a child entity of the live camera image entity, so Rerun's
    Spatial2D view composites it directly on top of the feed.

    Re-logging to the same path (one fixed path per object) replaces the prior marker in
    the viewer's current-time view, so no `rr.Clear` is needed here.

    ``orientation_tip``, if given, draws an arrow from ``point`` to it as a separate child
    entity -- the already-projected orientation guide (see
    ``pattern.orientation_arrow_points``, which does the homography-aware projection; this
    function only logs the two points it's handed, with no perspective math of its own).
    Whether an object has an orientation arrow at all is fixed for the whole session (set by
    its config, not toggled per episode), so there's no need to ever clear a stale one here.
    """
    path = f"{camera_key}/target/{object_name}"
    rr.log(
        path,
        rr.Points2D(
            positions=[point],
            radii=marker.radius_px,
            colors=[marker.color_rgba],
            labels=[object_name] if marker.label else None,
        ),
    )
    if orientation_tip is not None:
        rr.log(
            f"{path}/orientation",
            rr.Arrows2D(
                origins=[point],
                vectors=[(orientation_tip[0] - point[0], orientation_tip[1] - point[1])],
                colors=[marker.color_rgba],
            ),
        )
