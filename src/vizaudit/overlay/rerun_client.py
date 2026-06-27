"""Rerun-facing logging for the guided overlay's target markers.

Filesystem/dataset-facing code does not belong here — see dataset_watcher.py for that.
"""

from __future__ import annotations

import math

import rerun as rr

from vizaudit.overlay.config import MarkerConfig

Point = tuple[float, float]

# Mirrors the calibration tool's OBJECT_STACK_FAN_RADIUS/stackOffsetPosition exactly (see
# calibrate.js) so a stacked marker's visual layout in the live overlay matches what the
# operator already previewed while authoring the config.
STACK_FAN_RADIUS = 7.0


def _stack_offset_position(point: Point, level: int, stack_size: int) -> Point:
    if stack_size <= 1:
        return point
    angle = 2 * math.pi * level / stack_size
    return (point[0] + STACK_FAN_RADIUS * math.cos(angle), point[1] + STACK_FAN_RADIUS * math.sin(angle))

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
    level: int = 0,
    stack_size: int = 1,
    extra_placements: list[tuple[Point, Point | None, int, int]] | None = None,
) -> None:
    """Logs a target marker as a child entity of the live camera image entity, so Rerun's
    Spatial2D view composites it directly on top of the feed.

    Re-logging to the same path (one fixed path per object) replaces the prior marker in
    the viewer's current-time view, so no `rr.Clear` is needed here -- see `clear_target` for
    the one case that does need it (an object not active at all this episode, under
    `OverlayConfig.site_selection == "subset"`).

    ``level``/``stack_size`` (default ``0``/``1``, today's exact behavior -- a lone marker
    with no offset or badge) describe this object's place in a STACK of objects sharing one
    point this episode -- always derived from 2+ objects' patterns landing on the same point
    (`co_location: "stack"`), including a `"keep_apart"` residual collision that couldn't be
    avoided; session.py decides which objects are involved, this function only renders the
    result. When ``stack_size > 1``, the marker is fanned out by ``level`` around ``point``
    (see `_stack_offset_position`, mirroring the calibration tool's tower-legend anchor point
    exactly) so stacked markers are visually distinguishable, and the level is appended to its
    label.

    ``orientation_tip``, if given, draws an arrow from ``point`` to it as a separate child
    entity -- the already-projected orientation guide (see
    ``pattern.orientation_arrow_points``, which does the homography-aware projection; this
    function only logs the two points it's handed, with no perspective math of its own). The
    arrow is translated by the same fan offset as its marker, so it stays visually anchored
    to it. Whether an object has an orientation arrow at all is fixed for the whole session
    (set by its config, not toggled per episode), so there's no need to ever clear a stale one
    here.

    ``extra_placements``, if given, is a list of additional SIMULTANEOUS ``(point,
    orientation_tip, level, stack_size)`` tuples for this SAME object -- needed under
    `OverlayConfig.position_mode == "overlay"` when an object's own pattern has more than one
    point: every one of its points is shown at once (not swept), so one marker no longer
    represents the whole picture. All placements (the primary one plus every extra) are logged
    as N fanned positions/labels under ONE entity (and N orientation arrows, skipping any
    placement with no tip, under one child entity) -- a single `rr.log` call per entity means
    re-logging always fully replaces every previous placement, even if a PREVIOUS episode had
    MORE of them, with no separate clearing needed for the shrinking case. Omitting
    ``extra_placements`` (the default, and every pre-existing call site) is byte-for-byte
    unchanged from the single-marker behavior this function always had.
    """
    placements = [(point, orientation_tip, level, stack_size)] + (extra_placements or [])
    path = f"{camera_key}/target/{object_name}"
    display_points = [_stack_offset_position(p, lv, sz) for p, _, lv, sz in placements]
    labels = [f"{object_name} L{lv}" if sz > 1 else object_name for _, _, lv, sz in placements]
    rr.log(
        path,
        rr.Points2D(
            positions=display_points,
            radii=marker.radius_px,
            colors=[marker.color_rgba] * len(display_points),
            labels=labels if marker.label else None,
        ),
    )
    arrow_origins = []
    arrow_vectors = []
    for (p, tip, _lv, _sz), display_point in zip(placements, display_points):
        if tip is None:
            continue
        arrow_origins.append(display_point)
        arrow_vectors.append((tip[0] - p[0], tip[1] - p[1]))
    if arrow_origins:
        rr.log(
            f"{path}/orientation",
            rr.Arrows2D(
                origins=arrow_origins,
                vectors=arrow_vectors,
                colors=[marker.color_rgba] * len(arrow_origins),
            ),
        )


def clear_target(camera_key: str, object_name: str) -> None:
    """Removes a previously-logged target marker (and its orientation arrow, if any).

    Re-logging to the same path normally replaces stale data for free (see `log_target`'s
    docstring) -- but an object not in any of this episode's SELECTED sites (under
    `OverlayConfig.site_selection == "subset"` -- session.py's `show_targets` picks a subset of
    this episode's sites to show, see CLAUDE.md) has nothing new to log at all, so skipping the
    call entirely would leave the PREVIOUS episode's marker visibly stuck on screen instead of
    disappearing.
    """
    path = f"{camera_key}/target/{object_name}"
    rr.log(path, rr.Clear(recursive=True))
