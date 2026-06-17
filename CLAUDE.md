This file provides guidance to AI agents when working with code in this repository.

## Project overview

VizAudit audits the **spatial diversity** of robot demonstration data — e.g., for a
pick-place task, did the target object actually get placed in a diverse enough set of
positions across episodes — using purely visual (camera-frame) detection. It has two
delivery surfaces sharing one computation core:

- **Phase 1 (live):** a *guided data-collection* overlay inside the Rerun viewer that
  `lerobot-record --display_data` already opens. For each episode it directs the operator to
  place an object at a specific point on a configurable spatial pattern (e.g. a pixel-space
  semicircle), so placement sweeps a diverse set of positions across a session instead of
  being haphazard. v1 is directive-only (show the next target); auto-detecting where the
  operator actually placed the object and closing the loop is a deferred v2.
- **Phase 2 (post-hoc):** a metrics panel inside `lerobot-dataset-visualizer` (the Next.js
  dataset browser), summarizing spatial diversity per dataset/episode after recording.

`research_report.md` in this repo is the full research record behind every decision below —
read it before changing architecture, not just this file.

## Status

Phase 1 (guided overlay) in active implementation. Phase 2 (dataset-visualizer sidecar) is
still pre-implementation — see `research_report.md` §3/§5/§7 for that design.

## Key decisions (why, not just what)

- **Vision-only, no FK.** Confirmed against a real dataset (`Bperju/so101_cube_pick_place_50`,
  see report §1/§7) that `observation.state`/`action` are joint-angle space only — no
  Cartesian `eef_pos`/`eef_rot` anywhere in the schema. Forward kinematics would need a
  robot-specific URDF chain and isn't needed anyway: for a pick-place task, the thing worth
  auditing (object position diversity) is only visible in the camera image, not in joint
  state. The core engine consumes `observation.images.*` frames — never robot state.
- **Classical CV before ML detection.** v1 is OpenCV contour → PCA → `minAreaRect` on a fixed
  top-down camera. YOLO/MobileSAM/SAM/FoundPose (report §4) are deferred until classical CV
  proves insufficient on real footage (clutter, occlusion, variable lighting) — don't reach
  for a model until that happens.
- **One core engine, two thin sinks — never duplicate detection logic between phases.**
  Both phases call the same `compute_frame_metrics(frame: np.ndarray) -> dict`. Verified
  (report §7): live frames are already `np.ndarray`, HWC, `uint8`, RGB — the canonical format
  Phase 1 needs, zero conversion. `LeRobotDataset` frames come back as `torch.Tensor`, CHW,
  `float32` in `[0, 1]` — same RGB color space, so Phase 2 needs only a one-line adapter
  (`frame.permute(1, 2, 0).numpy() * 255` → `astype(np.uint8)`) before calling the same core
  function.
- **Standalone package, not a contribution to leLab or lerobot-dataset-visualizer.** leLab is
  hardcoded for SO-101 throughout (report §2); lerobot-dataset-visualizer is TS/Bun with no
  general-purpose Python compute backend (report §3). VizAudit stays Python, reads
  `LeRobotDataset` directly, and is robot-agnostic by construction (vision-only core, no
  per-robot code). It can be upstreamed into `lerobot/utils/spatial_audit.py` later once the
  interface is proven and dataset v3.0 stabilizes.
- **Phase 1 attaches via a shared Rerun gRPC server — never fork or patch `lerobot-record`.**
  Start a server first (`python -m rerun --serve-web --port <port>`), then point both
  `lerobot-record --display_ip/--display_port` and the VizAudit overlay process at it.
  VizAudit never touches lerobot's recording loop or source.
- **On-feed compositing needs a shared Rerun `recording_id`, passed explicitly — pinning
  `multiprocessing.authkey` does NOT work, despite what `rr.init()`'s docstring implies.**
  Verified against `rerun-sdk==0.26.2`: two independently-launched processes connecting to
  the same gRPC server do **not** automatically merge into one recording/view. An earlier
  version of this code pinned a shared `multiprocessing.current_process().authkey` before
  each process's `rr.init()` call, reasoning from the docstring's "default recording_id is
  based on `multiprocessing.current_process().authkey`" — **empirically falsified**: two
  independent processes with the byte-identical pinned authkey still got different
  recording_ids (confirmed via `rerun rrd print` on a saved `.rrd`, comparing `StoreId`
  values). Only genuine `multiprocessing`-spawned children (which inherit OS-level state via
  fork/spawn, not just the authkey value) actually share that default. The docstring's
  "you will need to manually assign them all the same recording_id... any random UUIDv4
  will work" line means literally pass `recording_id=` explicitly — confirmed this *does*
  merge two independent processes. Since `lerobot-record`'s `init_rerun()` calls
  `rr.init(session_name)` with no `recording_id`, and patching its source is off the table
  (see above), `scripts/vizaudit_record.py` instead monkeypatches `rerun.init` (an in-memory
  function-reference patch in our own process, not a file edit) before importing
  `lerobot.scripts.lerobot_record:main` — confirmed safe: `init_rerun()` does a lazy
  `import rerun as rr` inside its own function body, so the patch is picked up at call time
  regardless of import order. `overlay/rerun_client.py`'s `connect()` passes the same fixed
  `SHARED_RECORDING_ID` directly. Both constants are duplicated intentionally across the two
  files (cross-referenced by comment) — operators run `vizaudit_record.py` (in the `lerobot`
  env) instead of plain `lerobot-record` when they want the live guidance overlay.
- **A second, separate gotcha on top of the above: a Rerun recording's real identity is
  the `(application_id, recording_id)` *pair*, not `recording_id` alone — matching only
  `recording_id` still produced two distinct recordings.** Found this only after the
  `recording_id` fix above still showed the target marker in a separate, image-less view
  in the actual viewer ("the overlay is just a dark screen") — the viewer doesn't error on
  a mismatch, it silently renders the second process's data as its own empty view, so this
  is easy to miss without checking. Verified definitively via the dataframe API, not just
  the viewer: `rerun.dataframe.load_archive(path).all_recordings()` — two processes sharing
  only `recording_id` reported `num_recordings() == 2`; matching `application_id` too
  collapsed it to `1`. Fix: `overlay/rerun_client.py`'s `SHARED_APPLICATION_ID = "recording"`
  (chosen to match `lerobot_record.py`'s own hardcoded `init_rerun(session_name="recording",
  ...)` call, confirmed by reading that call site directly) — `connect()` always uses it, and
  `scripts/vizaudit_record.py`'s monkeypatch unconditionally overrides *both* values rather
  than relying on lerobot's "recording" string never changing.
- **Episode boundaries are detected from per-frame image-write cadence on disk, not from the
  Rerun stream or `dataset.save_episode()`.** Verified against `lerobot_record.py`: the Rerun
  stream carries no episode marker at all (recording and reset phases both log frames
  identically), and `save_episode()` fires *after* the reset window for that transition has
  already elapsed — too late to use as a "show the next target" trigger. `add_frame()` does
  write a PNG to `<dataset_root>/images/{image_key}/episode-{N:06d}/frame-{i:06d}.png`
  immediately every frame during recording (regardless of `streaming_encoding`), and those
  writes stop the instant the reset phase begins (reset's `record_loop()` call omits
  `dataset=`, so `add_frame()` never runs). `overlay/dataset_watcher.py`'s
  `EpisodeBoundaryWatcher` polls that directory's file-arrival cadence and fires when it goes
  idle for `~2/fps` — exactly the moment the operator should see the next target.
- **Pattern targets are pixel coordinates on the camera image, not physical arm-reach units.**
  Keeps the guidance config calibration-free and robot-agnostic, consistent with the
  vision-only/no-FK decision above — no per-robot reach data is ever read or needed.
- **Phase 2 ships a precomputed sidecar artifact (JSON/parquet), not a live backend call.**
  The dataset-visualizer's panels are pure frontend (parquet via `hyparquet`) — the heavy
  Python computation has to happen ahead of time and get written next to the dataset, not be
  invoked on page load.

## Environment

This project has its own conda env, **`vizaudit`, separate from the `lerobot` training env**
on this machine — that env carries every policy/training extra (act, diffusion, pi0,
smolvla, hilserl, aloha, gym envs, etc.), which this tool doesn't need and shouldn't depend on.

```bash
conda create -n vizaudit python=3.12 -y
conda activate vizaudit
pip install -e "/home/bogdan/lerobot[viz,dataset]"   # LeRobotDataset + rerun-sdk, no training extras
```

Installed and verified working end-to-end (loaded a real frame via `LeRobotDataset`) on
2026-06-17: `lerobot 0.5.2` (editable, against the local `/home/bogdan/lerobot` checkout),
`rerun-sdk 0.26.2`, `opencv-python-headless 4.13.0`, `torch 2.11.0+cu130`, `av 15.1.0`,
`numpy 2.2.6`.

`lerobot` is installed **editable** against `/home/bogdan/lerobot` — if that checkout moves
or its dataset-reading code changes incompatibly, reinstall with the command above. Don't add
training-only dependencies (no `transformers`, no policy extras) to this env; if a future
piece of work needs them, that's a sign it belongs in `lerobot` proper, not here.

The `vizaudit` package itself (`pyproject.toml`) declares only `rerun-sdk`, `numpy`, `pyyaml`
as direct dependencies — Phase 1 v1 never opens or analyzes a camera frame (it only watches
file-arrival timestamps and logs synthetic target points), so `opencv-python-headless` is
intentionally **not** a direct dependency yet even though it's present in the conda env via
`lerobot[viz,dataset]`; add it back to `pyproject.toml` when the deferred v2 auto-detect
feature (or Phase 2's `core/`) actually needs `cv2`. `scripts/vizaudit_record.py` has zero
dependency on the `vizaudit` package at all (stdlib `multiprocessing` + `lerobot` only) and
must be run from the `lerobot` env, never `vizaudit`'s — it needs the `hardware` extra
(robot/teleop drivers) that `vizaudit`'s env deliberately excludes.

## Planned architecture

```
src/vizaudit/
  core/                    # compute_frame_metrics(frame: np.ndarray) -> dict — placeholder until a later
                            # phase; pure vision only, no Rerun/dataset imports, ever
  overlay/                 # Phase 1: guided data-collection overlay
    config.py              # YAML pattern/object config — dataclasses + validation
    pattern.py             # pure pattern-generation functions (arc, line; pixel space)
    dataset_watcher.py      # dataset-root resolution + EpisodeBoundaryWatcher (file-write-cadence polling)
    rerun_client.py         # rr.init(recording_id=...) + connect_grpc + target-marker logging
    session.py              # orchestrator wiring the above together
    cli.py                   # `vizaudit-overlay` entrypoint
export/                     # Phase 2: batch-iterates a LeRobotDataset, adapts frames to canonical
                            # format, writes the sidecar artifact (not yet created)
scripts/
  vizaudit_record.py        # monkeypatches rerun.init to inject a shared recording_id, then
                            # calls lerobot-record's own main(); runs in the `lerobot`
                            # env, zero dependency on the vizaudit package — see Key decisions
```

`src/` layout deliberately mirrors `lerobot`'s own `src/lerobot/` convention.

- **Frame contract:** canonical input to `core/` is always `np.ndarray`, HWC, `uint8`, RGB.
  Format adapters live at the edges (`overlay/`, `export/`) — never inside `core/`. Phase 1 v1
  doesn't touch frame pixels at all (directive-only), so this contract isn't exercised yet —
  it becomes load-bearing once `core/` and the deferred v2 auto-detect feature land.
- **No robot-specific code in `core/` or `overlay/`.** If a function needs to know about SO-101
  joints/URDF/kinematics/reach, it doesn't belong here — that's the line that keeps this tool
  robot-agnostic. `overlay/`'s patterns are pixel-space only for exactly this reason.

## Next steps

See `research_report.md` §6 for the Phase 2 open-questions list, and
`/home/bogdan/.claude/plans/rustling-riding-meadow.md` for the full Phase 1 implementation plan
(repo layout, config schema, module breakdown, verification tiers, open risks). In rough order:

1. ✅ Done — Phase 1 implemented per that plan: `pyproject.toml`, `overlay/` modules,
   `scripts/vizaudit_record.py`, tests, example config.
2. ✅ Done — Tier 1 no-hardware smoke tests (plan's verification section): unit tests pass;
   `examples/fake_lerobot_session.py` + `vizaudit-overlay` run end-to-end against a fake
   dataset; `EpisodeBoundaryWatcher` fires at the right cadence (confirmed via log output);
   the shared-`recording_id` mechanism confirmed merging two independent processes by
   inspecting a saved `.rrd` with `rerun rrd print` (`StoreId` matches across both
   `application_id`s) — this is also how the authkey-pin bug above was caught. Visual
   confirmation in an actual Rerun viewer is still blocked on this machine by a
   GPU/WebGPU-in-remote-display rendering problem, unrelated to vizaudit's logic.
3. Tier 2/3: validate on real `lerobot-record` and SO-101 hardware once a working viewer is
   available. Then verify `minAreaRect` reliability on real `top` camera footage from
   `so101_cube_pick_place_50`, to start on the deferred v2 auto-detect addition and/or
   Phase 2's `core/`.
4. Design the Phase 2 sidecar artifact schema before writing the lerobot-dataset-visualizer
   panel component.
