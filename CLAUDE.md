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
  vision-only/no-FK decision above — no per-robot reach data is ever read or needed. The one
  exception is `sector` under `surface_calibration` (next bullet), which still never reads
  robot-specific data — only 4 operator-marked image points.
- **`sector` is a filled-area pattern shape, sampled area-uniformly, not just on a boundary
  curve like `arc`.** `arc` only ever places points *on* a circle's circumference (a 1D
  curve); real spatial coverage needs points distributed *throughout* a 2D region. `sector`
  (a pie-slice/annular-sector) samples `r = sqrt(uniform(inner_radius², outer_radius²))`,
  `theta = uniform(angle_start_deg, angle_end_deg)` — naively uniform-in-`r` would clump
  points near the center, the squared-uniform form is the standard fix for area-uniform polar
  sampling. Uses a seeded `random.Random` (stdlib, not numpy) for deterministic, testable
  output. `arc`/`line` are untouched and still boundary-only — sufficient for setups that
  want explicit, deterministic placement rather than statistical coverage.
- **`exclude_zones` keep generated points off immovable scene obstacles (robot base, a fixed
  cup) — circles only, v1.** Scene-level (not per-object, since obstacles are a property of
  the camera framing, shared by every object's pattern). Two different enforcement
  strategies, because `sector` and `arc`/`line` have fundamentally different degrees of
  freedom: `sector` is randomized, so a point landing in a zone is just rejected and
  resampled (capped at `count * 200` attempts total, then a clear `ValueError` — likely the
  zones are too large/numerous for the available area). `arc`/`line` are fully deterministic
  given their parameters — there's no alternate point to substitute — so a conflict there is
  a **config error**: raise immediately, naming the offending point, and let the operator
  adjust the pattern or the zone.
- **A camera that isn't perfectly top-down breaks naive pixel-space area-uniform sampling —
  fixed via an optional 4-point homography (`surface_calibration`), applied to `sector` only.**
  A circle defined directly in pixel space is only a true circle in the real workspace if the
  camera looks straight down; at any other angle the same pixel-space circle is foreshortened,
  so "uniform in pixel space" silently isn't "uniform in real space," and a pixel-space
  semicircle won't visually line up with the robot's true reach boundary as an angled camera
  sees it. Fix: the operator marks 4 pixel points forming a rectangle on the real workspace
  surface (via the calibration tool below); `overlay/perspective.py`'s `compute_homography()`
  solves the standard 4-point projective transform (pure `numpy` linear algebra, no `opencv`)
  from those corners to a canonical rectangle sized to match the corners' own average
  edge-length ratio (so canonical numbers numerically resemble pixel numbers for the common
  near-top-down case, not an arbitrary unit square). When `surface_calibration` is set,
  `sector`'s `center`/`inner_radius`/`radius` are interpreted in that canonical space — area-
  uniform sampling is only mathematically valid where circles are actually circles — and each
  sampled point is mapped through the homography to pixel space before `exclude_zones` are
  checked (obstacles are always a pixel-space fact, regardless of which space we sampled in).
  Scoped to `sector` only: `arc`/`line` are deterministic curves where perspective-correct
  *statistical* coverage doesn't apply the same way; revisit if their visual mismatch under a
  tilted camera matters in practice.
- **The calibration tool connects directly to a camera in the browser — it has no dependency
  on a dataset, a recording session, or Rerun at all.** Calibration has to be possible
  *before* anything has been recorded, so making it read a frame off an existing dataset
  (an earlier version of this tool did exactly that) is backwards — it forces the operator to
  start a recording just to calibrate for it. `static/calibrate.html` is a fully standalone
  page using the browser's own `getUserMedia()` camera API to show a live feed and let the
  operator pick coordinates directly on it, with a live YAML-ready snippet;
  `vizaudit-calibrate` just copies that one static file to `--output` (no dataset/camera-key
  args, no Pillow dependency — nothing in this module touches pixel data anymore). It's also
  not a Rerun-based tool: this environment has a known GPU/WebGPU-in-remote-display rendering
  problem (hit during this same Phase 1's own viewer verification — black/blank screens from
  the native viewer and the `--serve-web` browser path alike), and plain HTML5
  `<video>`/Canvas2D is a completely different rendering path with no GPU dependency (this
  environment's browser already renders ordinary 2D content fine — it was specifically
  Rerun's WebGPU canvas that failed). Calibrating, editing the config, and running the overlay
  are three fully decoupled steps: open the page and pick coordinates, hand-edit and save a
  YAML file, then `vizaudit-overlay --config <path>` reads whatever file you point it at —
  no step depends on another being "live."
- **`vizaudit-calibrate` serves the page on localhost and auto-opens the system browser by
  default, rather than just writing a file the operator opens themselves.** Hit in practice:
  opening the written file directly (`file://`, or via an editor's HTML preview pane) got
  "permission denied" from `getUserMedia()` — `file://` pages aren't a guaranteed secure
  context for camera access in every browser, and an editor's built-in preview pane typically
  blocks camera access outright regardless of origin. `serve_and_open()` (stdlib
  `http.server.HTTPServer` bound to `127.0.0.1` on an OS-assigned free port, plus
  `webbrowser.open()`) fixes both at once: `http://127.0.0.1` is universally treated as a
  secure context, and `webbrowser.open()` launches the real system browser, sidestepping any
  preview-pane limitation. The page is still fully static (no further server contact needed
  after the initial load) — serving it is purely to satisfy this one browser permission
  requirement. `--no-serve` keeps the old "just write the file" behavior for scripting/cases
  where the operator wants to serve it themselves.
- **The calibration page is a dual-canvas tool (camera view + orthographic view), not a
  single-canvas "click points, pair as circles" picker — and its dark/cyan visual style is
  lifted directly from `lerobot-dataset-visualizer`'s actual design tokens** (read from
  `/home/bogdan/lerobot/lerobot-dataset-visualizer/src/app/globals.css`:
  `--bg: #0a0e17`, layered surface panels, single cyan accent `#38bdf8`, slate text scale,
  6px radii — not guessed, copied from the real file so it actually matches the ecosystem's
  other tools). The 4 corner-drag handles for `surface_calibration` now render directly on
  the **live** camera feed as a semi-transparent quad with a perspective-warped grid inside it
  (an NxM canonical grid mapped through the live-recomputed homography on every corner drag) —
  this makes misalignment with the real surface immediately, visually obvious, instead of
  requiring the operator to interpret raw numbers. A second, independent **orthographic**
  canvas shows the same surface undistorted. The reason for two canvases: fitting a circle to
  a handful of "move the arm to its reach limit and click where it is" points only makes
  geometric sense in *undistorted* space (the same reason `sector` sampling itself happens in
  canonical space, not pixel space) — so those points are clicked on the live camera view
  (the operator needs to see the real arm while physically moving it), mapped through the
  *inverse* homography into canonical space, fit to a circle there (Kåsa least-squares — one
  formula handles both the exactly-3-points case and more), and that fitted circle can be
  manually fine-tuned (drag center/edge) on the orthographic canvas, with every edit
  immediately reprojected back onto the camera view so the operator can sanity-check against
  the real image and iterate. A "steps" field drives a live sample-point preview (exactly that
  many points, in both views) using the same area-uniform rejection-sampling approach as
  `generate_sector_points`, specialized to a full disk. (A polygon cut-tool and shape
  primitives were initially deferred here and added in a later pass — see below; the surface
  itself still stays a 4-corner quad, so the homography keeps working, and the fitted reach
  shape stays a circle — freeform/arbitrary-vertex *surface* polygons remain a possible
  future follow-up, not yet needed.)
  **All of this math (homography compute/invert/apply, canonical-rect sizing, circle fit,
  seeded disk sampling) is reimplemented in JavaScript inside `calibrate.html`, not shared
  with `perspective.py`/`pattern.py`** — calibration is deliberately backend-free (previous
  bullet), so there is no Python process to call into while calibrating. This is an inherent
  duplication, the same category as the existing `SHARED_RECORDING_ID` duplication between
  two Python files: verified correct independently (not just "looks right") by porting the
  same test assertions from `test_perspective.py` into a standalone Node script and checking
  them against the JS implementation directly, rather than assuming the port was faithful.
- **The fitted circle's valid sampling region is the intersection of the circle and the
  workspace rectangle, not the circle alone — the circle can legitimately be larger than the
  marked workspace (the operator's marked surface and the robot's actual reach aren't
  necessarily the same size), but every generated point still has to land somewhere the
  camera actually shows.** `sampleDiskPoints` takes an optional `bounds` (the canonical
  rectangle's `[width, height]`) and rejects-and-resamples any candidate outside it, the same
  rejection-sampling idiom `generate_sector_points` already uses for `exclude_zones`. The
  camera view's projected circle outline is now clipped to the workspace quad (`ctx.clip()`)
  so it visually stops at the quad's edge instead of appearing to expand past it — otherwise
  the outline alone (unclipped, just a projected stroke) made it look like the circle was
  growing the workspace rather than overlapping it.
- **Two calibration-page bugs found by actually using it, not just reading the code:** (1)
  `#cameraCanvas`'s CSS inherited the shared `background: var(--surface-0)` (opaque) from the
  `video, #cameraCanvas, #orthoCanvas` rule it was grouped under for the other shared
  properties — since this canvas is an absolutely-positioned *transparent* overlay on top of
  the live `<video>`, an opaque background on it paints over and hides the feed entirely
  except where something was explicitly drawn. Fixed with an explicit `background:
  transparent` override (CSS specificity already favors the `#cameraCanvas` ID rule). (2)
  Camera permission was only ever requested when the operator clicked "Start camera" — now
  also requested proactively as soon as the page's script runs (i.e. on load), with the
  button left in place as a manual retry (e.g. after the operator grants permission in the
  browser's own UI following an initial denial, or switches/plugs in a camera).
- **`sector`'s default distribution is `"grid"` (a deterministic, evenly-spaced lattice), not
  `"random"` — a random scatter doesn't make sense as the default for spatial-coverage
  auditing, since you want predictable, legible coverage, not statistical luck.** This is a
  config-level default only (`config.py`'s `_parse_pattern` applies `data.get("distribution",
  "grid")`): `generate_sector_points()`'s own bare-function default stays `"random"`, so
  every existing direct caller/test is unaffected. See the `grid`/`radial`/`random` bullet
  below for what `"grid"` actually does today.
- **The calibration tool gained a polygon/primitive cut-tool and a "uniform grid vs random"
  distribution dropdown — both needed real engine support, not just a prettier preview.**
  `ExcludeZoneConfig` gained `shape: "circle" | "polygon"` (default `"circle"`, so every
  existing config without a `shape` key is unaffected) and `vertices` for the polygon case;
  `pattern.py` gained `_point_in_polygon` (ray casting) and `_point_near_polygon` (the same
  shape, buffered outward by `border_width`); both feed `_point_in_any_zone`, shared by the
  deterministic arc/line validate-and-error path (always called with `border_width=0`, since
  arc/line have no `border_width` field at all) and the sector rejection-sampling path (where
  the buffering is real, but only when there's no homography — see the bullet above). In the
  calibration page: "Cut:
  rectangle"/"Cut: circle" are primitives (click-drag, immediately converted to a polygon --
  a circle becomes a 24-gon -- so there's only one internal cut representation), "Cut:
  polygon" is freeform (click to add vertices, a "Finish cut polygon" button closes it), and
  "Edit cuts" drags any existing cut's vertices. Cuts are drawn on the **camera** view (like
  extreme-reach points: the operator needs to see the real obstacle, e.g. the robot base, to
  trace it) and exported as pixel-space `exclude_zones` polygons directly — no canonical-space
  conversion needed for export, since `exclude_zones` are already supposed to be pixel-space.
  `canonical_rect_dims` (formerly private, `_canonical_rect_dims`) is now exported from
  `perspective.py` so `session.py` can compute the same `bounds` the calibration preview uses
  and pass it into `build_pattern`/`generate_sector_points` for the real run — without this,
  the real session and the calibration preview would disagree about whether a circle bigger
  than the workspace gets clipped.
- **Camera feed and orthographic canvas now resize to fit their panel, instead of being
  capped at a fixed pixel size regardless of available space.** Removed the hardcoded
  `max-width`; both panels use flexible widths and the orthographic canvas's drawing-buffer
  size is recomputed from its container's actual rendered width (plus a viewport-relative max
  height) on every render and on window resize, rather than fixed `480x360` JS constants.
- **`"grid"` and `"radial"` are two separate, real distributions — `"grid"` is a true
  Cartesian lattice, `"radial"` is the Fermat/Vogel spiral ("sunflower seed" arrangement).**
  An earlier version conflated the two: the UI/config called it `"grid"` but it was secretly
  always the spiral underneath, which produced a "weird radial-looking" result that got
  *worse*, not better, as the sample count grew, and didn't actually cover the full disk —
  exactly backwards from what a literal "grid" should do (bigger count → smaller, denser
  cells, still spanning the whole area). Now: `"grid"` is a regular lattice with column/row
  counts scaling with `sqrt(count)` (so cell spacing shrinks as `count` grows, the
  "covers all the spots" guarantee), no randomness, `seed` unused (see below for exactly how
  `"grid"`'s positions are now computed — this bullet predates that redesign but the
  "two distinct distributions" decision itself hasn't changed). `"radial"` keeps the
  spiral formula (`r_i = sqrt(inner² + (outer²-inner²)*(i+0.5)/N)`,
  `theta_i = offset + i * 137.50776...°`, the golden angle — irrational relative to a full
  turn, so no two points ever align radially or angularly, at any `N`) as a *visually
  distinct alternative* to `"grid"`, not a hidden implementation of it; `seed` draws one
  random rotation offset for the whole spiral. Implemented identically in `pattern.py` and
  the calibration page's JS port (`sampleSectorPoints`).
- **`"grid"`/`"radial"` no longer trim a grown candidate pool down to `count` at all — each
  computes exactly `count` *ideal* positions from a closed-form formula and individually
  relocates any invalid one, instead of over-generating and discarding.** Two earlier fix
  attempts at "missing points" both still trimmed a *population*: first a plain even stride
  (`items[(k*n)//count]`), then a per-group proportional stride
  (`_allocate_shares`/`_even_stride_select_grouped`/`_chunk`, largest-remainder apportionment
  per lattice column/spiral chunk before striding within it). Both helped, but a real-world
  report at the tool's actual operating range — tens of points, not thousands — showed both
  were still genuinely wrong, not just imprecise: trimming a population necessarily thins
  some regions more than others whenever rejection itself is uneven across the region (a
  lattice column near the circle's edge has fewer valid candidates to begin with than one
  through the center), and at low `count` that thinning's integer rounding is a *large*
  relative error — confirmed by literally dumping per-column point counts and comparing
  against the un-trimmed pool, not by eyeballing a render. The fix that actually held up:
  stop trimming. `"grid"`/`"radial"` each compute `count` ideal positions directly (a lattice
  cell center, or a spiral point — see below) and pass each one through
  `_relocate_if_invalid`, which returns it unchanged if already valid, else tries a
  closed-form `_clamp_to_region` (radius into `[inner,outer]`, angle into the pie slice,
  then — if still outside `bounds` — shrunk toward `center` *along the same angle* rather
  than an independent x/y clamp, which would collapse every point whose ideal x exceeds the
  bound onto the exact same vertical line regardless of its own y, found as a real bug:
  multiple lattice columns landed on literally identical output points), and only falls back
  to a local random search (seeded, growing radius) for the one case `_clamp_to_region` can't
  project in closed form: an arbitrary `exclude_zones` polygon. `_occupancy_guard` additionally
  rejects a relocation landing on a point an earlier index already placed — the residual case
  even the ray-shrink clamp doesn't fully eliminate (any two points that happen to share an
  exact angle from `center`, e.g. two lattice columns on the row through `center`'s own y,
  shrink to the literal same nearest boundary point).
- **`"grid"`'s common case (a full disk — always true for the calibration tool, and the
  typical YAML case) skips per-point relocation almost entirely via a per-column continuous
  fast path, which is what actually fixed the worst of the unevenness, not the relocation
  fallback alone.** Per-point relocation fixes *correctness* (exact count, no collisions) but
  doesn't by itself fix *evenness* — a shared global row grid across every column (what the
  fast path replaced) still meant short edge columns and long center columns drew from the
  same fixed row positions, so relocating individually-invalid cells still left visibly
  uneven density once `bounds` clipped one side hard. The fast path instead computes each
  lattice column's valid y-*range* directly from the circle's chord at that x (intersected
  with `bounds`) — pure geometry, not candidate filtering — allocates `count` across columns
  proportionally to each column's own range length (`_allocate_shares`, reused here for a
  *continuous* proportional split rather than thinning a discrete candidate list, which is
  what made the two earlier fix attempts above still uneven), then places that many points
  evenly spaced *within* the continuous range. No discrete candidates are ever generated or
  discarded, so there's nothing to thin unevenly, and no two columns can ever land on the
  same position. An annulus or restricted pie slice (never produced by the calibration tool)
  falls back to the near-square-lattice-plus-relocation path instead, trading a small residual
  collision risk for not needing an exact per-column annulus/sector chord formula. Verified
  via nearest-neighbor distance ratios (not just count/bounds): under heavy asymmetric
  `bounds` clipping at `count=50`, the fast path keeps every point's nearest neighbor within
  ~90% of the average spacing, versus the previous design's neighbors landing as close as
  1-7% of the average spacing apart (effectively indistinguishable from a missing point).
  `"radial"`'s spiral has no per-column structure to exploit, but also doesn't need it: the
  golden-angle increment never puts two different indices on the same angle from `center`, so
  per-point relocation alone (no fast path) already keeps it collision-free in practice.
  Implemented identically in `pattern.py` (`_clamp_to_region`/`_occupancy_guard`/
  `_relocate_if_invalid`/`_allocate_shares`) and the calibration page's JS port
  (`clampToRegion`/`occupancyGuard`/`relocateIfInvalid`/`allocateShares`).
- **`"grid"`'s fast path still had a real, separate bug even after the rewrite above: column
  (x) spacing came out much wider than row (y) spacing whenever `bounds` clipped a
  meaningful fraction of the circle away.** Root cause: `cols` columns were spaced over the
  *full circle diameter* (`step_x = 2*effective_outer/cols`), then whichever columns landed
  outside `bounds` were simply skipped -- the survivors stayed at their original
  full-diameter spacing, while every dropped column's share of `count` piled into the
  survivors' own row spacing instead, stretching x-spacing far past y-spacing (measured
  directly, not eyeballed: x-gaps came out ~70% wider than y-gaps for a circle clipped to
  70% of its width). Fix: compute the actual available `[x_min, x_max] x [y_min, y_max]`
  bounding box first (circle's own extent intersected with `bounds`), choose `cols` from
  *that* box's aspect ratio (`cols = round(sqrt(count * width/height))`, generalizing the
  earlier bare `sqrt(count)` which implicitly assumed a square region), and space columns
  over the actual box, not the full diameter. Verified by comparing x-gap to average y-gap
  directly post-fix (within 2x of each other across a battery of aspect ratios, including a
  deliberately extreme near-1D strip), not just re-running the existing nearest-neighbor
  check (which doesn't distinguish "evenly spread in 2D" from "evenly spread along a stretched
  1D axis").
- **Points relocated off an `exclude_zones` cut near `center` (e.g. marking out the robot's
  own base) clumped at unstructured angles relative to each other -- fixed by having
  `_relocate_if_invalid` try a radial search along the point's own angle before falling back
  to random jitter, and by making `_occupancy_guard` reject landings that are merely *too
  close*, not just exactly identical.** Previously, the only fallback for an
  `exclude_zones` violation (`_clamp_to_region` doesn't know about cuts -- no simple
  closed-form projection exists for an arbitrary polygon) was an unstructured local random
  search. For a handful of points all needing relocation off the *same* central obstacle
  (common case: cutting out the area around the robot base, which sits at radius 0 where the
  golden-angle spiral's points are already naturally close together), independent random
  jitter has no reason to spread them apart from EACH OTHER, only from the obstacle --
  confirmed by measuring actual angular gaps between relocated points: as tight as 0.2-3°
  next to gaps of 50+°, instead of anything resembling the spiral/lattice's own spacing.
  Fix, in order: (1) `_relocate_if_invalid` now tries moving outward (then inward) along the
  point's *own* angle from `center` first -- cheap, deterministic, and naturally keeps
  multiple points fanned out at their original angles, since a small obstacle relative to
  the whole disk is usually cleared this way; (2) `_occupancy_guard` changed from an exact
  (epsilon) coincidence check to a real `min_separation` distance check (still O(N) per call,
  trivial at this tool's scale) -- needed because even the radial search alone can still let
  two *different* points researched from slightly different starting radii land closer
  together than is visually acceptable, not just literally on top of each other; both grid
  and radial pass `0.5 * expected_spacing` as that minimum. Verified via nearest-neighbor
  distance around a real central cutout (not visually): both distributions now stay above
  30-65% of the average spacing even right next to an obstacle covering 16-36% of the disk's
  radius, instead of the near-zero (effectively duplicate-looking) distances measured before
  the fix. Implemented identically in `pattern.py` and the calibration page's JS port.
- **`"radial"`'s radius formula used the RAW `inner_radius`/`outer_radius` instead of
  `effective_inner`/`effective_outer` (border_width already applied) -- a real, reported bug
  independent of `exclude_zones` entirely: a `border_width` of 2 on a radius-10 disk collapsed
  45% of 60 points onto the exact `effective_outer` boundary ring, because every point whose
  ideal radius (computed over the FULL, unshrunk disk) exceeded the actual boundary got
  ray-shrunk there by `_clamp_to_region`.** Found by testing `border_width` alone, with no
  `exclude_zones` at all, after a report that "the issue is present in both grid and radial...
  same when I change the border size" -- confirming border_width itself, not just zones, could
  trigger the same boundary-pileup pattern. Fix: use `effective_inner`/`effective_outer` in the
  radius formula, one line. `"grid"`'s rarer general (annulus/restricted-angle) fallback path
  had the identical bug (its initial bounding-square lattice was sized from the raw
  `outer_radius`) and got the same fix; `"grid"`'s common full-disk fast path was already
  correct (it always used `effective_outer`). The calibration page's JS port is disk-only and
  has no `effective_inner` concept at all (border_width never carves out a hole at the exact
  center there -- a deliberate, narrower scope than `pattern.py`, not ported), so only its
  outer-radius half of this fix applies: `r = sqrt(t) * effectiveOuter`, not the raw
  `outerRadius`.
- **A central (or any) `exclude_zones` cut no longer just relocates individual out-of-zone
  points -- `"grid"`'s per-column ranges and `"radial"`'s radius distribution now both
  subtract the zone's TRUE blocked area before deciding how many points a region gets, so the
  whole pattern's density adapts to the area actually lost instead of clumping the displaced
  points into a dense ring right at the zone's edge.** Found from a follow-up report, after
  the previous bullet's radial-search-along-own-angle fix: "the radial one got worse overall...
  when I insert a square at the center I want all of them to stay uniform by changing grid
  size." Root cause: relocating each point individually is correct FOR THAT POINT, but wrong in
  aggregate -- every point whose *unobstructed* ideal position fell inside a central zone
  relocated outward along its own angle to roughly the *same* radius (just past the zone's
  edge), so the zone's entire "lost" share of `count` reappeared as a dense ring there instead
  of being absorbed as very slightly tighter spacing across the whole remaining disk (measured:
  13 of 60 points within 0.6 units of a radius-4 zone's boundary, instead of the ~4 a uniform
  fill would put there). Two distinct fixes, matching each distribution's own structure: (1)
  `"grid"`'s full-disk fast path already gives each lattice column its own continuous y-range;
  each column now ALSO subtracts whatever `exclude_zones` cross it (`_excluded_y_intervals_at_x`
  -- circles via an exact quadratic chord formula, polygons via `_polygon_vertical_intervals`,
  a standard scanline edge-crossing clip, after `_inflate_polygon` approximates the
  `border_width` buffer by pushing vertices outward from the centroid), via the new
  general-purpose `_merge_intervals`/`_subtract_intervals` interval algebra -- a column
  straddling the zone splits into two sub-intervals, each getting its OWN `_allocate_shares`
  share of `count` proportional to its own (now correctly smaller) length. (2) `"radial"` has
  no column structure, so instead its radius itself is drawn from a numerically-integrated
  *available-area* CDF (`_radial_available_area_cdf`/`_invert_radial_cdf`, ~300 radius samples)
  whenever `exclude_zones` is non-empty, rather than the unobstructed closed form -- the CDF
  integrates, at each radius, the angular span NOT blocked by any zone
  (`_circle_zone_angle_block`, an exact closed-form "angular extent of a circle as seen at a
  given radius" formula via the law of cosines; `_polygon_angle_block`, which finds every
  edge-circle crossing via `_segment_circle_intersection_angles` and tests each arc's midpoint
  with the existing `_point_in_polygon` rather than reasoning about polygon winding directly),
  merged across zones with wraparound-aware `_merge_angle_intervals` (duplicates each interval
  at -360/+360 offsets, merges on that extended line, clips back to `[0,360)` -- the standard
  fix for an arc that straddles the 0° reference). A radius band mostly inside the zone is thus
  assigned proportionally fewer points UP FRONT; each computed point still goes through the
  existing `_relocate_if_invalid` as a safety net (e.g. an off-center zone blocks only part of a
  given radius's angular span, which the CDF -- a function of radius alone -- can't capture),
  but rarely needs it and rarely has to move far. Both paths only run this extra computation
  when `exclude_zones` is non-empty; with none, the exact pre-existing closed forms are
  unchanged (verified via the existing 113 tests, all still passing byte-for-byte). Verified
  via the same radius-band-histogram method that found the bug (not nearest-neighbor distance
  alone, which didn't clearly distinguish "evenly spread" from "one dense ring plus one sparse
  gap" in earlier checks): the boundary-ring count for a radius-4 central zone in a 60-point
  pattern dropped from 13 to 4 for `"radial"` (proportional expectation is ~4) and grid's
  nearest-neighbor ratio improved from 0.65 to 0.86. The calibration page's JS port mirrors all
  of this exactly, with one structural simplification it can take advantage of: cuts are
  always stored canonical-native and a circle cut is already approximated as a fixed 32-gon by
  `cutShapeBoundary` before anything else sees it (see the "Cut shapes are stored in CANONICAL
  space" bullet below), so the JS port needs only the polygon code paths
  (`polygonVerticalIntervals`/`polygonAngleBlock`) -- no separate circle closed form, and no
  homography branching, unlike `pattern.py`.
- **The exclude_zones-aware radial CDF fix above was incomplete: it never accounted for
  `bounds` (the workspace rectangle), which is this tool's primary, near-universal scenario
  -- not an edge case -- since the whole point of `surface_calibration`/`bounds` is that the
  fitted reach circle commonly extends beyond the marked workspace.** Found from a direct
  follow-up report after that fix shipped: "the grid works flawlessly but the radial one is
  just as broken as with the previous changes." Root cause, found by testing `bounds` alone
  with zero `exclude_zones`: a workspace rectangle clipping roughly a third of the circle away
  gave radial a 0.37 nearest-neighbor ratio -- unchanged from before the CDF fix, since that
  fix's gating condition (`if exclude_zones:`) never even ran the CDF when only `bounds` was
  restrictive. Extending the CDF to also integrate `_outside_bounds_angle_intervals` (computed
  by reusing `_polygon_angle_block` against the inset rectangle treated as a "zone" -- giving
  the *inside* intervals -- then taking the complement) raised the ratio only to ~0.41, which
  led to the second, deeper part of this bug: **`_clamp_to_region`'s bounds-shrink step moves
  a point along its own ray *toward `center`* (i.e. reduces its radius) whenever its angle
  lands outside `bounds` -- which silently throws away the CDF's carefully-chosen radius for
  every such point**, since `_relocate_if_invalid` tries `clamp` before anything else. The
  CDF can correctly tell a radius shell "only 60% of your angles are open, so take 60% of the
  unobstructed point share," but if the points sent to that shell still use the blind
  golden-angle formula for their angle, many land in the closed 40% and get yanked back to
  whatever (much smaller) radius the bounds rectangle allows at that exact angle --
  regenerating the same "ring of points piled up near the boundary" failure this whole fix
  exists to prevent, just relocated to wherever the rectangle happens to be narrowest. Fix:
  factored the per-radius blocked-angle computation out into
  `_available_angle_intervals_at_radius` (used by both the CDF, which only needs the summed
  length, and the per-point loop), and added `_place_in_available_intervals`, which maps the
  point's golden-angle *fractional position* (not the raw angle) onto whatever sub-intervals
  are actually open at its own assigned radius -- the same idea as the grid fast path placing
  a point within a column's available sub-interval, just in angle instead of y. This makes
  each point valid by construction in the common case rather than relying on `clamp`/jitter to
  fix a blind angle choice -- confirmed by instrumenting `_relocate_if_invalid`'s call count
  directly (not just the final metric): 0 of 60 points needed any relocation after this fix,
  down from a meaningful fraction before it. Final measured nearest-neighbor ratio for the
  same bounds-clipping scenario: 0.76-0.83 across several seeds -- a large improvement over
  the 0.37-0.41 before, though still somewhat below grid's ~0.97 under identical clipping;
  the residual gap appears to be an inherent property of mapping a continuous golden-angle
  sequence through a smoothly-varying (not column-discretized) available-angle function,
  rather than a remaining defect -- a quarter-disk-only clipping test (more severe than the
  one above) actually scored *higher* (0.975), suggesting the 0.76 case's exact geometry,
  not a systemic issue, sets the lower end of the range. Implemented identically in
  `pattern.py` (`_outside_bounds_angle_intervals`/`_available_angle_intervals_at_radius`/
  `_place_in_available_intervals`) and the calibration page's JS port
  (`outsideBoundsAngleIntervals`/`availableAngleIntervalsAtRadius`/
  `placeInAvailableIntervals`, gated on `cutPolygonsCanonical.length > 0 || !!canonicalSize`
  rather than `exclude_zones` alone).
- **`"radial"`'s restricted-angle-span case (e.g. a semicircle pattern) folded the
  full-circle golden-angle sequence into the narrower span via plain `% angle_span` --
  this breaks the golden ratio's special low-discrepancy property, since that property is
  specifically tuned for a 360-degree step, not whatever ratio `(360 * conjugate) %
  angle_span` happens to produce for an arbitrary span.** Found from a direct report,
  independent of any `exclude_zones`/`bounds` interaction: a 60-point semicircle (flat
  "diameter" edge at one side, curved arc on the other) showed visibly sparse coverage right
  at that flat edge -- "sparse approaching the upper diameter line" -- with a measured
  nearest-neighbor ratio of only 0.80, well below the ~0.92 a full circle gets with the same
  count and seed. Root cause: golden angle's magic (no two points ever share an angle, at
  any `N`) comes from `1 - 1/phi` being a particularly badly-approximable irrational number
  *relative to a full turn*; reducing the resulting 360-degree step modulo an arbitrary
  narrower span produces a *different* number (`step / angle_span`) with no guarantee of
  being similarly hard to approximate by simple fractions, so the folded sequence can clump
  unevenly, worst right at the boundary the fold wraps around. Fix: compute the golden-angle
  step directly from `angle_span * _GOLDEN_RATIO_CONJUGATE` (the same `1 - 1/phi` ratio,
  scaled to whatever span is actually in use) instead of computing the full-360 step and
  folding it down afterward -- exact no-op for the common `angle_span == 360` case (the
  scaled step is numerically identical to the old hardcoded constant), confirmed via a
  regression test asserting that equivalence directly, not just relying on it being true by
  inspection. Raised the semicircle's nearest-neighbor ratio from 0.80 to 0.97. This bug
  predates today's `exclude_zones`/`bounds` work entirely (it's purely a function of
  `angle_start_deg`/`angle_end_deg`) and was found only because a user's actual pattern used
  a restricted span, not the calibration tool's always-360-degree preview -- `calibrate.html`
  has no `angle_start`/`angle_end` concept at all and was therefore never affected, so this
  fix has no JS counterpart.
- **An off-center (not centered) `exclude_zones` cut measurably degraded `"radial"`'s
  evenness (nearest-neighbor ratio ~0.5-0.7 across seeds, vs. ~0.92 unobstructed) -- this was
  initially investigated and left as a documented, deferred limitation, but a follow-up
  report ("border thickness... gives less consistent results... make sure it's resized
  correctly") showed the SAME root cause also hit the much more common `border_width`
  +`bounds` combination, with results swinging non-monotonically (0.57-0.89) across a
  `border_width` sweep on an otherwise-unchanged shape -- common enough, and severe enough,
  to revisit and fix rather than leave deferred.** Root cause (confirmed via
  instrumentation): mapping each point's golden-angle-derived fraction into the available
  angular sub-interval at its OWN radius (`_place_in_available_intervals`) correctly
  preserves *aggregate* density per radius shell (via the CDF) but not golden angle's
  point-to-point separation guarantee -- that guarantee belongs to the raw, unmapped
  `i * golden_step` sequence specifically, and two different points, with different raw
  fractions, can still coincidentally map to nearly the same absolute angle once their
  available-interval shapes (which vary continuously with radius near any asymmetric
  restriction) happen to align that way. Neither `_relocate_if_invalid` (never fires --
  every point is valid on the first try) nor tightening `_occupancy_guard`'s threshold
  (the closest pairs are roughly as far apart as the realistic *average* spacing, not a
  literal collision) catches this, because it isn't a validity violation, just a local
  *evenness* shortfall.

  Fix: `_refine_radial_local_separation`, a bounded post-process pass (only run when
  `exclude_zones`/`bounds` actually restrict the disk) that, for up to 3 sweeps, finds any
  point closer than `0.75x` the placed set's OWN average nearest-neighbor distance to its
  nearest neighbor and searches 24 candidate angles across the available interval at that
  point's *own, unchanged* radius (never touching radius, which the CDF already got right
  in aggregate) for whichever maximizes its distance to every other current point. The
  threshold deliberately uses the placement's own empirical average, not the theoretical
  `expected_spacing` (the tightest-possible-packing estimate) -- an earlier version of this
  fix used `expected_spacing` directly and silently never fired, since realistic
  nearest-neighbor distances are typically well above that theoretical floor even when
  noticeably tighter than the achieved average; found by checking the refinement actually
  ran before trusting the metric, not by assuming a plausible-looking implementation worked.
  Measured: the `border_width` sweep's nearest-neighbor ratio went from a non-monotonic
  0.57-0.89 to a consistent, near-monotonic 0.76-0.89; the off-center-zone-across-seeds case
  went from 0.49-0.68 to a consistent 0.75-0.77 -- both now close to grid's evenness under
  comparable restriction, without the architectural restructuring (coordinating placement
  across points sharing a radius neighborhood, the way grid's per-column placement already
  does) that the original, deferred write-up assumed would be required. Implemented
  identically in `pattern.py` (`_refine_radial_local_separation` +
  `_candidate_thetas_in_intervals`) and the calibration page's JS port
  (`refineRadialLocalSeparation` + `candidateThetasInIntervals`), gated the same way as the
  CDF itself.
- **The calibration page's cut-tool cursor (a knife icon shown while drawing a rectangle/
  circle/polygon exclusion zone) used to render the actual knife emoji via an SVG `<text>`
  element, with a hand-picked hotspot `(2, 26)` guessed to land on the blade's visual tip --
  a real, reported bug ("the cursor is offset" while drawing cut boxes): an emoji glyph's
  rendered shape, and therefore where its tip actually falls within the image, varies by
  font/platform/browser, so a hotspot tuned by eye on one system reliably looks wrong on
  another.** Fixed by replacing it with a plain vector wedge (`<polygon points="2,2 20,8
  8,20">`) drawn with explicit, known coordinates instead of relying on glyph metrics --
  the tip is exactly `(2, 2)` by construction, and the CSS cursor hotspot is set to that
  same point, so the visual tip and the actual click position are guaranteed to coincide
  regardless of platform.
- **The calibration tool rescales `fittedCircle` and every cut shape when the workspace
  corners move, instead of leaving their canonical numbers untouched.** Root cause of "moving
  the corners cuts the sampled circle": `canonicalSize` (and therefore the whole canonical
  coordinate system's scale) is *re-derived from the corners' own pixel distances* on every
  render (`canonicalRectDims`) — so even a small corner nudge to fix alignment changes what
  one canonical unit means in real terms, while `fittedCircle`/cuts hold absolute canonical
  numbers computed against the *old* scale. If the canonical rect shrinks, the same
  absolute-radius circle suddenly represents proportionally more of the workspace and pokes
  out further, getting clipped harder by the circle∩workspace intersection (the existing,
  intentional clipping rule) — looking like the circle randomly "got cut" from an unrelated
  edit. Fix: `renderAll()` tracks `lastCanonicalSize`; whenever the freshly recomputed
  `canonicalSize` differs from it, `rescaleCanonicalShapes(scaleX, scaleY)` rescales
  `fittedCircle.center`/every cut vertex by `(scaleX, scaleY)` component-wise and every
  circle's radius by the geometric mean `sqrt(scaleX*scaleY)` (keeps it a circle rather than
  distorting it into an ellipse under non-uniform scaling) — applied every render, so it
  also tracks smoothly frame-by-frame during a live corner drag, not just on release. This
  has no Python-side equivalent: `pattern.py`'s `homography`/`bounds` are computed once from
  a static config and never change mid-session, so this is a calibration-tool-only fix.
- **Cut shapes (the calibration tool's exclusion-zone editor) keep their own type —
  `circle`/`rectangle`/`polygon` — instead of all being flattened into a generic polygon on
  creation.** Flattening a circle primitive into a 24-gon meant "editing" it afterward meant
  dragging one of 24 nearly-coincident vertices, which barely changed its shape and was
  fiddly/pointless — exactly backwards from what a circle's 2 real degrees of freedom
  (center, radius) should feel like to edit. `getCutHandles(shape)` returns the *right*
  handles per type (circle: center + one radius handle; rectangle/polygon: one handle per
  vertex), so editing matches the shape's actual structure.
- **Cut shapes are stored in CANONICAL space (`cutShapesCanonical`), exactly like
  `fittedCircle` already was — not pixel space, which is what an earlier version did.** That
  earlier pixel-native storage was the root cause of two reported bugs: (1) cuts didn't
  visually "follow the surface" when a corner was dragged/twisted, because their drawn pixel
  position was a fixed number unrelated to the current homography, while the grid lines and
  circle inside the same quad *did* visually track it; (2) the orthographic view could show
  sample points landing inside a drawn cut, because the validity check's canonical-space
  approximation of the cut was a *separate* derivation (pixel shape → inverse-homography →
  polygon) from whatever was actually drawn, so the two could disagree. Making cuts
  canonical-native fixes both: `cutShapeBoundary(shape)` (circle → fixed-segment polygon,
  rectangle/polygon → vertices, all directly in canonical space, no homography involved) is
  now the single source of truth used identically by the validity check, the orthographic
  render, the camera-view render (forward-mapped through the *current* homography — this is
  what makes them visually follow a twisted corner), and the YAML export (also
  forward-mapped, since `exclude_zones` in the YAML are still pixel-space — see the
  `pattern.py` docstring). One consequence: a circle cut forward-mapped through a homography
  is generally an ellipse in pixel space, and `config.py`'s `exclude_zones` schema has no
  ellipse shape, so every cut — circle included — now exports as `shape: polygon` (the
  circle's forward-mapped boundary approximation) rather than `shape: circle`, to avoid a
  second, redundant, lossy circle-from-ellipse approximation on top of the first.
- **The fitted circle and every cut shape are now editable on *both* the camera view and the
  orthographic view, not just whichever view originally owned that capability.** Previously
  only the orthographic view could drag the fitted circle, and only the camera view could
  edit cuts. `circleHandles(circle)` is shared by `fittedCircle` and circle-type cut shapes;
  on the camera view, handle positions are forward-mapped through the homography for
  hit-testing/rendering, and a dragged pixel point is converted back to canonical via the
  inverse homography before being applied — on the orthographic view, handles are used
  directly in canonical space (just scaled by `orthoScale`), no homography involved at all.
- **`border_width` now genuinely applies to *every* boundary (circle, workspace, and every
  cut) — it previously only shrank the fitted circle/workspace bounds, silently doing nothing
  for cuts, which read as "border width only affects one thing" and isn't what the field
  promises.** Root cause: cuts were checked in pixel space while `border_width` is a
  canonical-space distance whenever a homography is active (always true in this tool) — not
  comparable units, so the original design deliberately skipped buffering cuts by it (see the
  `pattern.py` `is_valid` rationale a few bullets up). Fix: when a homography is active, *all*
  exclude_zones get mapped into canonical space once per render/sample (via
  `invert_homography`, new in `perspective.py` — a circle zone becomes a 32-gon
  approximation), and every validity check (circle radius, workspace bounds, and now cuts)
  happens in that one consistent space, with `border_width` applied uniformly throughout.
  This also happens to fix a second, related symptom ("the orthographic view shows samples
  inside a cut"): the canonical-space cut polygon used for the validity check and the one
  drawn in the orthographic view are now *the same object*, not two independently-derived
  approximations that could quietly disagree.
- **The calibration tool has ONE mutually-exclusive `tool` state (`"corners"` / `"extreme"` /
  `"rectangle"` / `"circle"` / `"polygon"` / `"editCuts"`) instead of a separate `mode`
  dropdown plus a `cutTool` icon group that had to be merged via priority in
  `activeInteraction()`.** An earlier pass kept "drag corners"/"mark extreme points" in a
  `mode` dropdown and cut-related actions in a separate `cutTool` icon group with its own
  `"none"` value, requiring `activeInteraction()` to decide which one won. Once "editing
  corners" *also* became an icon button (next bullet) there was no longer a reason for two
  separate state machines — every interaction mode is now a plain radio-button-style icon
  button via `setTool(newTool)`, and `"corners"` itself serves as the neutral default (no
  `"none"` value needed: clicking any tool button always selects something concrete).
  `activeInteraction()` is now just `return tool`. A freeform polygon cut still finishes on
  double-click or the checkmark button, and <kbd>Esc</kbd> still cancels an in-progress one
  at any time.
- **"Mark extreme points" stays a separate, plainly-labeled text button outside the "Edit"
  icon group — requested explicitly, to keep "what you click to start marking reach-limit
  points" visually distinct from "what you click to edit existing structure."** Every other
  interactive tool (drag corners/circle, the three cut shapes, edit cuts) lives in the "Edit"
  toolbar group as an icon button, including a dedicated lock-corners toggle (padlock 🔓/🔒)
  that disables the corner-drag hit-test in `cameraCanvas`'s mousedown handler — but not
  circle editing, which stays independent of corner state — so an operator can mark points
  or draw cuts near a corner without risking nudging the calibration by accident. Locked
  corners also render dimmed (`--text-faint` instead of `--accent`) as a visual cue. The
  "drag corners" tool's icon is a small inline SVG (a square outline with a filled dot at
  each corner, using `currentColor` so it inherits the button's hover/active color exactly
  like the emoji-glyph icon buttons) rather than a Unicode glyph — there's no good single
  character for "this represents the 4 draggable corner handles," so it's drawn directly.
- **The cut-tool toolbar is icon buttons (▭ ◯ ⬠ ✎ ✓), not dropdowns/text buttons, with a
  knife cursor over the camera view while a cut-drawing tool is active — the only text
  buttons left in the "Edit" group are "Reset corners" and "Clear all cuts."** Requested
  explicitly: a row of dropdowns and generic buttons didn't read as "this is the cut tool,"
  and there was no visual indication while drawing that you were in a destructive/exclusion
  mode rather than a normal click-to-place one. The knife cursor is a small inline-SVG data
  URI (the 🔪 emoji rendered as SVG `<text>`, percent-encoded for the CSS `url()`) with a
  `, crosshair` fallback for browsers that can't parse it; a separate `grab` cursor applies
  while "Edit cuts" is active, on both views. "Clear all cuts" stays a labeled text button
  because — per the request — it should be the one thing in this group an operator can't
  mistake for something reversible/exploratory; "Reset corners" stays text for the same
  reason (and gets `disabled` while corners are locked).
- **The calibration tool's "Save to file..." button saves to a fixed, predictable location
  by default (`vizaudit_calibration.yaml` in the current directory) instead of the browser's
  generic downloads folder — and `vizaudit-overlay --config` defaults to reading from that
  exact same path, so the two tools chain together with no flags at all.** This needed a real
  server round-trip, not just a browser API: `vizaudit-calibrate`'s already-running local
  `http.server` (started for the secure-context fix above) now also handles `POST /save` by
  writing the request body straight to `--save-to` (`calibrate.py`'s `_make_handler`); the
  page's save button does `fetch("/save", ...)` as the primary path, falling back to the
  native save-dialog/download approach only when there's no server to talk to (`--no-serve`).
  `DEFAULT_SAVE_PATH` is one constant shared between `calibrate.py` and `cli.py` so the two
  defaults can't drift apart.
- **The calibration tool's "drag corners" tool and "edit cuts" tool were merged into one
  "Edit" tool that drags every kind of EXISTING geometry — the 4 corners, the fitted circle,
  every cut, and extreme-reach points — on BOTH the camera and orthographic views, with a
  separate "Move sample points" tool/button for nudging individual generated preview points.**
  Requested explicitly, for two gaps the previous two-tool split left: extreme-reach points
  could be added or all cleared, but never individually repositioned once placed on either
  view; and the fitted circle was draggable on the orthographic view *regardless of the
  active tool* (an inconsistency with the camera view, where it only worked in "corners"
  mode) while cuts were draggable on the orthographic view only in "editCuts" mode — three
  different gating rules for what is conceptually the same action. `circleHandles()` and
  `getCutHandles()` already returned the identical `{point, onDrag}` shape, so
  `getAllShapeHandles()` just concatenates them, and one drag path (`activeDrag = {kind:
  "corner"|"extreme"|"shape"|"sample", ...}`, shared by both canvases' mousedown/mousemove)
  now handles all four geometry kinds uniformly — replacing the previous three separate
  drag-state variables (`draggingCornerIndex`, `circleDragMode`, `editDragTarget`). Hit-testing
  is priority-ordered (corners, when unlocked, beat shape handles beat extreme points) rather
  than a global nearest-of-everything search, matching the old "corners always win" behavior.
  "Move sample points" is a deliberately SEPARATE tool rather than folded into "Edit": moving
  one generated point overrides the *algorithm's own output* for that index, a different kind
  of action from editing the structural geometry that drives the algorithm, and merging their
  hit-testing risked grabbing the wrong thing when a sample point sits close to a handle.
  Overrides live in `sampleOverrides` (canonical space, keyed by array index), applied in
  `renderAll()` after `sampleSectorPoints()` runs; cleared when steps/seed/distribution change
  (a different count or algorithm makes the index<->point correspondence meaningless and a
  stale override would silently misapply to an unrelated point) but NOT on border-width
  changes or corner/circle/cut edits, which are continuous fine-tuning an override should
  survive — `rescaleCanonicalShapes()` (the existing fix for corner-driven canonical-scale
  drift) now rescales `sampleOverrides` the same way it already rescaled `fittedCircle`/cuts.
  "Lock corners" (the padlock) keeps its original, narrower scope on purpose — it only ever
  suppressed corner-drag hit-testing, never circle/cut/extreme/sample dragging — confirmed via
  the merge that this is still exactly the desired behavior ("only the grid should be locked"),
  not something the merge needed to change.
- **The calibration tool gained undo/redo (Ctrl+Z / Ctrl+Shift+Z or Ctrl+Y, plus toolbar
  buttons), scoped to GEOMETRY only — corners, extreme points, the fitted circle, cuts, and
  sample overrides — not to form fields (steps/seed/distribution/border width), the corners
  lock, or the active tool.** Those excluded items are all plain UI/form state an operator can
  already "undo" by just re-entering the old value or re-clicking a tool button; scoping undo
  to them too would mean tracking a lot more state for little benefit. `recordUndo()` pushes
  one JSON-deep-cloned snapshot (cheap at this tool's data scale) right before each discrete
  mutation — a drag's first mousedown (not every mousemove, so an entire drag gesture undoes
  as a single step), or a button click that changes geometry (fit circle, clear points, reset
  corners, clear cuts, commit a new cut shape) — and clears the redo stack, the standard "new
  action invalidates old redos" rule. A in-progress polygon draft (`polyDraft`) is deliberately
  NOT undo-tracked — only its final commit is — so undo/redo can't leave a half-built polygon
  in a confusing state; `undo()`/`redo()` both call `resetCutDraftState()` defensively for the
  same reason. Verified with a Node test harness (DOM-stub `document`/`window`/`navigator`/
  canvas-2d-context objects, the extracted `<script>` body run in the same scope so its
  top-level `let`/`const` bindings stay visible to the appended assertions) that drives the
  ACTUAL mousedown/mousemove/mouseup listeners end-to-end — not just the underlying
  hit-test/undo functions in isolation — confirming e.g. that dragging a corner, undoing, and
  redoing round-trips correctly even though a corner drag also side-effects `fittedCircle`'s
  absolute canonical numbers via the pre-existing rescale-on-corner-move behavior (the
  snapshot/restore approach handles this for free, since it captures ALL tracked state
  together, not just whatever the current drag directly touches).
- **Drawing a NEW cut (rectangle/circle/polygon) and marking a NEW extreme point now both work
  starting from EITHER view, not just the camera view** — a follow-up report after the
  Edit/Move-samples-tool merge above ("i cant perform cuts on the orthographic surface, not
  mark extreme points") pointed out that merge only covered *editing existing* geometry on
  both views; *creating new* geometry (the "rectangle"/"circle"/"polygon"/"extreme" tools) was
  still wired to `cameraCanvas` alone. Since `cutAnchor`/`cutPreview`/`polyDraft` were already
  canonical-space (not pixel-space) by construction, adding the orthographic-view code paths
  needed no homography at all — `orthoCanvas`'s mousedown/mousemove for these tools use the
  click's canonical point (`p / orthoScale`) directly, where the camera view's equivalent
  needs `toCanonical()`/`toPixel()` conversions; the rectangle/circle commit-on-mouseup logic
  needed no changes at all, since it already only read the (already canonical) draft state,
  not which canvas produced it. Extracted `pushExtremePoint()`/`finishPolygonDraft()` as the
  single shared implementation each tool needs once, rather than duplicating per-canvas.
  `lastMousePos` (pixel-space, camera-only, used only for the polygon-draft rubber-band line)
  was generalized to `polyHoverCanonical` (canonical-space, updated by both canvases' own unit
  conversion), so the rubber-band preview itself is now drawn on both views too — previously
  the orthographic view didn't render an in-progress cut draft at all. Marking a NEW extreme
  point from the orthographic view has a real, documented gap versus the camera view (no live
  arm feed underneath it — see the original "the operator needs to see the real arm" rationale
  above), but the capability was requested explicitly anyway, e.g. for fine-tuning a fit by
  eye against the undistorted surface with the camera view open alongside to cross-check;
  left enabled, not blocked, since restricting it wasn't what was asked for. The knife cursor
  (`.cut-cursor`) and its CSS rule were likewise extended to `#orthoCanvas`, matching whichever
  view a cut tool is actually active on. Verified via 9 new Node-harness assertions (rectangle
  cut drawn end-to-end via simulated mousedown/mousemove/mouseup on `orthoCanvas`, a 3-vertex
  polygon cut closed via simulated `dblclick`, and a new extreme point added via simulated
  mousedown — all starting and finishing entirely on the orthographic view) plus the full
  152-test Python suite (untouched, this is JS-only).
- **Bimanual (2+ robot arms sharing one camera frame) needed NO new top-level schema concept
  in `config.py` -- it's just N `objects[]` entries, each with its own pattern region and an
  optional per-object `marker:` color override, since `ObjectConfig`/`build_pattern`/
  `log_target` already supported independent multi-object patterns end to end.** Investigated
  the actual gap before designing anything: `session.py` already builds one independent
  pattern per `variable: true` object and shows a marker for each, every episode; the only
  thing literally missing was that every object's marker was forced to the SAME global
  `marker:` style, so two simultaneous targets (one per arm) were visually indistinguishable
  except by their text label. Fix: `MarkerConfig` moved above `ObjectConfig` in the file
  (forward-reference-safe either way under `from __future__ import annotations`, but cleaner
  to read), `ObjectConfig` gained a `marker: MarkerConfig` field that's ALWAYS resolved at
  parse time (never `None`) -- `_parse_marker` gained `fallback`/`context` parameters, and
  `load_config` now parses the top-level `marker:` BEFORE objects so each object's own
  optional `marker:` block can fall back to it field-by-field (so an object overriding just
  `color_rgba` still inherits `radius_px`/`label` from the shared default, rather than from
  `MarkerConfig()`'s hardcoded ones). `session.py`'s `log_target` call changed from
  `config.marker` to `obj.marker` -- no `obj.marker or config.marker` fallback dance needed
  anywhere downstream, since it's pre-resolved. `exclude_zones`/`surface_calibration` needed
  *zero* changes: both are already scene-level/shared, which is exactly correct for two arms
  in the same frame (mark each arm's own base as an `exclude_zones` circle, and BOTH arms'
  patterns correctly avoid BOTH bases). See `examples/bimanual.example.yaml` for the resulting
  config shape -- two `objects[]` entries, one shared `surface_calibration`, two
  `exclude_zones` circles (one per arm's base), no "arms:" key anywhere.
- **`vizaudit-calibrate` gained an "Arms" concept -- a row of named, colored chips, each an
  independently-calibrated reach circle -- because hand-running the calibration tool twice
  and manually merging two YAML snippets (re-marking the same 4 corners each time) is not
  "intuitive" for a bimanual session, even though the engine itself needed no new concept
  (previous bullet).** Each arm owns its own `extremePointsCamera`/`fittedCircle`/
  `sampleOverrides` plus its own steps/seed/distribution/border-width -- everything tied to a
  SPECIFIC arm's reach -- while `cameraCorners`/`cutShapesCanonical` stay global/shared
  (one camera frame, shared obstacles, matching the config-side decision above). Implementation
  deliberately avoids threading an explicit "current arm" parameter through every existing
  function (sampleSectorPoints, the edit/sample hit-testers, every drag handler, ~20 call
  sites) -- `extremePointsCamera`/`fittedCircle`/`sampleOverrides` stay plain top-level
  variables holding the ACTIVE arm's LIVE state, and `syncActiveArmFromGlobals()`/
  `loadActiveArmIntoGlobals()` copy between those globals and `arms[activeArmIndex]` at the
  handful of points that actually need to know arms exist: switching arms, recording an undo
  snapshot, and the top of every `renderAll()` (so `arms[activeArmIndex]` -- and therefore the
  export panel and "other arms" rendering, both of which read the `arms` array directly -- are
  never more than one editing gesture stale). These are cheap reference copies, not deep
  clones: an in-place mutation (e.g. dragging `fittedCircle.center`) stays visible through
  `arms[activeArmIndex]` automatically between syncs, since both names point at the same
  object until the next `loadActiveArmIntoGlobals()` swaps the globals to a different arm.
  Undo/redo's `snapshotState()`/`restoreState()` serialize `{cameraCorners,
  cutShapesCanonical, arms, activeArmIndex}` instead of the old flat per-circle fields --
  `snapshotState()` calls `syncActiveArmFromGlobals()` first so the active slot is fresh
  before it's cloned. `rescaleCanonicalShapes()` (the existing fix for corner-driven canonical-
  scale drift) now rescales EVERY arm's circle/overrides, not just the active one's -- a
  corner edit changes the shared homography every arm is interpreted through, regardless of
  which is currently selected, confirmed via a regression test that moves a corner while arm 2
  is active and checks arm 1's (inactive) stored circle also rescaled. Rendering shows the
  active arm at full detail (circle + edit handles + sample-point preview) in its own color,
  exactly as the single-arm tool always did, while every OTHER arm renders as a dimmed
  (`globalAlpha` 0.4-0.55) outline-plus-extreme-points in ITS OWN color, deliberately omitting
  the other arm's sample-point preview/edit handles (those stay active-arm-only, both to limit
  scope and because editing only ever applies to the active arm anyway) -- enough to visually
  confirm two arms' reach circles don't unexpectedly overlap, without recomputing a full sample
  set for every inactive arm on every render. The active arm's circle/extreme-points/samples
  switched from the tool's old hardcoded colors (`--result` green / `--marker` white, now
  removed from `:root` as dead code) to `arms[activeArmIndex].color` directly, so "this arm's
  geometry is color X" stays true whether that arm is active or not -- the first arm keeps the
  tool's original green as `ARM_PALETTE[0]`, so a true single-arm session looks unchanged.
  Export emits one `pattern:` (+ a `marker: {color_rgba: ...}` suggestion, matching the
  config-side feature above) block per arm with a fitted circle, labeled by name in the
  paste-target comment; with exactly one arm, the suggested-marker block is omitted entirely
  and the output is byte-for-byte what this tool already produced before bimanual support
  existed -- verified by a regression test asserting no `marker:` substring appears in that
  case. "Mark extreme points" and the three cut tools stay as before (add to whichever is
  active for extreme points; cuts are always shared, not per-arm). Verified via 35 new
  Node-harness assertions (arm add/remove/switch isolation, undo/redo across `addArm()`/
  `removeArm()`, the corner-edit rescale reaching inactive arms, chip rendering and
  click-to-switch exercised through the real DOM stub rather than calling `switchArm()`
  directly, and both single- and multi-arm export formats) plus the full pre-existing
  46-assertion suite (unaffected) -- 81 total.
- **A bimanual rig's two arm circles typically OVERLAP in the shared middle of the
  workspace, and sampling each circle's pattern independently (the original "Arms" design)
  silently double-samples that overlap** — every point in the overlap region is reachable by
  EITHER circle's own independent pattern, so it ends up at roughly twice the point density
  of the non-overlapping parts. Found by direct user pushback before any code was written
  ("the ranges of work of a bimanual setup intersects between the 2 arms and that's the main
  problem you should solve, how to distribute the points in the union of 2 ranges") — fixed
  at the ENGINE level, not just the calibration tool, since a calibration-tool-only fix would
  have nothing real to export: `pattern.py` gained `generate_union_points(circles, count,
  seed, distribution, ...)`, `config.py` gained a `shape: "union"` pattern
  (`circles: [{center, radius}, ...]`, no `inner_radius`/angle fields — scoped to full disks
  only, exactly what every fitted reach-circle already is), and `vizaudit-calibrate` gained an
  "Independent" toggle in the Arms section, **off by default** (combined/union mode) since
  with only 1 arm the two modes are identical, so this is purely additive for existing
  single-arm sessions. ON restores the original per-arm-independent behavior, for the case
  where two arms genuinely work separate, non-overlapping zones and N independent patterns
  really is what's wanted.
  - **`"grid"`** generalizes the existing single-circle full-disk fast path almost for free:
    each lattice column's valid y-range becomes the UNION of every circle's chord at that x
    (`mergeIntervals`/`_merge_intervals`, the SAME interval algebra already used for
    `exclude_zones` subtraction in the single-circle path), instead of one circle's chord.
    Merging is exactly what prevents overlap double-counting — a column straddling two
    circles' chords gets ONE merged range, with `count` allocated to it proportionally to
    that merged range's own length (`_allocate_shares`), not to each circle's chord length
    separately (which would double-count the overlap). `exclude_zones`/`bounds` apply
    per-column exactly as before.
  - **`"random"`** is bounding-box rejection sampling (uniform x/y over the union's bbox,
    accept if inside ANY circle), deliberately NOT "pick a circle weighted by area, then
    sample within it" — that alternative sounds right but reintroduces the exact bug this
    function exists to fix: a point in the overlap is reachable from EITHER circle's draw, so
    it gets accepted roughly twice as often as a non-overlapping point of the same area.
    Plain bbox rejection has no such bias; the only cost is reduced efficiency when circles
    are far apart relative to their radii (same `count * 200` attempt cap as everywhere else
    in this engine — revisit only if that's ever a real reported problem).
  - **`"radial"` is explicitly NOT YET SUPPORTED for a union** (raises a clear error naming
    `'grid'`/`'random'` as the alternatives) — the Fermat/Vogel spiral is inherently
    single-center, and correctly generalizing its area-CDF machinery to an arbitrary union of
    differently-centered circles needs its own derivation (sketched but deliberately deferred:
    reuse `_circle_zone_angle_block`'s closed form — built for "angular extent an EXCLUDE zone
    blocks at radius r" — to instead compute "angular extent a union MEMBER allows at radius
    r" from a chosen origin, merge those as an inclusion constraint via
    `_merge_angle_intervals`, and feed the result into the existing CDF-based per-point
    placement). Shipped without it because `"grid"`/`"random"` alone are already a complete,
    correct, testable increment, and radial-for-union deserves its own pass.
  - **Relocation has no single `center`/`effective_outer` to ray-search from for a union** (the
    existing `_relocate_if_invalid` signature assumes one) — resolved by delegating each
    invalid point to whichever circle's center it's closest to (by signed distance to that
    circle's own boundary: `dist(point, center) - radius`, so "already inside" scores
    negative), then clamping/searching using THAT circle's own geometry. A per-point, not a
    per-pattern, choice — two different points needing relocation can delegate to two
    different circles.
  - **`expected_spacing` (the relocation search-scale AND the `_occupancy_guard`'s
    `min_separation` basis) had to be based on TOTAL CIRCLE AREA, not the bounding box's
    diagonal** — a real test failure, not a hypothetical: two very differently-sized circles
    (or any two circles separated by empty space) share one wide bounding box, so a
    bbox-diagonal-based estimate systematically OVERESTIMATES how loosely packed the SMALLER
    circle's own points actually are once `_allocate_shares` gives it its proportionally
    smaller share of `count` — the occupancy guard's `min_separation` ended up tighter than
    that circle's true point-to-point spacing, so legitimate grid points were rejected as
    "too close" and the relocation fallback exhausted its 200 attempts trying to find room
    that didn't exist within that one circle. Fixed by computing
    `expected_spacing = sqrt(total_circle_area / count)` (ignoring overlap — fine for a
    search-scale estimate, not a correctness-critical value) instead.
  - **The calibration tool's `independent` flag governs ONLY which sampling mode is used for
    steps/seed/distribution/border_width/sample-overrides — `extremePointsCamera`/
    `fittedCircle` (each arm's own calibration) stay per-arm regardless of the flag**, since
    calibrating each arm's own reach circle is always a per-arm action either way. A new
    `combinedSettings` object (shaped like one arm's settings fields, minus the circle/extreme
    points) holds the shared steps/seed/distribution/border_width/overrides used when
    `independent` is false; `syncActiveArmFromGlobals()`/`loadActiveArmIntoGlobals()` (already
    the sync points for every other arm-related state) gained one branch each, choosing
    `arms[activeArmIndex]` vs `combinedSettings` as the sync target/source based on
    `independent` — every other call site (rendering, export, undo/redo) was untouched.
    Rendering: in combined mode every arm's circle/extreme-points render at FULL strength in
    its own color (not the dimmed "other arms" treatment independent mode uses), and the
    shared sample-point dots render in a neutral color (`#f5f5f5`, the tool's pre-bimanual
    default) rather than any one arm's color, since the result doesn't "belong" to one arm.
    Export: 2+ circles emit one `shape: union` block; exactly 1 circle emits the original
    `shape: sector` block (NOT a degenerate 1-circle union) specifically so a single-arm
    session's export stays byte-for-byte identical to what this tool emitted before bimanual
    support existed — caught by a test that initially failed because the first implementation
    used `shape: union` unconditionally whenever `independent` was false, regardless of count.
  - Verified via 13 new Python tests (including a closed-form circle-circle lens-area formula
    as ground truth for "the overlap's actual point share is close to its analytic area
    share, not ~2x it" — not just "no obvious crash") and 16 new Node-harness assertions
    (direct `sampleUnionPoints` checks, end-to-end toggle wiring through the real checkbox
    element, and both single-circle/multi-circle export formats) — 169 Python tests and 97 JS
    assertions total, all passing.
- **The calibration page's "Radial" distribution option is labeled "Radial (beta)" and gets
  disabled (with a fallback to "grid") whenever it's actually unusable, instead of silently
  letting the operator pick it and only finding out from an error in the export panel.**
  "(beta)" reflects that radial has had substantially more iterative correctness fixes than
  grid/random (see the long history of radial-specific bullets above) — an honest signal,
  not a new restriction, since it's always been fully usable for the one circle it's actually
  scoped to. The disable rule: unavailable exactly when combining 2+ circles (`independent:
  false` and more than one arm has a fitted circle), since `generate_union_points`/
  `sampleUnionPoints` don't support radial for a union at all (see the bullet above) —
  independent mode always samples exactly one (the active) circle at a time, so radial stays
  available there regardless of how many arms exist overall. `updateDistributionAvailability()`
  toggles the `<option>`'s `disabled` attribute and, if "radial" was already selected when it
  becomes unavailable (e.g. a second arm just got fitted), force-falls-back to "grid" rather
  than leaving the preview stuck showing an error.
  - **Fixed a real, separate bug found while building this, not just a missing feature: a
    single-arm session in the DEFAULT (combined) mode routed through `sampleUnionPoints`
    unconditionally, which rejects `"radial"` regardless of circle count** — so a brand-new
    single-arm session (the common case) selecting "Radial" got a preview error for no good
    reason, since the union machinery doesn't even apply when there's only one circle to
    union. Fixed with `sampleCombinedPoints(circles, count, seed, distribution)`, which
    dispatches to `sampleSectorPoints` (supporting all three distributions, including radial)
    for exactly one circle, and `sampleUnionPoints` only for 2+ — mirroring
    `updateExportText`'s existing "exactly 1 circle → `shape: sector`" special case, so the
    live preview and the exported config always agree. Implemented by temporarily repointing
    the module-level `fittedCircle` (which `sampleSectorPoints`/`isValidCandidate` read
    directly) at the one circle being sampled, restoring it in a `finally` — restoration must
    happen even if `sampleSectorPoints` throws, since `fittedCircle` is relied on elsewhere to
    mean "the ACTIVE arm's circle," not whichever circle happened to be the lone entry in the
    union (the lone circle isn't necessarily the active arm's own — the active arm could still
    be uncalibrated while a different, inactive arm already has one).
  - Verified via 9 new Node-harness assertions (single-arm combined-mode radial now matches
    `sampleSectorPoints` called directly, the option's `disabled` state flips correctly as a
    second circle is added/removed and as `independent` toggles, the auto-fallback-to-grid
    behavior, and `fittedCircle` is correctly restored afterward) and a Python test asserting
    the static HTML actually contains the "(beta)" label — 170 Python tests and 106 JS
    assertions total, all passing.
- **Orientation guidance is an arrow projected from canonical (plane) space into camera
  (pixel) space, not a 3D primitive — because a homography already is the correct planar-
  perspective transform for anything resting on the calibrated surface.** Considered, and
  rejected, representing an oriented object with an actual 3D primitive (a box/arrow rendered
  via Rerun's 3D pinhole-camera machinery and projected): that needs a real camera pose
  (rotation + translation + intrinsics) to decompose the homography into, which this project
  has deliberately never collected anywhere else (vision-only, calibration-free, no per-
  robot/camera data — see the no-FK decision at the top of this file). A homography is not
  angle-preserving (it maps lines to lines, but not angles between them), which is exactly
  why projecting a canonical-space rotation through it gives the correctly foreshortened/
  skewed look a tilted camera should show — the identical principle `sector`'s area-uniform
  sampling already relies on for position. The cost: this can't represent a tall object's
  out-of-plane tilt, but for a footprint-on-a-surface pick-place/orientation task that's out
  of scope anyway.
  - `pattern.py` gained `generate_rotation_angles(count, method, angle_start_deg,
    angle_end_deg, seed)` — `count` target angles, generated the same way a position
    pattern's points are: `"uniform"` places each angle at the center of its own equal
    sub-slice (`angle_start_deg + (i + 0.5) * span / count`), NOT at both inclusive
    endpoints the way `generate_arc_points` does for a deliberately partial arc — endpoint-
    inclusive would duplicate the 0°/360° wraparound point for the common full-rotation
    case. `"random"` is `count` independent seeded `random.Random(seed).uniform(...)` draws.
    Mirrors the midpoint convention `generate_sector_points`'s `"radial"` distribution
    already uses for its own evenly-spaced parameter `t`.
  - `pattern.py` also gained `orientation_arrow_points(position, angle_deg, length,
    homography)` returning the `(tail, tip)` pixel-space points of the guide arrow. The one
    subtlety: by the time an episode's target `position` is selected, it's already pixel
    space (`build_pattern` applies the forward homography internally) — there's no separate
    canonical-space copy of it lying around in `session.py`. Rather than threading one
    through just for this, the function recovers canonical coordinates via
    `invert_homography`, applies the rotation there, and forward-maps the tip back — exact
    up to floating point, since `apply_homography`/`invert_homography` are exact inverses.
    With no `surface_calibration` (the common, uncalibrated case), canonical space and pixel
    space are the same thing, so the rotation is applied directly with no round trip.
  - **The position pattern's `count` and the orientation pattern's `count` are deliberately
    independent, not coupled 1:1** — "how many rotations" is its own knob, not derived from
    "how many positions." Both cycle per-episode via the same `target_for_episode` (now
    generalized with a `TypeVar` so it works for `float` angles, not just `Point`s), each
    with its own list length — e.g. lengths 5 and 4 only repeat the exact same position/
    rotation pairing every 20 episodes. Decoupling them only adds coverage diversity over a
    session; there was no reason to force them equal or derive one from the other.
  - `config.py` gained `OrientationConfig` (`count` required; `method` default `"uniform"`;
    `angle_start_deg`/`angle_end_deg` default the full `0`/`360` rotation; `seed` default
    `0`; `arrow_length` default `40.0`, in the same space as the object's own pattern) and
    `ObjectConfig.orientation: OrientationConfig | None = None` — purely opt-in, so every
    existing config without an `orientation:` block is completely unaffected (no arrow,
    `Points2D`-only marker, exactly today's behavior). Gated on `variable: true` the same way
    `pattern` is — an `orientation:` block on a `variable: false` object is a config error,
    since there's nothing to vary either way.
  - `rerun_client.py`'s `log_target` gained an `orientation_tip` parameter: when given, logs
    an `rr.Arrows2D` as a `{path}/orientation` child entity right alongside the existing
    `Points2D` marker. Deliberately takes already-projected pixel-space points and does no
    perspective math itself — that stays in `pattern.py`/`perspective.py`, keeping this
    module a thin Rerun-logging wrapper, consistent with its existing role. Whether an object
    has an orientation arrow at all is fixed for the whole session by its config (never
    toggled per episode), so there's no need to ever clear a stale one.
  - `session.py` builds a `rotations: dict[str, list[float]]` alongside the existing
    `patterns` dict (only for objects with `orientation` set), and per episode computes
    `orientation_tip` via `orientation_arrow_points` using the session's one shared
    `homography` (already computed once for `sector`/`union` patterns) before calling
    `log_target`.
  - Verified via 23 new Python tests (14 in `test_pattern.py` covering
    `generate_rotation_angles`'s uniform-midpoint/no-duplicate-endpoint/random-determinism/
    validation behavior, `orientation_arrow_points`'s tail-equals-position/no-homography-
    pixel-rotation/identity-homography/canonical-space-rotation behavior — the last verified
    by independently recomputing the expected tip via `invert_homography`/`apply_homography`
    and confirming it differs from a naive pixel-space rotation — and the generalized
    `target_for_episode` working over `float`s; 9 in `test_config.py` covering parsing,
    defaults, and every validation error) — 193 Python tests total, all passing.
  - `calibrate.html` gained the same feature as a live preview, in a follow-up pass: a new
    "Orientation preview" toolbar group (Show/Count/Method/Start°/End°/Seed/Arrow len) draws
    the same guide arrow from every previewed sample point. The JS port is actually SIMPLER
    than `pattern.py`/`session.py`'s version, not just a mirror of it: the calibration tool
    already keeps sample points in canonical space directly (`samplePointsCanonical`), so
    `orientationTipCanonical(positionCanonical, angleDeg, length)` just rotates there with no
    `invert_homography` round-trip needed at all (that round-trip in `pattern.py` exists
    specifically because `session.py` only ever has the final pixel-space point). `tip` is
    then forward-mapped for the camera view (`toPixel`) and used directly, scaled by
    `orthoScale`, for the orthographic view — drawn with a small new `drawArrow()` helper (a
    line plus a 2D "V" head, no perspective math of its own, since the caller already
    projected its endpoints).
  - `orientationEnabled`/`orientationCount`/`orientationMethod`/`orientationAngleStart`/
    `orientationAngleEnd`/`orientationSeed`/`orientationArrowLength` were added to `makeArm()`
    and `combinedSettings`, and wired into `syncActiveArmFromGlobals`/
    `loadActiveArmIntoGlobals` exactly like `steps`/`seed`/`distribution`/`borderWidth`
    already were — per-arm in independent mode, shared in combined mode (off by default, so
    every single-arm session is unaffected). Deliberately does NOT clear `sampleOverrides` on
    change (unlike steps/seed/distribution): orientation never changes the position sample
    set's size or algorithm, only what's drawn alongside it, so a position override has no
    reason to become stale when an orientation field changes.
  - Export gained `orientationExportBlock(settings)`, emitting a sibling `orientation:` block
    (same field names as `OrientationConfig`) right after the relevant `pattern:` block —
    once per arm in independent mode, once for the combined pattern otherwise — and emitting
    nothing at all when disabled, so a config that never touches this stays byte-for-byte
    unaffected.
  - Verified via 19 new Node-harness assertions (`generateRotationAngles`/
    `orientationTipCanonical` unit checks; end-to-end enable/disable through the real
    checkbox+change-event path, confirming tip count, the expected angle cycling, and the
    camera-space projection all via independent recomputation rather than just "it ran with
    no error"; export-text contains/omits the block correctly; and per-arm vs. combined
    settings isolation, mirroring the existing steps/seed isolation tests) — 125 JS
    assertions total, all passing, plus the unaffected 193 Python tests (no Python files
    touched in this pass).
- **`generate_rotation_angles`'s `angle_start_deg`/`angle_end_deg` are RELATIVE to
  `initial_angle_deg`, not absolute** -- `initial_angle_deg=90, angle_start_deg=-45,
  angle_end_deg=45` spreads +-45 degrees around direction 90, instead of forcing the operator
  to compute absolute angles by hand to spread around an arbitrary general direction
  (reported as "hard to control which general angle"). Both `"uniform"` and `"random"` use
  `start = initial_angle_deg + angle_start_deg` / `end = initial_angle_deg + angle_end_deg`;
  negative values are fine (no sign validation). Default `initial_angle_deg=0` makes this an
  exact no-op for every config written before this field existed. `session.py`'s `"random"`
  per-point branch and `calibrate.html`'s JS port both pass `initial_angle_deg` through too
  (the JS port previously only passed it on the shared "uniform" path).
- **`generate_sector_points` gained `count_mode: "fixed" | "variable"`** (`PatternConfig`/
  `config.py`'s sector schema, default `"fixed"`, sector-only). `"fixed"` is the existing,
  unchanged exact-count-via-relocation behavior. `"variable"` (grid/radial only -- raises for
  `"random"`) generates the same IDEAL closed-form positions but just drops whichever aren't
  valid instead of relocating them, so the final count can be `<= count`. Trades "exactly
  `count` points" for a much simpler, relocation-free path -- no `_allocate_shares`, no CDF,
  no refinement pass -- requested specifically because relocation can clump points unevenly
  near an irregular cut/boundary, and a "just drop invalid ones" mode sidesteps that entirely.
  Ported identically to `calibrate.html` (`sampleSectorPointsVariable`, gated by a new "Count
  mode" dropdown that disables "Variable" the same way "Radial (beta)" is disabled -- for
  `distribution: random`, or when combining 2+ arm circles, since the union path doesn't
  support it).
- **The calibration tool's box-select tool (⬚ `toolOrientToggle`) stamps the CURRENT toolbar
  orientation settings onto enclosed points as a full per-point override** (`orientationOverrides`,
  index -> `{enabled, count, method, angleStart, angleEnd, seed, arrowLength, initialAngle}`),
  not just an on/off flag -- drag again to clear it. To disable a subgroup while others stay
  on: uncheck "Show", drag-select them (stamps `enabled:false`), recheck "Show" for the rest.
  Still calibration-tool PREVIEW-ONLY (no export hook), same as `sampleOverrides`.
- **`generate_rotation_angles`'s `angle_start_deg`/`angle_end_deg` are relative to
  `initial_angle_deg`** (e.g. `initial_angle_deg=90, start=-45, end=45` spreads +-45 degrees
  around direction 90). Its `"uniform"` spacing now divides by `count-1` (endpoint-inclusive,
  so the arrows always span the literal `[start, end]` regardless of `count`) unless the span
  is a full 360 wrap (divides by `count`, no duplicate at the seam) -- dividing by `count`
  unconditionally (the prior behavior) left a gap that only shrank toward `end` as `count`
  grew, making the spread look like it depended on `count` instead of being set directly by
  `start`/`end`. Ported identically to `calibrate.html`.
- **`count_mode="variable"` now treats `count` as a TARGET, iteratively re-trying the grid/
  radial generation at a higher/lower density (`_search_variable_density` in `pattern.py`,
  `searchVariableDensity` in `calibrate.html`) until the survivor count is as close to
  `count` as achievable** -- a single fixed-density shot could land far short of `count`
  whenever much of the candidate grid/spiral falls outside the valid area (small circle in a
  big bounding box, a restrictive cut, etc.). The calibration tool also shows the achieved
  count next to "Steps" (`actualCountLabel`) whenever count_mode is "variable".
- **Fixed a bug where toggling "Independent" with only 1 arm changed the effective
  settings** -- `arm`/`combinedSettings` are still two separate objects even with 1 arm, so
  whichever wasn't the live sync target silently kept stale/default values. `SETTINGS_FIELDS`
  + a mirror step in `syncActiveArmFromGlobals` now keeps both objects identical whenever
  `arms.length === 1`, so the toggle is a true no-op in that case (matching "a union of 1
  circle IS that circle").
- **Fixed `"radial"` (fixed count_mode) ignoring `border_width` near an off-center
  `exclude_zones` cut.** Root cause: `_refine_radial_local_separation`'s candidate search
  only checked a cut's APPROXIMATE inflated boundary (vertices pushed outward from the
  centroid, not a true Minkowski offset -- see `_inflate_polygon`'s own docstring), and
  picked/moved a point there without ever re-validating against the exact
  `_point_near_polygon` check, unlike every other placement path in this engine. Fix: it now
  takes `is_valid` and skips any candidate that fails the exact check. Ported identically to
  `calibrate.html`'s `refineRadialLocalSeparation` (via `isValidCandidate`).
- **Box-select stamps a visible marker (yellow ring) on every overridden sample point**, on
  both views -- the toggle had no visual feedback at all before, so there was no way to tell
  which points were actually selected/affected after a drag. A plain click (no real drag)
  with the same tool clears every override at once.
- **Fixed combined-mode sample dots/arrows not rendering on the camera view at all when the
  ACTIVE arm has no fitted circle yet** (e.g. right after adding a 2nd arm, before fitting
  it) -- they were nested inside `if (fittedCircle)` (the active arm's OWN circle), but in
  combined mode the dots belong to the UNION of every arm's circle, not just the active one's.
  Moved them out of that gate; the orthographic view already had this right.
- **Added a "+" add/remove-sample-point tool**: click empty space to add a point (canonical,
  appended after generation), click an existing one to remove it. `extraSamplePoints`/
  `removedSampleIndices` (per-arm/`combinedSettings`, synced like `sampleOverrides`);
  `removedSampleIndices` clears on steps/seed/distribution/count_mode changes (its indices
  stop meaning the same thing), `extraSamplePoints` survives them (absolute points, not
  index-based). `sampleOrigins` tracks what each final point actually is (generated vs.
  manually added) so a click knows whether to un-remove or delete.
- **`count_mode="variable"` is now supported for a UNION of 2+ arm circles (`distribution:
  "grid"` only), not just a single circle** -- previously hard-locked out (`Variable` disabled
  in the calibration tool) whenever combining 2+ circles, for no reason beyond "not
  implemented yet"; a user asked directly why. `generate_union_points`/`sampleUnionPoints`
  both gained a `count_mode` param mirroring `generate_sector_points`'s existing variable
  branch almost exactly: enumerate an ideal lattice over the union's bounding box at a given
  density, keep only points inside the union/bounds/exclude_zones, and retry via the same
  `_search_variable_density`/`searchVariableDensity` until the survivor count is as close to
  the target as achievable -- no relocation/CDF machinery needed, so this was actually
  *simpler* to add than the fixed-count union path. Scoped to `"grid"` only (mirrors fixed
  union mode's own radial restriction: the Fermat spiral has no union generalization yet).
  `updateCountModeAvailability()` simplified to just `distribution === "random"` --
  `updateDistributionAvailability()` already disables `"radial"` outright once 2+ circles are
  combined, so by the time count-mode availability is checked, `"grid"` + 2+ circles is
  always reachable and always valid.
- **Box-select's "stamp current settings onto enclosed points" design was backwards from how
  a selection-bound editor should behave -- reported directly: "I selected some points and
  changed the settings but the unselected ones were affected instead."** Root cause: the old
  design wrote a frozen snapshot into `orientationOverrides[i]` at DRAG time, then kept
  writing every subsequent toolbar edit into the shared DEFAULT settings -- so a point you'd
  just selected was the one edit immune to further changes, and everything else (visually
  unmarked, "not selected") kept reacting to the panel. Fixed by adopting the standard
  selection + inspector-panel pattern (Figma/Blender-style): `orientationSelectedIndices`
  (transient, not undo-tracked, not per-arm) tracks which points the box-select tool has
  selected; `syncActiveArmFromGlobals()` now branches on it -- selection non-empty -> every
  orientation field edit overwrites ONLY the selected points' `orientationOverrides` entries
  (full snapshot each time, shared `captureOrientationInputsAsConfig()`/
  `populateOrientationInputsFrom()` helpers also deduplicating 3 previously-separate copies
  of the same 8-field block); selection empty -> edits the shared default exactly as before.
  A box-select drag now toggles SELECTION membership (not override existence); a plain click
  deselects everything WITHOUT deleting any override (selecting/deselecting is no longer
  destructive, unlike the old "click clears all" -- the only way to actually clear an
  override now is the new "Reset to default" button, which also clears the selection itself,
  since leaving a just-reset point selected would have the very next render's sync
  immediately re-stamp it from the still-live panel fields). Selected points render an
  additional cyan ring (`#38bdf8`) alongside the existing yellow "has an override" ring, on
  both views, and the orientation toolbar-group's label live-updates to "(N point(s)
  selected)" so it's never ambiguous what the panel is currently bound to.
- **Added a thin visual divider (`.toolbar-divider`) between functional sub-clusters inside
  the "Edit" toolbar-group** -- requested directly ("UX is getting overwhelming... group
  things closer"): that one group had accreted 10 buttons across many separate passes
  (corner/circle/cut editing, moving sample points, the add/remove-point tool, the
  orientation box-select tool, 3 cut-drawing primitives, clear-cuts) with nothing but
  adjacency distinguishing genuinely different categories of action. Grouped into [Edit, lock
  corners, reset corners] | [move samples, add/remove point, select points] | [rectangle/
  circle/polygon cut, finish] | [clear all cuts] -- the same segmented-toolbar idiom Figma/
  Photoshop use for an analogous problem. Pure CSS/HTML, no behavior change. (Superseded by
  the card-based layout below, which moved this entire group out of the top toolbar; the
  same cluster grouping carried over as separate rows within the new "Editing" card.)
- **A plain click with the box-select ("select points") tool clears every orientation
  override, not just the selection** -- reported directly: "theres still the yellow rings
  that do not disappear." The previous round deliberately made deselecting non-destructive
  (only the explicit "Reset to default" button cleared overrides), reasoning that "deselect"
  shouldn't delete data -- but that left no quick, discoverable way to make the yellow rings
  go away, which is what people actually expect from clicking elsewhere. Click now resets
  both (selection + every override); a real box-drag still only toggles SELECTION membership
  (unaffected) and "Reset to default" still exists for scoping a reset to just the current
  selection while leaving everything else untouched.
- **Reorganized the page from one long top toolbar + a wall of explanatory prose into:
  a short subtitle (detail now lives in each button's `title` tooltip, already present) +
  the camera/orthographic views with a THIRD column beside them holding three cards --
  Editing (reach-circle actions, the tool icon rows, undo/redo, grid toggle), Pattern
  (steps/seed/distribution/border/count mode), and Direction (orientation).** Requested
  directly: the old layout put every control in one toolbar panel ABOVE the views, so
  switching tools or tweaking the pattern while looking at a view meant scrolling up and
  back down repeatedly. Putting the cards in the same flex row as the views (`.side-column`,
  wraps below on narrow viewports) keeps them in view together at all times -- no element
  IDs changed, only their HTML position, so no JS beyond two new visibility toggles (below)
  was needed. "Camera" (device/start) and "Arms" stay in the original top toolbar -- they're
  session/profile setup, not per-edit tool switches, so they don't share the scrolling pain.
- **The Direction card's count/method/angle/seed/arrow-length/reset fields are hidden
  entirely unless "Show" is checked** (`orientationDetailFields`, `display:none` toggled in
  `renderAll()`) -- requested directly ("unless enabled the direction settings should not be
  visible"); orientation is off by default for most sessions, so showing 8 fields nobody's
  using by default was pure clutter.
- **The orthographic view's "extreme-reach points / fitted circle / samples are colored per
  arm" legend line is now hidden whenever there's only 1 arm** (`armColorLegend`, the common
  case) -- requested directly ("either make it variable or get rid of it"); the sentence is
  only meaningful once a SECOND arm's color actually needs distinguishing from the first.
- **The previous round's side-column-of-cards was itself revised: only "Editing" stays beside
  the views, as a narrow ICON-ONLY rail (`.icon-rail`), not a full card -- requested directly
  ("we need all the horizontal space for the 2 views"). Workspace setup/Pattern/Direction went
  back to being cards in the top toolbar panel.** The rail has no room for text labels, so
  clicking a tool now shows a one-line hint (`toolHint`/`TOOL_HINTS`) below the icon stack
  instead of relying on hover-only tooltips -- the "expands once clicked" behavior asked for.
  "Finish polygon" was dropped from the rail entirely (double-click or Esc already finish a
  polygon cut; the button was redundant). Undo/redo and the Grid toggle moved into the
  Orthographic view's own header row (top-left), since they're view-display controls, not
  tools -- freeing the rail down to just: Edit, Move samples, Add/remove point, Select points,
  Rect/Circle/Polygon cut, Clear cuts.
- **Added a "Workspace setup" toolbar-group: Mark reach points / Fit circle / Clear points,
  plus a NEW dedicated corner-and-edge tool (`toolCorners`, its own icon) separate from the
  general "Edit" tool.** Previously "Edit" silently also dragged the 4 corners alongside
  circle/cuts/reach-points, which stopped matching its own tooltip once Workspace setup became
  its own labeled category -- `findCameraEditTarget` no longer touches corners at all (moved
  entirely to `findCameraCornersTarget`, camera-view-only since the orthographic rectangle's
  corners aren't independently draggable). The new tool ALSO hit-tests the 4 edge midpoints
  (`kind: "edge"`) -- dragging one translates both its corners by the same delta, so the whole
  side slides/scales instead of only ever being able to drag one corner vertex at a time
  (requested explicitly: "allowing scaling it fully and just dragging the sides not only
  corners"). `cornersLocked` now guards both corner AND edge hits.
- **Added "lock circle" (Workspace setup) and "lock pattern" (Pattern card) toggles, mirroring
  the existing "lock corners" pattern.** `circleLocked` removes the ACTIVE arm's fitted circle
  from `getAllShapeHandles()` (cut-shape circles are unaffected -- a different thing entirely)
  and disables "Fit circle" so a re-fit can't silently overwrite a locked circle.
  `patternLocked` just disables the Pattern card's 5 inputs directly -- there's no drag
  interaction to gate, so no handle-filtering needed.
- **Pattern preview"/"Orientation preview" renamed to plain "Pattern"/"Direction", and the
  "(combined)"/"(active arm)" suffix only appears once there are 2+ arms** -- with exactly one
  arm "combined" vs "independent" is a meaningless distinction (a union of 1 circle IS that
  circle), so showing it was just noise, reported directly ("doesnt make sense" for one arm).
- **Rewrote the top intro text from an implementation-detail wall of prose into one sentence
  (what this tool is) + a numbered how-to** -- reported directly that the previous rewrite
  still explained HOW each tool worked internally rather than what to DO, and never actually
  defined "reach point" (renamed to "Mark reach points" everywhere in the UI, and now spelled
  out in the how-to: "click 3+ spots at the edges of where the arm can physically reach").
  Implementation detail (homography, undistorted space, etc.) was dropped entirely -- it's not
  something a first-time operator needs to know to use the tool.
- **The icon-rail-beside-the-views from the previous round was rejected outright ("wtf is this
  sidebar?? It shouldnt occupy horizontal space at all") and replaced with a floating,
  collapsible toolbar (`#floatingToolbar`, `position: fixed; top/right`)** -- takes ZERO
  layout space from the 2 views (it's outside the flex flow entirely) and stays on-screen
  regardless of scroll, which is what actually solves the original "have to scroll to switch
  tools" complaint, not a sidebar. Collapsed to one toggle button (`toolbarToggleBtn`,
  hamburger icon) by default; clicking it shows/hides the tool row (`#floatingToolbarTools`).
  The per-tool hint text under the old rail ("the description underneath") is gone entirely --
  it was redundant with the tooltips every button already had.
- **Workspace setup's tool is now the ONLY way to edit corners/edges/the reach circle/reach
  points -- the general "edit" tool (floating toolbar) dropped them and is cuts-only now.**
  Requested directly: "reach circles and points should be editable from the setup tab edit
  tool only." `findCameraCornersTarget`/`findOrthoCornersTarget` (Workspace setup's tool)
  gained circle + reach-point hit-testing (on top of the corner/edge hit-testing they already
  had); `findCameraEditTarget`/`findOrthoEditTarget` had circle/reach-point hit-testing
  removed, leaving cuts as the only thing they touch. Circle handles only render while the
  "corners" tool is active (camera AND ortho now, for consistency -- the ortho view used to
  draw them unconditionally regardless of tool).
- **The "edit" tool gained click-a-cut's-body-to-select-it + Delete/Backspace-to-remove-just-
  that-one** -- asked directly ("did i say that the cuts should also be able to be selected
  and deleted one by one?"); previously the only removal path was "Clear all cuts" (everything
  at once). `selectedCutIndex` (-1 = none) is set by `pointInCutShape` when a mousedown in the
  "edit" tool doesn't land on a handle; rendered as a thicker/brighter outline on both views.
- **"Move sample points" and the orientation box-select tool merged into one "select" tool**
  -- requested directly ("what if i want to select a bunch of points and move them around?").
  Drag a box to select multiple points (unchanged -- the Direction panel still edits exactly
  that selection); drag any ALREADY-selected point (when 2+ are selected) to move the WHOLE
  selection together (`activeDrag.kind = "sampleGroup"`, storing each selected point's
  original position plus one drag anchor, so every mousemove applies the same delta to all of
  them -- not pairwise from the previous mousemove, which would drift). Dragging a point that
  ISN'T part of a multi-selection still just moves that one point, identical to the old
  "Move sample points" tool -- so the common single-point-nudge workflow is unchanged.
- **`patternLocked` now also blocks moving/adding/removing sample points, not just disabling
  the 5 form inputs** -- reported directly ("should actually lock all samples from getting
  moved or deleted"). `startSelectDrag()` (shared by both canvases' "select" tool) and the
  "addRemovePoint" tool's mousedown both check `patternLocked` and refuse to start a
  mutation; box-select itself (just a selection, not a mutation) is unaffected.
- **Pattern/Direction/Workspace setup went back to being separate `.panel` cards (each with
  its own border) instead of `.toolbar-group`s sharing one bordered panel** -- requested
  directly ("I still wanted the pattern and direction sections to be separated by cards, its
  more visually appealing"). The outer `.toolbar` div is now just an unstyled flex row; each
  former toolbar-group is its own `.panel` child.
- **The previous round's 5-card split was itself wrong -- corrected to exactly 2 cards: [Camera
  + Arms + Workspace setup] and [Pattern + Direction]**, per explicit correction ("you split
  all of them into cards, camera arms and workspace should be in the same card" / "put pattern
  and direction in a different card together"). "Separated by cards" meant two groups
  separated from EACH OTHER, not five individually-boxed sections.
- **The floating toolbar moved from a page-corner `position: fixed` overlay into the Camera
  view's own header row (in-flow, right-aligned via `margin-left: auto`)** -- reported
  directly ("move it closer to the actual edit space in a way that wont collide with any
  other visuals"): a page-corner overlay can drift far from the views on a tall page and risks
  overlapping the cards above. Living in the view's own header can't collide with anything
  (same pattern the orthographic view's undo/redo/grid controls already used) and is about as
  close to "the edit space" as a control can get without overlapping the canvas itself. Still
  collapsible (`toolbarToggleBtn`), still costs no extra width (the header row already existed
  for the "Camera view" title).
- **The cuts-only "edit" tool was folded into "select"** -- reported directly ("combine the
  box selection and edit tool together into one they dont collide in any functionality"):
  cuts and sample points are different geometry with different hit-tests, so checking
  cut-handle -> cut-body -> sample-point -> box-select in sequence (same "first match in
  priority order" idiom already used everywhere else in this file) merges them with zero
  ambiguity. One fewer tool/button; `findCameraEditTarget`/`findOrthoEditTarget` (now called
  from inside the "select" branch, not their own `tool` value) are otherwise unchanged.
- **Delete/Backspace now also removes box-selected SAMPLE points, not just a selected cut** --
  reported directly ("I cant use the delete function for points after box selecting them").
  `deleteSelectedSamples()` maps each selected index through `sampleOrigins` exactly like
  `addRemoveSamplePointAt` already does for a single point, but must splice `extraSamplePoints`
  highest-index-first within the batch -- removing several ascending would invalidate the
  later `extraIndex` values the loop already captured (each removal shifts everything after it
  down by one).
- **Fixed a real, separately-reported bug: deleting/adding a point via the "Add/remove point"
  tool left a stale `orientationSelectedIndices`/`orientationOverrides` pointing at whatever
  points now occupy the OLD index positions** ("if i box select some then delete with the
  point add/delete then the selection shifts to the next 2 points") -- every later sample's
  index shifts by one when a point is added/removed, exactly the same index-invalidation
  hazard steps/seed/distribution changes already guard against. Fix: `addRemoveSamplePointAt`
  now clears both, identically to those existing listeners.
- **Removed the "(N point(s) selected)" suffix from the Direction card's label** -- reported
  directly ("why do we need to know how many points are selected??"); the selection is already
  visible as cyan rings on the points themselves, so the label no longer needs to restate it.
- **Renamed the orientation "Show" checkbox to "Enable"** -- reported directly: it isn't a
  local preview toggle, it's the real `orientation:` config option, and Rerun's live overlay
  during actual recording won't draw direction arrows unless this is on too. The old label
  read as "preview-only," which undersold what flipping it actually controls.
- **Four small calibration-tool bugs, all reported directly from actual use:** (1) Workspace
  setup's edge handles hit-tested distance-to-MIDPOINT only, so a long edge (typically
  top/bottom in a landscape frame) was only grabbable within ~15px of its exact center while
  a short edge (typically left/right) was grabbable almost anywhere along it -- read as "only
  side edges scale." Fixed by hit-testing distance-to-the-whole-SEGMENT
  (`pointToSegmentDistance`, already used elsewhere) instead, so any of the 4 edges is
  grabbable anywhere along its length. (2) Delete/Backspace removed box-selected sample
  points even while "Lock pattern" was on -- `patternLocked` already blocked every other
  sample mutation (move/add/remove) but the keydown handler's delete branch had no such
  guard; added one. Cut deletion stays unguarded by `patternLocked` on purpose -- cuts are
  scene geometry, not pattern samples, matching `patternLocked`'s existing, narrower scope.
  (3) A manually-added sample point (the "+" tool) was pushed straight into
  `extraSamplePoints` with no validity check at all, unlike every generator-produced point --
  so it could land inside a cut and just sit there ("new points do not adapt to cut outs").
  Fixed via `relocateNewPointIfInvalid`, reusing the same closed-form-clamp-then-radial-search
  (`relocateIfInvalid`) the grid/radial generators already use, falling back to the original
  point only if no valid spot exists nearby at all. (4) A cut shape dragged into
  (near-)zero area (an accidental click-without-drag) had no real interior, so
  point-in-polygon/point-in-circle could never match it -- it could be seen but never
  selected, and therefore never deleted, by clicking its body. `pointInCutShape` now treats
  any cut whose bounding-box's longest side is under 6 canonical units as degenerate and
  hit-tests a 10-unit click radius around its centroid instead.
- **Three follow-up reports on the round above, each pointing at a gap the first pass left
  open rather than a regression:** (1) "only side edges scale" wasn't fully fixed by the
  segment-distance hit-test alone -- the user also wanted dragging a side to move ONLY along
  its own axis (left/right edges: x only, never drifting y; top/bottom: y only) instead of
  free-form following the mouse, plus a way to scale the WHOLE quad at once, not one edge at
  a time. `findCameraCornersTarget` now classifies an edge as vertical/horizontal once, at
  grab time (`Math.abs(a[1]-b[1]) > Math.abs(a[0]-b[0])`), and the drag handler zeroes
  whichever axis doesn't apply; a new handle at `quadCentroid(cameraCorners)` (a small
  diamond marker, "corners" tool only) scales all 4 corners together about that centroid,
  by `scale = distance(mouse, centroid) / baseDist` (`baseDist` = each corner's own distance
  from centroid, captured once at drag-start, not recomputed incrementally -- avoids drift
  the same way the existing `sampleGroup` drag already does). Checked AFTER the fitted
  circle's own center handle in priority, since a circle centered in the workspace (common)
  puts its center handle exactly on the quad's centroid too -- editing that specific circle
  should still win the click there. (2) Fix (3) above only stopped a NEW point from being
  added inside a cut -- it never touched what happens when a cut is drawn (or a cut/the
  active circle is reshaped) OVER an already-existing manual point or override, which just
  sat there untouched ("if i create a cutout over added points they still remain there").
  Fixed by `revalidateManualPoints()` (re-maps `cutPolygonsCanonical` fresh, then re-runs
  `relocateNewPointIfInvalid` over every entry of `extraSamplePoints`/`sampleOverrides`),
  called after every cut commit (rectangle/circle/polygon) and after any `"shape"`-kind drag
  ends (covers both cut edits and the fitted circle's own edits) -- the generator's own
  base points already self-heal every render, so only these two fixed-coordinate stores
  needed an explicit hook. (3) Fix (4) above made a degenerate cut's BODY click-selectable,
  but never stopped it from ALSO being grabbed as a handle first -- a near-zero-area shape's
  handles (e.g. all 4 rectangle corners) sit virtually on top of each other, and handle
  hit-testing is checked before body-select in the "select" tool's priority order, so a click
  there almost always grabbed (and dragged) a handle instead of ever reaching the body-select
  branch ("i can drag it around but...cannot delete it"). `getAllCutHandles()` now skips
  every degenerate cut's handles entirely, so the click falls through to body-select.
- **Three more follow-ups on the same area, again pointing at gaps rather than regressions:**
  (1) the centroid scale-handle's fix above ("starts small every time") was incomplete --
  `scale = distance(mouse, centroid) / baseDist` still snapped the quad down hard at the
  START of every single drag, because the click that grabs the handle is itself always
  close to the centroid (within the 20px hit threshold), so its OWN raw distance is already
  near zero. Fixed by anchoring to the grab point instead of the centroid directly:
  `scale = 1 + (distance(mouse, centroid) - distance(grabPoint, centroid)) / baseDist`, so
  the gesture starts at exactly 1x and grows/shrinks smoothly from there. "Keep the state
  memorized" falls out for free from something already true of the handle's hit-test: it
  snapshots `corners`/`baseDist` from the CURRENT (already-edited) `cameraCorners` at every
  new mousedown, not from some original -- so a second scale gesture compounds on top of the
  first instead of resetting. (2) "the new points still ignore the cut" turned out not to be
  about adding NEW points (verified working) or drawing a cut over an EXISTING one (also
  verified working, see the bullet above) -- it was dragging an existing point INTO a cut:
  `sample`/`sampleGroup` drags wrote straight into `sampleOverrides` with zero validity
  check, unlike every other way a point's position changes in this tool. Fixed by extending
  the same `revalidateManualPoints()` hook (already called after a `"shape"`-kind drag ends)
  to also fire after `"sample"`/`"sampleGroup"` drags end. (3) Added click-to-delete for
  sample points: a plain click (movement under 2 canonical units between mousedown and
  mouseup) on a single point in the "select" tool now deletes it directly via the existing
  `addRemoveSamplePointAt(null, index)` removal path, instead of requiring box-select then
  Delete/Backspace every time ("click points to remove not just box select"). Distinguished
  from an intentional move by comparing the point's position at mouseup to a `startCanonical`
  snapshot captured in `startSelectDrag` at mousedown; an actual drag (movement over that
  threshold) is unaffected and still just relocates the point as before.
- **Three corrections to the round above, each a real miss rather than a regression:**
  (1) the centroid scale-handle's "anchor to the grab point" fix still couldn't scale DOWN
  at all -- placing the handle exactly AT the centroid means its own distance-from-centroid
  is always ~0, and that distance can never go negative, so the scale (`1 + (d - d0) /
  baseDist`) could only ever grow ("i cant scale it down since the point is in the centre,
  only up", reported directly). Fixed by moving the handle itself off-center -- a new
  `scaleHandlePosition()` returns a point 60% of the way from the centroid toward corner
  index 1 ("near the top right, inset so it stays in bounds", per the report) -- giving it a
  real, non-zero rest distance from the centroid (`restDist`) to anchor the scale formula to
  instead of the grab point. Now dragging toward the centroid (less than `restDist`) shrinks,
  dragging away grows, and "memorization" still holds because `restDist`/`baseDist`/`corners`
  are all re-derived from the CURRENT geometry at every new grab, same as before. (2) "the
  new points still ignore the cut" was never about adding a point or drawing a cut in
  isolation (both verified working) -- it was that the previous fix only hooked specific
  mutation call sites (cut create, cut/circle handle drag end, sample drag end), and at least
  one real path was still missed. Rather than keep chasing individual call sites,
  `revalidateManualPoints()` now runs unconditionally at the top of every `renderAll()` --
  exactly like the generator's own points already self-heal every render -- which by
  construction can no longer miss a path, since literally everything that changes the scene
  re-renders. Every scattered explicit call site was removed as redundant. This also surfaced
  a real, separate correctness issue: it had been using the FULL `isValidCandidate` (circle +
  bounds + cuts) for `sampleOverrides`, but an existing, intentional test asserts a dragged
  override CAN sit outside the fitted circle on purpose (an operator's manual placement is
  meant to be exact, not clamped) -- so overrides now only ever get pushed out of CUTS
  specifically (`isInsideAnyCut`/`relocateOverrideIfInCut`, no circle/bounds involved at all),
  while brand-new `extraSamplePoints` keep the full check, unchanged. (3) misread "click
  points to remove not just box select" as "click to DELETE" -- it meant "click to SELECT,"
  the same membership-toggle the box-select drag already does, just for one point at a time.
  Reverted the delete-on-click behavior and replaced it with the toggle, reusing the exact
  selection-toggle logic the box-select branch already had.
- **Calibration page layout/UX pass: more breathing room within the two top cards, undo/redo
  relocated, and Independent auto-forced for a single arm.** (1) The "cramped" report led to a
  new `.subsection`/`.subsection-label` pattern: each card is now visually divided into
  labeled groups (Camera / Arms / Workspace edit tools in card 1; Pattern / Direction in card
  2) with a border + 16px gap between them (`.subsection + .subsection`), replacing the old
  inline `<span class="toolbar-divider" style="width:100%...">` hack. `.panel` padding and
  `.toolbar-group-inner` gaps were also bumped (14px-\>20px, 6px-\>9px) for more air generally.
  (2) Undo/redo moved from the orthographic view's header to the LEFT edge of the camera
  view's own floating toolbar (before the hamburger toggle) -- "move the redo undo to the
  left of the toolbar," consolidating them with the rest of the editing tools instead of
  living in the other view's header. (3) `independent` is now forced `true` whenever
  `arms.length === 1` (`updateIndependentAvailability()`, called at the top of every
  `renderAll()`): with only one arm, combined vs. independent sample identically, but
  combined mode renders every sample dot (new ones included, since color is computed once for
  the whole set, not per-point) in a neutral color instead of that arm's own -- which reads as
  "wrong color" the moment there's only one arm to begin with (reported directly, along with
  "whatever new points i spawn...it should spawn with that color," which the same fix covers
  for free since sample-dot color was already uniform across the whole set in independent
  mode). The checkbox is disabled in this case since toggling it cannot change anything.
  `independentUserPreference` tracks the user's actual last choice separately from the
  forced `independent` value, specifically so a forced `true` at 1 arm doesn't get mistaken
  for the user's real preference and leak forward once a 2nd arm is added back -- adding an
  arm restores whatever combined/independent choice was in effect before, rather than always
  landing on "independent" just because that happened to be the effective value at 1 arm.
- **`rescaleCanonicalShapes` was scaling `extraSamplePoints`/`sampleOverrides` by scale^2,
  not scale, whenever the live globals aliased one of `combinedSettings`/`arms[i]`'s own
  copy of the same field -- which they always do with exactly 1 arm, and often do otherwise
  too.** Reported directly as "when i scale the workspace the points do not scale
  proportionally" (they were scaling, just by the wrong, compounding factor -- e.g. a 2x
  workspace scale produced a 4x point scale). Root cause: `syncActiveArmFromGlobals` mirrors
  `combinedSettings`/`arms[activeArmIndex]` together by REFERENCE whenever `arms.length===1`
  (`other[f] = settings[f]`, not a copy), so `combinedSettings.extraSamplePoints` and
  `arms[0].extraSamplePoints` become the literal same array as the live `extraSamplePoints`
  global -- but the rescale function iterated `[combinedSettings].concat(arms)`
  unconditionally, scaling that one aliased array once per entry in the list. The existing
  `fittedCircle`/`sampleOverrides` handling already avoided this (by using the live globals
  directly for the active arm and per-arm stored copies only for inactive arms), but
  `extraSamplePoints` never got the same treatment, and `sampleOverrides`/`extraSamplePoints`
  also have a SEPARATE, less common gap the active/inactive split alone doesn't cover: in a
  2+-arm INDEPENDENT-mode session, `combinedSettings`'s own (currently not-live)
  `sampleOverrides`/`extraSamplePoints` were never rescaled at all, so switching back to
  combined mode later would show stale, out-of-proportion points. Fixed uniformly by
  deduplicating on REFERENCE IDENTITY (`rescaleArrayOnce`/`rescaleOverridesOnce`, tracking
  arrays/objects already processed in a `seenArrays`/`seenOverrideObjects` list) rather than
  reasoning case-by-case about which of `combinedSettings`/`arms[i]`/the live globals currently
  alias which -- correct for 1 arm, 2+ arms independent, and 2+ arms combined alike, with no
  per-case branching.
- **A plain click (no drag) with the box-select tool on empty space no longer clears
  `orientationOverrides` -- it now only deselects.** An earlier round made it clear overrides
  too, specifically so lingering yellow "has an override" rings would go away without having
  to find the separate Reset button -- but that meant clicking anywhere else right after
  editing a selection's direction config (a completely normal "I'm done, moving on" gesture)
  silently destroyed the edit, reported directly: "whenever i select some points to edit
  their direction config it doesnt get saved so whenever i click somewhere else it just goes
  back to the general config." Deselecting has to be a safe, non-destructive action; clearing
  override data now only ever happens via the explicit "Reset to default" button (which
  already correctly scoped itself to "selected points only, else everything," per its own
  comment -- that comment turned out to already assume this exact fix, suggesting the
  destructive-click behavior was a later regression against the original intent, not a
  deliberate tradeoff anyone re-confirmed).
- **The calibration page's top card layout changed from 2 cards side by side to 2 cards
  stacked vertically (Setup above Pattern/Rotation), with Setup's own 3 subsections
  (Camera/Arms/Workspace edit tools) arranged as columns instead of stacked rows -- both
  requested directly.** The outer `<div class="toolbar">` flex-row wrapper (which put the two
  cards side by side) was removed entirely -- the two `.panel` cards are now direct body
  siblings, stacking via each `.panel`'s own default `margin-bottom`, no extra CSS needed.
  A new `.setup-columns` class wraps Setup's 3 subsections in their own flex row (wrapping to
  stacked rows on a narrow viewport, same as every other flex-wrap group on this page);
  `.setup-columns > .subsection + .subsection` swaps the normal stacked-subsection look (a
  border ABOVE each one) for a vertical rule BETWEEN columns instead (`border-left` +
  `padding-left`, with `border-top`/`margin-top`/`padding-top` reset to none so the two rules
  don't both apply). The card's own title also shortened from "Camera / Arms / Workspace
  setup" to plain "Setup" (requested directly) now that the 3 subsection labels underneath
  already say what's in each column. "Direction" was renamed to "Rotation" throughout the
  visible UI (`orientationPreviewLabel`'s "Rotation"/"Rotation (combined)"/"Rotation (active
  arm)" text, the intro paragraph's "Pattern/Rotation," and one descriptive code comment) --
  the underlying `orientation*` identifiers (config field names, element IDs, JS variables)
  are intentionally unchanged, since the user only asked to rename the visible label, not the
  underlying `orientation:` config concept itself (which the calibration page's own "Enable"
  checkbox tooltip already explains is a real, named YAML field).
- **Phase 2 ships a precomputed sidecar artifact (JSON/parquet), not a live backend call.**
  The dataset-visualizer's panels are pure frontend (parquet via `hyparquet`) — the heavy
  Python computation has to happen ahead of time and get written next to the dataset, not be
  invoked on page load.
- **`calibrate.html`'s inline `<style>`/`<script>` were extracted into sibling
  `calibrate.css`/`calibrate.js` files (linked by relative path), with a section-index comment
  banner added to the top of `calibrate.js`.** Requested directly, ahead of adding the
  multi-object feature below, specifically so a ~3000-line single file would be easier for a
  human or an agent to navigate and edit going forward — pure extraction, byte-identical
  content otherwise (verified directly: `diff` against the original inline blocks before any
  further edits landed, plus the Node harness's full pre-existing assertion set re-run
  unchanged against the extracted file). `calibrate.py`'s `write_calibration_html` now copies
  both sibling files alongside the HTML (`_STATIC_SIBLINGS`), since `serve_and_open` only ever
  served `output_path.parent` — without this, the page would 404 on its own CSS/JS the moment
  `--output` pointed somewhere other than this source tree.
- **The calibration tool gained a multi-OBJECT layer — named, colored things placed onto
  subsets of the existing sample-point cloud — deliberately NOT a rename/generalization of
  "arms."** Investigated and rejected merging the two concepts: arms describe independently-
  calibrated REACH REGIONS that feed the shared sample cloud (the existing per-arm circle/
  extreme-points/pattern machinery, untouched by this feature); objects describe WHAT GETS
  PLACED on points already in that cloud. The two counts are unrelated by design — a bimanual
  rig might have 2 arms and 2 objects (one per gripper) with no relation between the numbers,
  or 1 arm and 3 objects (three things placed within one arm's reach across different
  episodes). `makeObject()` (`calibrate.js`, new "Objects" section) is structurally a
  stripped-down arm: same `orientationEnabled/orientationCount/.../orientationInitialAngle`
  field names and its own `orientationOverrides` (so it plugs directly into the EXISTING
  `captureOrientationInputsAsConfig`/`populateOrientationInputsFrom`/`orientationExportBlock`/
  `generateRotationAngles` with zero adapter code), plus `assigned` (sample-point indices into
  `samplePointsCanonical` — many-to-many: one point can hold several objects, one object can
  occupy many points). `currentDefaultOrientationSettings()` — what the Rotation card and its
  per-point box-select overrides actually edit — now resolves to the ACTIVE OBJECT once any
  object exists, falling back to the pre-existing arm/combinedSettings target otherwise;
  `loadOrientationTargetIntoGlobals()` (factored out of `loadActiveArmIntoGlobals()`) repoints
  the live `orientationOverrides` global and the form fields at whatever that resolves to,
  called on every arm switch (as before) AND on every active-OBJECT switch (new) — one seam,
  not two parallel ones.
- **No manual `level` field anywhere — stacking order at a shared point is derived
  automatically from ASSIGNMENT ORDER, not hand-typed.** The first version of this feature
  gave each object a manual integer `level`; reported directly as backwards ("the level
  should get automatically assigned to a point if more objects use it not manually"): an
  operator stacking blocks shouldn't have to invent and keep consistent a numbering scheme
  across however many objects/points happen to share a spot. Fix: `makeObject()` has no
  `level` field at all, only `assignedSeq` (an object → point map of `nextAssignmentSeq`'s
  value at the moment that assignment was made; `nextAssignmentSeq` is one monotonically
  increasing module-level counter). `objectStackAt`/`computeEpisodePlacements` sort the
  objects sharing a point by `assignedSeq[thatPoint]` ascending and hand back each one's
  0-based rank in that sorted order AS its `level` (first assigned = 0 = bottom) — there is
  deliberately no single "this object's level," only "this object's rank wherever it happens
  to be stacked," since the SAME object can rank differently at two different shared points
  depending on assignment order there. Un-assigning and re-assigning issues a fresh seq, so an
  object moved off and back onto a stack lands back on top, not wherever it originally was.
  `nextAssignmentSeq` is included in undo snapshots and restored via `Math.max` (never
  decreasing across a restore), so an undo/redo round-trip can't make a future assignment's
  seq collide with or sort before one already on the restored objects.
- **Assignment is "select first with the general tool, THEN click Assign/Unassign" — not a
  dedicated drag-to-assign tool.** The first version added a separate "Assign object" tool
  (`toolAssign`) an operator had to switch into before they could even see what they were
  selecting; reported directly as backwards ("first select using the general tool then assign
  for the object"). Fix: removed `toolAssign`/`assignBoxToggle`/the `"assignObject"`
  interaction branches entirely. The Objects card instead has two buttons,
  `assignSelectedBtn`/`unassignSelectedBtn`, that act on whatever `orientationSelectedIndices`
  currently holds — the EXACT SAME selection state the pre-existing "select" tool's box-select/
  click-to-select already populates for scoping the Rotation card's per-point overrides.
  Assigning is now just another action over that one selection, not a second, parallel
  selection mechanism — select with "select" (box-drag, or click one point), then Assign/
  Unassign. Two distinct buttons rather than one toggle, deliberately: a toggle's meaning is
  ambiguous the instant the selection is a MIX of already-assigned and not-yet-assigned
  points, which is the common case once an operator selects by region rather than one point
  at a time. `toggleObjectAssignment(pointIndex)` (issues/clears the `assignedSeq` entry) is
  the shared primitive both buttons call per selected index; unlike
  `orientationSelectedIndices` itself (deliberately NOT undo-tracked, pure UI focus), an
  Assign/Unassign click IS pushed through `recordUndo()` — this feature's actual deliverable,
  not a selection.
- **Episode model: each object sweeps its OWN sorted `assigned` list independently
  (`sortedAssigned[episodeIndex % length]`), the same `target_for_episode` index-modulo idiom
  `session.py`/this tool's own rotation-angle cycling already use** — decoupled per object on
  purpose, exactly like position-count and rotation-count are already deliberately decoupled
  elsewhere in this engine (see the rotation-arrow bullets above). Two or more objects whose
  *current* episode index resolves to the same point index stack (sorted by the automatic
  assignment-order rank above, bottom to top), fanned out by a small fixed canonical offset
  (`OBJECT_STACK_FAN_RADIUS`, `stackOffsetPosition`) so stacked dots/arrows render as visually
  distinct rather than exactly overlapping — purely cosmetic; the exported position is always
  the point's own un-fanned coordinate. Deliberately **even marginals only** (the user's
  explicit call): no joint decorrelation/collision solver across objects — each object's
  `assigned` subset is whatever the operator chose, and nothing here re-balances one object
  against another's choices.
- **Preview is BOTH an episode stepper and a marginal overlay, picked via a `previewMode`
  dropdown in the Objects card** — the marginal overlay (default) renders every object's full
  `assigned` cloud simultaneously, closest to this tool's pre-existing always-on preview
  style, for verifying overall coverage; the episode stepper (`computeEpisodePlacements`)
  renders the exact simultaneous multi-object scene for one specific episode index (with the
  rest of the cloud shown faint, for spatial context), for verifying the real recording
  sequence object-by-object. `maxEpisodeCount()` is the longest of any one object's own
  `assigned` list (shorter lists just cycle back to their own start sooner, never an error).
  With zero objects, the legacy single-target orientation preview (`orientationTipsCanonical`
  built from one shared arm/combinedSettings config, drawing every configured angle from every
  point uniformly) is completely untouched; with 1+ objects it's suppressed in favor of
  `objectStacksAtPoint`/`computeEpisodePlacements` instead, since "draw one tool-wide arrow
  set from every point" stops making sense the moment different points can hold different,
  independently-configured objects. `renderObjectPlacementsCamera`/`renderObjectPlacementsOrtho`
  are the camera/ortho-specific render functions (forward-mapping via the homography vs.
  scaling directly by `orthoScale`, mirroring every other dual-view render path in this file).
- **Index invalidation gets the PRECISE fix here, not the wholesale "just clear it" the
  pattern-level `orientationOverrides`/`orientationSelectedIndices` already settled on for the
  analogous hazard** — a deliberate departure, because object assignment is this feature's
  main deliverable (losing it on every single point add/remove would be a real regression,
  unlike losing a point's rotation override). A pure append (`extraSamplePoints.push`, the
  "+" tool's add case) never shifts any EXISTING final index, so no remap is needed at all; a
  removal (the "+" tool's remove case, or `deleteSelectedSamples`'s batch delete) calls
  `remapObjectAssignmentsForBatchRemoval(removedIndices)` first, which drops every object's
  reference to a removed index and renumbers every surviving reference — `assigned`, its
  `orientationOverrides`, AND its `assignedSeq` (a point's automatic level is meaningless once
  its index changes, so the seq recorded against it must follow the same renumbering) — down
  by however many removed indices preceded it, computed directly from the batch (correct
  regardless of removal order), not via repeated single-index shifts. A structural
  regeneration (steps/seed/distribution/count-mode changing — a genuinely different point set,
  where no remap is meaningful) still wholesale-clears every object's `assigned`/
  `orientationOverrides`, exactly like the pre-existing pattern-level fields, via the new
  shared `clearStaleSampleDerivedState()` (factoring out 4 previously near-duplicated listener
  bodies in the same pass).
- **Export: with 1+ objects defined, the legacy single-implied-object `pattern:`/`marker:`
  block is suppressed entirely in favor of a top-level `objects:` list (`objectsExportBlock`)
  — one entry per object with explicit, homography-forward-mapped pixel-space `points:`
  (`shape: points`, sorted-assigned order), a marker color, and the SAME
  `orientationExportBlock(o)` every arm/combinedSettings export already used.** Manual
  per-point assignment can't be expressed as an analytic pattern the way `sector`/`union` are,
  so it's exported as the explicit point list directly. **At the time this bullet was first
  written, engine support for `shape: points` (and a parallel per-point `levels:` array this
  export originally also emitted) was a documented follow-up, not built yet — that follow-up
  has since landed, but NOT as a `levels:` array; see the "Behavioral randomization knobs"
  entry below for why that array was dropped in favor of a top-level `stacks:` mechanism
  instead, and for the real `shape: points`/`sequencing`/`jitter_px`/`presence` schema that
  actually shipped.** With ZERO objects (every session before this feature, and the default
  for a new one), the `objects:` branch is unreached — the pre-existing independent/combined
  `pattern:` export code is byte-for-byte untouched, verified directly via a Node-harness
  assertion that removes every object after exercising the feature and confirms the export
  reverts to exactly `shape: sector` with no `objects:` block.
  - Verified via an ad-hoc Node harness (this project's established, not-committed-to-the-repo
    convention for `calibrate.js` — see the harness-based verification notes throughout this
    file): object add/remove isolation and palette-color distinctness; automatic level
    derivation (the FIRST-assigned object at a shared point ranks 0/bottom, the second ranks
    1/on top, purely from assignment order, with re-assigning moving an object back to the top
    of that point's order) and distinct fanned positions; `computeEpisodePlacements`'
    independent per-object cycling at two different episode indices;
    `remapObjectAssignmentsForBatchRemoval`'s index-drop/shift correctness across `assigned`,
    `orientationOverrides`, AND `assignedSeq` together; export gating in both directions
    (objects present → `objects:`/`shape: points`/no legacy block; objects removed → reverts
    to byte-identical `shape: sector`); undo/redo round-tripping an assignment with
    `nextAssignmentSeq` never decreasing across the restore; and, driving the REAL `mousedown`/
    `mousemove`/`mouseup`/button-`click` listeners end-to-end (not the underlying functions in
    isolation): a real box-drag with the EXISTING "select" tool populating
    `orientationSelectedIndices`, confirming selecting alone does NOT assign anything, then a
    real click on the Assign button assigning every selected point, then Unassign clearing it.
    The full pre-existing 14-assertion suite (Part-A extraction correctness) was re-run
    alongside these and stayed green throughout.
- **Behavioral randomization knobs (sequencing, positional jitter, presence/dropout, and
  randomized stack levels) — requested as a follow-up brainstorm on "what, besides marginal
  and episode preview, makes multi-object data collection more sample-efficient," with the
  explicit framing of "even marginals over MORE axes than position": each object should land
  on each of its own points, AND each level of a stack it's part of, roughly equally across a
  session — the data-collection analog of domain randomization (a policy that only ever sees
  red-on-blue never learns blue-on-red).** All four knobs are REAL `config.py`/`pattern.py`/
  `session.py` fields the live overlay consumes — not preview-only — and all four default to
  today's EXACT prior behavior when unset (lockstep cycling, no jitter, always present,
  declaration-order levels), so every config written before this feature is byte-for-byte
  unaffected.
  - **`pattern.py` gained `sequence_index(episode_index, length, mode, object_ordinal,
    num_objects, seed)`**, replacing the bare `episode_index % length` lockstep cycle
    `target_for_episode` always used for position, with two decorrelation alternatives —
    `"phase"` (a fixed, evenly-spaced offset per object — cheapest, no randomness) and
    `"shuffled"` (each object visits its own points in its own seeded-random order — still a
    permutation, so the even-marginal guarantee holds exactly) — because two objects with
    EQUAL-length patterns moving in lockstep always pair the same way (objA-here always means
    objB-there) — exactly the spatial correlation this whole project audits against, just at
    the object-pairing level instead of single-object position. `ObjectConfig.sequencing: str
    = "lockstep"` selects the mode; `seed` is read from the object's own `PatternConfig.seed`
    (see the `shape: "points"` bullet below for why that field had to be widened to accept
    one). A THIRD mode, `"stratified"` (striding by an integer coprime with `length`, distinct
    per object), shipped briefly in this same pass and was removed before ever reaching a
    release — see the dedicated bullet below for why, and for two real bugs (one a genuine
    JS/Python parity break, not just a design question) found while re-examining `"phase"` in
    response to that same review.
  - **`pattern.py` gained `apply_jitter(point, jitter_px, seed, episode_index,
    object_ordinal)`** — seeded sub-cell positional noise (area-uniform within a disk, the
    same squared-uniform-radius trick `generate_sector_points` already relies on), applied
    fresh every episode so a policy doesn't overfit to the exact pixel a target sits at. The
    single highest-value, lowest-effort knob of the four, and fully independent of the other
    three. `ObjectConfig.jitter_px: float = 0.0` (no-op default).
  - **`pattern.py` gained `is_present(episode_index, presence)`** — whether an object is shown
    at all this episode, true for only a configurable fraction (`presence` in `[0, 1]`) of
    episodes, for clutter/distractor robustness training. Evenly spread via a running-total
    (Bresenham-style) test (`floor((i+1)*presence) - floor(i*presence) == 1`) rather than an
    independent per-episode coin flip — unbiased in aggregate, but a coin flip can still streak
    many consecutive absences by chance, which the running-total test can't. Needs no total
    episode count up front, matching the session's own "runs until interrupted" model.
    `ObjectConfig.presence: float = 1.0` (always present, the default).
  - **`pattern.py` gained `level_order(stack_size, episode_index, strategy, seed)`** — the
    displayed level (z-order, `0` = bottom) for each declaration/assignment-order slot in a
    stack this episode, the level-axis equivalent of `sequence_index` above. `"fixed"`
    (default) is today's exact behavior (identity — order never changes). `"cycle"`/
    `"balanced"` are both Latin squares (any one slot visits all `stack_size` levels, all
    distinct, every `stack_size` episodes) — `"cycle"` rotates the identity ordering (no
    randomness), `"balanced"` rotates a SEEDED ordering instead, so which level an object
    starts at is itself randomized, not always slot `i` starting at level `i`. `"shuffle"` is a
    plain independent seeded permutation per episode — evenly distributed in aggregate, but
    without the Latin square's per-window guarantee.
  - **All four knobs share one small, deliberately portable seeded PRNG, `_mulberry32` in
    `pattern.py` / `mulberry32` in `calibrate.js` — NOT stdlib `random.Random` (Mersenne
    Twister), which has no portable JS equivalent.** The calibration tool's episode-stepper
    preview has to show EXACTLY what a real session would record for these seeded knobs, and
    this project's established way of guaranteeing that is porting the SAME algorithm into
    both languages and verifying the outputs match bit-for-bit, not just "looks equivalent."
    `calibrate.js` already HAD a `mulberry32` (used for the `"random"` sampling distribution
    and the radial spiral's rotation offset) — reused as-is rather than adding a second,
    duplicate implementation, after explicitly verifying it's bit-for-bit identical to the new
    Python `_mulberry32` by running both with seed `0` and diffing the first two draws
    (`0.26642920868471265`, `0.0003297457005828619` — identical to 15+ significant digits).
    `_combine_seed`/`combineSeed` (a masked multiply-accumulate hash) folds several integers
    (seed, episode index, object ordinal) into one 32-bit seed for these functions, masked at
    EVERY step rather than only the final result — needed because an unmasked multiply-
    accumulate can exceed JS float64's exact-integer range for large inputs, which would
    silently desync the two languages' outputs without that discipline.
  - **Stacks come into existence two ways — AUTHORED sites (`OverlayConfig.stacks:
    list[StackConfig]`, a fixed `point` + named `objects` + `level_strategy`/`seed`) and
    EMERGENT coincidence (two objects' independently-sequenced positions happening to land on
    the same point this episode, needing no config entry at all) — because they serve
    different dataset shapes.** An authored site guarantees a stack/tower exists every episode
    (a 3-block assembly task needs this); emergent stacking falls out for free from the
    Objects-layer's pre-existing "assign multiple objects to overlapping points across their
    own independent multi-point patterns" authoring model, with no new schema needed for it.
    `session.py`'s `show_targets` computes every present object's BASE (pre-jitter) position
    first (an authored-stack member's position is its stack's fixed `point`, full stop,
    overriding its own pattern entirely; everyone else uses `sequence_index` against their own
    pattern), GROUPS objects by exact base-position equality (this is what finds emergent
    stacks), THEN runs `level_order` per group (authored → its own `level_strategy`/`seed`;
    emergent → always `"fixed"`, since there's no `StackConfig` to read a strategy from), and
    only THEN applies `apply_jitter` per member. Grouping deliberately happens on the
    PRE-jitter position: jitter is independent per-member cosmetic noise, not part of what
    makes a group a group — grouping on POST-jitter positions (as an earlier wording of this
    feature's own plan literally said) would make emergent stacks vanish almost the instant
    any member has `jitter_px > 0`, since two independent continuous offsets essentially never
    land on the exact same point; recognized and deliberately overridden during implementation,
    not an oversight carried through from the plan.
  - **`rerun_client.py`'s `log_target` gained `level`/`stack_size` parameters, fanning a
    stacked marker out by a small fixed radius keyed on `level` (not declaration-order RANK) —
    mirroring the calibration tool's pre-existing `OBJECT_STACK_FAN_RADIUS`/
    `stackOffsetPosition` formula exactly (`STACK_FAN_RADIUS = 7.0`,
    `_stack_offset_position`).** Keying the fan angle on the (possibly randomized) LEVEL rather
    than the stable rank means a stacked marker visibly moves to a different fan slot whenever
    its level changes episode to episode — the intended visual confirmation that the stacking
    order actually changed, not an accident of using whichever value was more convenient. An
    absent object (`is_present` false) calls the new `clear_target` (a recursive `rr.Clear`)
    instead of skipping the call entirely — skipping would leave the PREVIOUS episode's marker
    visibly stuck on screen, since re-logging-replaces-prior-data only works when something
    actually re-logs.
  - **`config.py` gained `shape: "points"` on `PatternConfig`** (`points: list[Point]`,
    explicit/manually-authored positions, returned as-is by `build_pattern` with no sampling —
    what the calibration tool's per-object point assignment exports) **and `StackConfig`**
    (`point`, `objects: list[str]` validated to exist and be `variable: true`, `level_strategy:
    str = "fixed"`, `seed: int = 0`) **on a new `OverlayConfig.stacks: list[StackConfig]`.**
    `shape: "points"` also accepts an optional `seed: int = 0` — NOT used for sampling (there's
    nothing to sample) but as the seed `sequence_index`'s `"shuffled"` mode and `apply_jitter`
    draw from for that object, found as a real gap mid-implementation: every other shape
    already had a `seed` field for its own sampling, so reusing `PatternConfig.seed` for the
    NEW per-object randomization knobs was free for them, but `shape: points` had never had a
    reason to accept `seed` at all until these knobs existed for it to feed.
  - **The calibration tool's Objects card gained per-object `Sequencing`/`Jitter px`/
    `Presence`/`Seed` fields (synced via the same `renderObjectChips`/switch-object pattern
    `objectColorInput`/`objectNameInput` already used) and one GLOBAL (not per-object, this
    pass — per-stack override deferred) `Stack level strategy`/`Stack seed` pair, plus an
    Animate/Play button on the episode stepper (a plain `setInterval` advancing
    `currentEpisode`, the one pure-preview addition in this whole feature — it has no engine
    counterpart because "auto-advance the local preview" has no recording-time analog).** The
    MARGINAL overlay's level badge deliberately stays the plain assignment-order rank
    regardless of the new strategy setting — there's no single "this episode" for a static,
    always-on view to vary it across — only the EPISODE STEPPER's `computeEpisodePlacements`
    actually applies `sequenceIndex`/`isPresent`/`applyJitter`/`levelOrder`, mirroring
    `session.py`'s own grouping order (group on pre-jitter base position, THEN jitter each
    member) exactly.
  - **Export: `objectsExportBlock` emits each non-default knob (`sequencing`, `jitter_px`,
    `presence`, the pattern's own `seed`) only when it differs from the default, so an object
    that never touched these fields exports byte-identically to before they existed — and the
    OLD parallel `levels:` array this function used to emit (back when engine support was
    still a documented follow-up, see above) is gone entirely, replaced by a NEW
    `stacksExportBlock`.** That function only emits an authored `stacks:` entry for a group of
    2+ objects whose ENTIRE `assigned` list is exactly one shared point — a deliberately
    NARROWER case than everything the calibration tool's per-object multi-point assignment can
    represent. An object assigned to several points that only INCIDENTALLY overlaps another
    object's point at some episodes is left to the engine's own emergent stacking instead
    (no config entry, "fixed" level only): exporting an authored pin for that case would
    incorrectly force the two objects together EVERY episode, not just the ones where their
    independent per-object cycles actually happen to coincide — a real correctness trap
    identified and deliberately avoided while designing the export, not found via a test
    failure.
  - Verified via ~68 new Python tests across `test_pattern.py` (the four new functions plus
    their private helpers — `_mulberry32`/`_combine_seed`/`_seeded_permutation`/
    `_coprime_stride` — covering determinism, the even-marginal/Latin-square guarantees, and a
    pinned exact-value regression test against the algorithm itself), `test_config.py`
    (`shape: points`/`seed`, the three `ObjectConfig` knobs, and `StackConfig`/`stacks:`
    parsing/validation), and new `test_session.py`/`test_rerun_client.py` files (default-knobs
    back-compat byte-for-byte; presence emitting `clear_target` on absent episodes; jitter
    matching `apply_jitter` called directly; an authored stack's shared point/levels; a
    `"balanced"` strategy actually varying across episodes; an EMERGENT stack forming only once
    two independent patterns coincide; a stacked object's orientation cycling every episode
    rather than once per its own pattern length; the fan-by-level offset and recursive-clear
    behavior in isolation) — 285 Python tests total (`pytest tests/ -q`). JS side verified via
    58 new ad-hoc Node-harness assertions (same
    convention as the rest of this file): bit-for-bit parity of every ported function against
    the Python outputs above; `computeEpisodePlacements` wiring for sequencing/presence/jitter/
    levels including the Latin-square check and the jitter-doesn't-break-grouping case; export
    gating in both directions for the new per-object lines AND the pinned-vs-incidental
    `stacks:` distinction; real DOM-event wiring for every new input/select/button, including
    Play/Pause toggling and an active-object switch correctly reloading its own knobs back into
    the form. `examples/stacking.example.yaml` (a 3-block authored stack with `balanced`
    levels plus an independently-swept distractor exercising all three remaining knobs) is the
    new example config for this feature, and was run through `load_config` plus a live
    `run_session` smoke (mocking only the dataset/Rerun I/O) to confirm the documented behavior
    end-to-end, not just unit-level.
- **`"stratified"` sequencing was removed, and `"phase"` got a real correctness fix plus
  eager validation, after being asked directly "what is the point of anything but the shuffle
  sequencer" — a fair challenge that, on honest re-examination, exposed one weak design choice
  and two real, independent bugs (one of them a genuine cross-language correctness break, not
  just a missing nice-to-have).**
  - **`"stratified"`'s own justification didn't hold up:** its claimed benefit (decorrelated
    visit order without needing the RNG, in the same spirit as the radial pattern's golden-
    angle choice) is something `"shuffled"` already does at least as well — and unlike a true
    random permutation, a FIXED coprime stride is pure modular arithmetic, so it risks
    resonating with whatever periodic structure the underlying point grid already has, which
    randomness can't. It was added because the original brainstorm proposed it, not because it
    solved something `"shuffled"` didn't. Removed entirely: `pattern.py`'s `_coprime_stride`
    and the `"stratified"` branch of `sequence_index`, `config.py`'s `_SEQUENCING_MODES` entry,
    and `calibrate.js`'s `coprimeStride`/`gcd` and its `sequenceIndex` branch and HTML
    `<option>`. `ObjectConfig.sequencing`'s allowed values are now just `"lockstep"`/`"phase"`/
    `"shuffled"`.
  - **`"phase"`'s actual, defensible reason to exist: unlike `"shuffled"`, it gives a
    GUARANTEED, not probabilistic, non-collision property** — two objects of the same
    `length` with distinct offsets will NEVER share an index, for as long as the session runs,
    whereas two independently-shuffled equal-length objects still coincide on roughly
    `1/length` of episodes in expectation (by chance, possibly worse for an unlucky seed).
    That's the right tool specifically when coincidence is a hard constraint (e.g. a literal
    collision-safety requirement between two simultaneously-active targets), not just a
    coverage nicety where `"shuffled"` is the better general default.
  - **Bug 1 (a real correctness gap, found while writing the justification above, not from a
    test failure): the offset formula's guarantee was CONDITIONAL, not automatic, and nothing
    enforced the condition.** `offset(o) = round(o * length / num_objects)` maps `num_objects`
    ordinals into only `length` possible values — if `num_objects > length`, the pigeonhole
    principle FORCES at least two ordinals to share an offset, and those two objects would then
    collide every single episode, forever: the exact opposite of what `"phase"` exists to
    prevent, and silently so. Fixed by adding `pattern.py`'s `find_phase_offset_collisions
    (length, ordinals, num_objects)` (pure pigeonhole arithmetic, no `episode_index` involved,
    since the offset never changes across a session) and calling it eagerly from a new
    `session.py` `_validate_phase_sequencing`, right after `patterns` is built (the first point
    real per-object lengths are known) and before connecting to Rerun — grouping every
    `"phase"`-sequenced object by its pattern's length, and raising a `ValueError` naming the
    first colliding pair the moment any group's offsets aren't pairwise distinct. Consistent
    with this project's existing "a conflict in something deterministic is a config error: fail
    loudly, name the offending thing" rule (see the arc/line-vs-exclude_zones bullet earlier in
    this file) — a guaranteed-forever collision should never just happen silently.
    `calibrate.js` got the identical check (`findPhaseOffsetCollisions`/`phaseOffset`, sharing
    the same JS/Python parity discipline as everything else in this section) wired to a new
    `phaseWarning` note in the Objects card, updated every `renderAll()` — so the calibration
    tool warns an operator BEFORE they ever run a session that would hit this, not just after.
  - **Bug 2 (a genuine JS/Python PARITY break, found by actually testing the claim rather than
    assuming `round()`/`Math.round()` are equivalent): Python's builtin `round()` uses
    round-half-to-EVEN ("banker's rounding"), while JavaScript's `Math.round()` always rounds
    half AWAY FROM ZERO — confirmed disagreeing, not just theoretically different:
    `round(2.5) == 2` in Python vs `Math.round(2.5) === 3` in JS; same for `0.5`→`0` vs `1` and
    `4.5`→`4` vs `5`.** The phase-offset formula (`o * length / num_objects`) lands on an exact
    `.5` boundary for ordinary, non-adversarial combinations (e.g. `length=3, num_objects=2`
    gives `1.5` for `o=1`) — meaning the calibration tool's preview could show a DIFFERENT
    offset than the live session would actually use for the identical config, silently, for
    perfectly normal inputs, not just edge cases. This is exactly the class of bug this
    project's "verified bit-for-bit, not just looks equivalent" parity discipline exists to
    catch — found here by actually running both languages' rounding on the same values and
    diffing, not by assuming the port was faithful. Fixed with `pattern.py`'s
    `_round_half_up(x) = math.floor(x + 0.5)`, which is mathematically identical to
    `Math.round()` for the non-negative values this formula ever produces — `calibrate.js`
    needed NO equivalent helper at all, since its native `Math.round()` already behaves the way
    Python had to special-case to match it (confirmed directly, not assumed: `Math.floor(2.5 +
    0.5) === Math.round(2.5)`, etc.). `_phase_offset`/`phaseOffset` (the one place this
    formula is computed) are the shared, single source of truth `sequence_index`/
    `find_phase_offset_collisions` and their JS twins all route through, so the fix and the
    validation above can never drift apart from what's actually used to place a target.
  - Verified via 5 new `test_pattern.py` tests (the `_round_half_up`/`Math.round`
    disagreement pinned directly with the exact values above; `_phase_offset` distinctness for
    a safe case; `find_phase_offset_collisions` returning empty for distinct offsets and a
    real pigeonhole pair for a forced-collision case) and 3 new `test_session.py` tests (a
    safe `"phase"` config running cleanly end-to-end; a forced-collision config raising a
    `ValueError` naming both objects; confirming two DIFFERENT pattern lengths are never
    compared against each other, since only same-length objects can possibly collide on a
    shared offset), net of removing every `"stratified"`-specific test — 289 Python tests
    total. JS side re-verified with 6 new Node-harness assertions (`phaseOffset`/
    `findPhaseOffsetCollisions` matching Python exactly, INCLUDING at the exact-half rounding
    boundary that would have exposed Bug 2 had it not been fixed; the `phaseWarning` note
    correctly appearing for a forced-collision scene and staying hidden for a safe one), net
    of removing the `"stratified"`-specific ones — 62 JS assertions total, all passing.
- **`"phase"` sequencing was removed entirely, and the authored `stacks:` config block was
  replaced with a SCENE-LEVEL `co_location` toggle, after a direct report that the feature
  conflated two genuinely different intents: "2+ objects in the same place at the SAME
  episode" (a real stack) vs. "2+ objects in the same place at DIFFERENT episodes" (a shared
  point used at different times, where same-episode coincidence is an accident to avoid, not a
  stack) — `"phase"`'s per-object guarantee and the authored single-site `stacks:` block each
  only ever addressed one narrow corner of this, not the actual ambiguity.**
  - **Stacking is no longer authored.** `config.StackConfig`/`OverlayConfig.stacks` are gone.
    A stack is now ALWAYS derived directly from 2+ objects' own `pattern: {shape: points}`
    lists sharing a literal coordinate — exactly what `vizaudit-calibrate`'s per-object point
    assignment already produces when two objects are assigned the same point, so authoring a
    *separate* `stacks:` block for the common case (an object pinned to exactly one shared
    site) was redundant with what the tool already exports naturally. The previous design's
    real gap wasn't the lack of an authored block — it was that *coincidental* multi-object
    sharing had no way to express "keep these apart instead," covered by the next point.
  - **New scene-level `OverlayConfig.co_location: "stack" | "keep_apart"`** (default
    `"stack"` — today's prior coincidence-stacking behavior, byte-for-byte). `"keep_apart"`
    actively avoids same-episode co-occupancy: `pattern.py` gained
    `resolve_keep_apart(natural_indices, assigned_lists, episode_index) ->
    (resolved_indices, residual_collision_ordinals)` — a pure, deterministic, RNG-free
    greedy resolver. In DECLARATION order, the first object wanting a point keeps it; each
    later object wanting an already-taken point advances through its OWN assigned points
    (cyclically, starting at `episode_index % length`, so it doesn't always retry the same
    alternate first) to the first currently-free one; if none of its own points are free, it
    keeps its natural (colliding) index and is reported as a RESIDUAL collision rather than
    silently overlapping. Objects with disjoint assigned-point sets (the common case) are
    mathematically untouched — nothing here perturbs an object that never collides with
    anyone. `session.py`'s `show_targets` calls this once per episode (only when
    `co_location == "keep_apart"`), then runs the SAME coincidence-grouping it always did on
    the (possibly nudged) resolved positions — a residual collision therefore gracefully
    DEGRADES to rendering as a visible, leveled stack (not a silent, indistinguishable
    overlap), and `logger.warning(...)` names the colliding objects and episode.
  - **`OverlayConfig.level_strategy`/`level_seed` moved from per-`StackConfig` to SCENE-LEVEL**
    (one setting for the whole session, matching the calibration tool's pre-existing global
    controls) — used whenever 2+ objects coincide, whether that's a "stack"-mode coincidence
    or a "keep_apart"-mode residual fallback. Per-stack override remains a possible future
    refinement, not built.
  - **`"phase"` sequencing is gone from `sequence_index`** (`_SEQUENCING_MODES` is now just
    `"lockstep"`/`"shuffled"`) — along with `_round_half_up`/`_phase_offset`/
    `find_phase_offset_collisions` and `_validate_phase_sequencing` in `session.py`. Its one
    real justification (a guaranteed, not probabilistic, non-collision property between two
    *separate* objects' independent sweeps) conflated object-PAIR decorrelation with the
    distinct, scene-level "should co-located objects ever coincide at all" question that
    `co_location` now answers directly and unconditionally — `"phase"`'s guarantee was also
    only ever conditional (pigeonhole-breakable once `num_objects > length`, the bug fixed
    earlier the same day this redesign happened), whereas `resolve_keep_apart` has no such
    failure mode: it always reports rather than silently violating its contract.
  - **Calibration tool**: `coLocationSelect` (Stack / Keep apart) replaces the per-object
    "Phase offset" `<option>`; toggling to `"keep_apart"` disables the (still scene-level)
    stack-level-strategy/seed controls, since they only apply to an actual stack or a residual
    fallback. `computeEpisodePlacements` was restructured to return `{sites, residual}` —
    `sites` is `[{point, members}]`, ONE entry per occupied point (never per member: a stack of
    3 is one site with 3 members, not 3 separate placements) — mirroring `show_targets`
    exactly: natural `sequenceIndex` per object, then the SAME `coLocation` branch (`"keep_
    apart"` → `resolveKeepApart`, ported bit-for-bit; `"stack"` → untouched), THEN grouped by
    exact resolved-position equality pre-jitter (jitter stays per-member, applied after
    grouping, same reasoning as before). `objectStackAt` (the marginal-overlay's per-point
    data, INTENTIONALLY unaffected by `coLocation` — it's a static "every point this object
    could occupy" view with no per-episode dynamics to branch on) also dropped its own
    fan-out math for the same reason as the render change below. A bounded scan
    (`KEEP_APART_SCAN_LIMIT = 50` episodes) feeds `updateCollisionWarning()`, the renamed/
    repurposed warning note (was `phaseWarning`), so an operator sees residual collisions
    before ever recording, not just via `session.py`'s per-episode log line.
  - **Tower-legend rendering replaces the fan-ring**, directly per a report that a scattered
    ring of identical-looking same-color-coded dots a fixed radius apart didn't make "which
    object is on top" legible. `stackOffsetPosition`/`OBJECT_STACK_FAN_RADIUS` are gone
    entirely — every stack member now shares the literal SAME render point (no fan), and a new
    `drawStackLegend(ctx, point, sampleRadius, members)` draws exactly one dot (colored by the
    TOP/highest-level member, so "what's visually on top" matches the real-world stacking)
    plus, for 2+ members, a small `×N` badge and a compact vertical legend beside the dot
    (connected by a thin line) listing every member top→bottom with a color swatch, name, and
    explicit level number — unambiguous regardless of stack height, applied identically in the
    marginal overlay and the episode stepper, on both the camera and orthographic views.
  - **Sample/target counter** (`updateSampleCountLabel`, a new `sampleCountLabel` note in the
    Objects card) — requested directly, since stacking/rotation can both inflate the real
    number of targets a session shows beyond the configured pattern size. BASE is
    `samplePointsCanonical.length` — the ACTUAL rendered point count, not the configured
    `steps` value, since `count_mode: "variable"` can render fewer points than configured (the
    one explicit correctness requirement given: "count the variable amount if its not a fixed
    sample size... the amount of points in the scene not in config"). With objects defined:
    `placements = Σ object.assigned.length` (counted once per object-point ASSIGNMENT, so
    `placements > basePoints` can only happen via 2+ objects sharing a point — this IS the
    stacking signal, surfaced as "+N stacked"; `placements < basePoints` surfaces as "N
    unused"), and `targets = Σ assigned.length × (orientationCount if enabled else 1)`,
    surfaced as "+N from rotations" on top of placements. With no objects (legacy path): a
    single shared tool-wide rotation setting multiplies `basePoints` directly.
  - Verified via 5 new `resolve_keep_apart` Python tests (disjoint sets untouched; overlapping
    sets separated onto a free point; residual reported when no alternative exists; absent
    objects skipped; deterministic given a fixed `episode_index`) plus 4 new `test_config.py`
    tests (`co_location`/`level_strategy`/`level_seed` default/parse/validate) and 4 rewritten
    `test_session.py` tests (stack-forms-from-coincidence and scene-level `level_strategy`
    replacing the old authored-`StackConfig` tests verbatim in behavior; `keep_apart`
    separating objects with an alternative; `keep_apart`'s residual collision degrading to a
    visible stack AND logging a warning naming the object) — 285 Python tests total (every
    `"phase"`/`"stratified"`/`StackConfig` test removed, net). JS side: 86 Node-harness
    assertions (up from 62) — `resolveKeepApart` bit-for-bit parity vs. `resolve_keep_apart`
    (including the residual and disjoint-untouched cases); `computeEpisodePlacements`'s new
    `{sites, residual}` shape (one site per coincidence, not per member; `keep_apart`
    separating/falling-back correctly); `objectStackAt` sharing one unfanned point across
    members; export gating for the new scene-level fields and the ABSENCE of any `stacks:`
    block; the co-location toggle's UI wiring (including disabling the level controls);
    `updateCollisionWarning`'s scan in both directions; and the sample-counter's disjoint/
    stacked/rotation-delta math. `examples/stacking.example.yaml` was rewritten to match (the
    3 stacked blocks' shared coordinate now implies the stack with no `stacks:` block at all;
    `level_strategy`/`level_seed` moved to top-level scene fields) and re-verified end-to-end
    via the same `load_config` + mocked `run_session` smoke as before — identical recorded
    behavior to the pre-redesign version, confirming the new derived-stacking mechanism is a
    faithful replacement, not just a differently-shaped config.
- **A second ambiguity in the same area, found by the same user one pass later: "stack vs.
  keep_apart" only ever disambiguated what happens AT a shared point — it said nothing about
  how many distinct points/stacks are visible AT ONCE, every episode.** Verbatim report: "there
  can be multiple stacks per episode and a single stack as a point per episode same as there
  could be one object per episode or multiple objects per episode, account for that in a
  separate setting." Until this point, every variable object was shown EVERY episode,
  unconditionally (modulo the now-removed `presence` knob, see below) — there was no way to say
  "show only one object, or one coincidentally-bonded group of objects, per episode, rotating
  round-robin across the session" (the tool's original UC1: "place one object at a time at the
  designated spot"). Fixed with a new scene-level `OverlayConfig.episode_targets: "all" | "one"`
  (default `"all"`, today's exact behavior — every existing config is byte-for-byte unaffected).
  - **Getting the rotation UNIT right took three attempts — the first two each broke a
    different, directly-reported real scenario, and the eventual fix abandoned the entire
    "pre-compute a static partition of objects" framing rather than patching either bonding
    rule.** Attempt 1: `pattern.partition_rotation_groups`, a union-find merging any two
    objects whose CONFIGURED pattern point lists shared at least one coordinate. Reported
    broken ("i set up 3 objects ... no difference between the modes"): with several objects
    each sweeping multiple points, pairwise/transitive overlaps merged ALL of them into one
    ever-present mega-group, so `"one"` showed that one group every episode — visually
    identical to `"all"`. Attempt 2: tightened the bonding rule to require IDENTICAL point
    SETS (not just any overlap), reasoning that only a deliberately-pinned stack should bond.
    Reported broken immediately after ("instead of a stack it displays each point of the stack
    separately one by one"): a real stack whose members sweep mostly-different points and only
    SOMETIMES coincide (an emergent stack, the normal case via `co_location: "stack"`) has
    non-identical full point sets, so it no longer bonded at all — it split into separate
    single-object turns, never showing together. The actual fix: DELETE
    `partition_rotation_groups` entirely (Python AND its JS twin
    `calibrate.js`'s `partitionRotationGroups`) — there is no static, pre-computed grouping
    that's right for both "always coincide" and "sometimes coincide" stacks, because whether
    objects coincide is fundamentally a PER-EPISODE runtime fact (this is exactly what the
    pre-existing, unrelated stack-detection logic already computes for `"all"` mode — grouping
    by exact resolved-position equality). So `episode_targets == "one"` doesn't pre-classify
    objects into groups at all: it lets EVERY object compute its own natural point exactly like
    `"all"` mode (against the raw `episode_index`, with no special-casing), groups them into
    this episode's actual SITES via that same existing logic, and only THEN picks exactly one
    of those sites to keep — round-robin, ordered by each site's earliest-declared member.
    Whatever coincides this episode (a permanent stack, an emergent one, or nothing) becomes
    one rotation unit by construction, with no separate notion of "bonded" needed at all.
  - **`session.py`'s `show_targets`: an object's own sweep/orientation/level ALWAYS use the
    raw `episode_index`, identically to `"all"` mode — `episode_targets` only filters WHICH
    resulting site gets shown, never how an individual object's own cadence advances.** An
    earlier version of the fix (alongside attempt 1/2 above) introduced a `group_turn`
    (`episode_index // num_groups`) so a group only active 1-in-N episodes wouldn't visually
    freeze at its first point — but `group_turn` depended on the now-deleted static group
    count, so it had to go too. Its absence has one known, accepted tradeoff: if an object's
    own pattern length evenly divides the rotation period (commonly: the object count, in the
    no-coincidence case), that object can resonate and keep landing on the same point every
    time it's shown. Not fixed here — a narrower, lower-priority issue than the two concrete
    bugs above, and avoidable today by picking a pattern length that doesn't divide evenly
    into the object count.
  - **The calibration tool's episode stepper's `maxEpisodeCount()` no longer multiplies by a
    precomputed group count (deleted along with `partitionRotationGroups`) — it multiplies the
    base period by the live object count instead, a soft UX bound, not an exact period** (the
    true period can vary once coincidence is involved, since the number of sites itself then
    varies episode to episode — exactness was never the goal, just "let the stepper run far
    enough to preview a second visit"). `computeEpisodePlacements` mirrors `show_targets`
    exactly: compute every object's natural point against the raw `episodeIndex`, group by
    resolved-position equality (recomputed fresh every call), then under `episodeTargets ===
    "one"` keep only `entries[episodeIndex % entries.length]` after sorting sites by earliest-
    declared member. A new `episodeTargetsSelect` (Objects card, beside `coLocationSelect`)
    drives the scene-level toggle; export (`sceneLevelExportBlock`, renamed from
    `sceneLevelCoLocationExportBlock` now that it covers more than co-location) emits
    `episode_targets: one` only when non-default.
  - **`episode_targets` is ONLY visible in the calibration tool's EPISODE preview, never the
    marginal one — and selecting `"one"` now auto-engages that episode preview, because not
    doing so made the toggle appear completely inert (reported directly: "does not work at all,
    there is no difference between the modes").** Root cause: `computeEpisodePlacements` (the
    only consumer of `episodeTargets`) only feeds the render when `previewMode === "episode"`;
    the default `previewMode` is `"marginal"`, whose whole job is to draw EVERY assigned point of
    EVERY object at once (a coverage check across all episodes) — so by construction it cannot
    express "one site per episode," and toggling `episodeTargets` while in it changed nothing on
    screen. Fix is in the `episodeTargetsSelect` change handler: selecting `"one"` (with objects
    present) flips `previewMode`/`previewModeSelect`/`episodeStepper` to the episode view so the
    rotation is immediately visible; selecting `"all"` leaves the current preview untouched
    (marginal stays perfectly meaningful there — it's the pre-existing default). Deliberately a
    one-shot auto-switch in the handler, NOT a per-render guard, so a later DELIBERATE switch back
    to marginal (a legitimate coverage check even under `"one"`) isn't fought. Note this means a
    scene where every object always coincides into a single site (or there's only one object
    total) shows no `all`-vs-`one` difference at all even in episode mode — correct, not a bug:
    there's only ever one site to rotate through.
- **In the same pass, both remaining per-object knobs from the original "behavioral
  randomization" brainstorm — `jitter_px` (per-episode positional noise) and `presence`
  (probabilistic per-episode absence) — were removed outright, per a direct question ("why do
  we need the jitter and presence, remove that") rather than a redesign.** Neither survived
  scrutiny once `episode_targets`/`co_location` existed to handle the *structural* presence/
  absence questions they were partially trying to approximate: `presence` was a blunt,
  probabilistic way to make an object sometimes invisible, which `episode_targets: "one"` now
  does deterministically and meaningfully (an object is absent exactly when it's not part of
  this episode's active site, not at random); `jitter_px` added sub-pixel positional noise with
  no analogous structural justification at all once examined directly — removed as dead weight
  rather than defended. Removed entirely, not deprecated/defaulted-off: `pattern.py`'s
  `apply_jitter`/`is_present` functions, `ObjectConfig.jitter_px`/`presence` fields (and their
  `config.py` parsing/validation/allowed-keys entries), `calibrate.js`'s `applyJitter`/
  `isPresent` and the `jitterPx`/`presence` fields on `makeObject()`, the
  `objectJitterInput`/`objectPresenceInput` calibration-page controls, and every export line
  for both. `session.py`'s `show_targets` is simpler for it: every object always has a real
  natural index now (no `None` from a `presence` check), and `apply_jitter` is gone from the
  final per-object loop entirely (`point = base_positions[obj.name]` directly, no per-member
  noise step). `examples/stacking.example.yaml`'s distractor-cube object lost its
  `jitter_px`/`presence` lines (kept `sequencing: shuffled`, its one remaining real per-object
  knob) and gained a commented `# episode_targets: all` alongside the existing commented
  `# co_location: stack`, explaining the two settings' distinct scopes directly in the example.
  - Verified via 3 new `test_config.py` tests (`episode_targets` default/parse/invalid)
    replacing the removed `jitter_px`/`presence` validation tests; 4 new `test_session.py`
    tests (`episode_targets="one"` rotating independent objects round-robin; keeping a
    permanently-coincident stack together as one site while an independent object alternates
    with it; an object's own sweep using the raw `episode_index` directly, not a per-turn
    counter; and the core fix's regression guard — an EMERGENT stack that only coincides on
    SOME episodes still shows together exactly when it coincides, with the site count/rotation
    period itself varying episode to episode) replacing the removed `jitter`/`presence` session
    tests and the abandoned `partition_rotation_groups` tests — 281 Python tests total (net
    down, since the deleted static-partition function's own tests outnumbered its replacement,
    which needed none — there's no longer a separate pure function to unit-test, only
    `show_targets`' existing per-episode grouping, already covered). JS side: 95 Node-harness
    assertions — the same scenarios as the Python session tests run directly through
    `computeEpisodePlacements`, `maxEpisodeCount`'s object-count multiplier, export gating for
    `episode_targets` in both directions, and the `episodeTargetsSelect` UI wiring (including
    the auto-engage-episode-preview behavior) — net of every `applyJitter`/`isPresent`/
    `jitterPx`/`presence`/`partitionRotationGroups` assertion removed.
- **`episode_targets: "one"`'s per-episode DYNAMIC-coincidence design (the previous bullet)
  turned out to be the wrong fix for what it was built to solve, reported directly: out of 60
  episodes (expected 20) only 3 stacked, and not even fully (2 of 3 objects) — replaced with a
  STATIC, point-driven model where assignment to the same point IS a guaranteed stack, not a
  coincidence that has to line up.** Root cause: that design computed each object's position
  via its own independent `sequence_index` sweep EVERY episode, and only formed a stack when
  two objects' sweeps *happened* to land on the same point THAT episode — exactly the same
  rare-coincidence failure mode the very first "Bimanual" stacking bullet earlier in this file
  already diagnosed and fixed for the position-PATTERN engine (`co_location`/`resolve_keep_apart`
  exist precisely because independent sweeps rarely coincide), but `episode_targets: "one"`
  had reintroduced an equivalent dependency on coincidence one layer up, at the SITE-selection
  level. Separately, `calibrate.js`'s `maxEpisodeCount` multiplied the base period by the
  object count (`base * numObjects`) as a soft UX bound for this same per-episode-coincidence
  uncertainty — 20 points × 3 objects gave 60, not 20, independent of whether anything actually
  stacked.
  - **Fix: `pattern.occupied_sites(object_points)` / `calibrate.js`'s `occupiedSites()`** —
    a pure, `episode_index`-FREE structural function over the STATIC union of every object's
    own assigned points, grouped by exact coordinate (Python: float-tuple equality, the same
    equality `show_targets`'s "all"-mode coincidence grouping already used; JS: shared
    `samplePointsCanonical` index equality, simpler since two objects "sharing a point" there
    are always the literal same array index already). Each returned site carries every
    object's ordinal present at that exact point — 2+ ordinals there is now a PERMANENT,
    guaranteed stack, full stop, never a per-episode maybe. Sites are ordered by the smallest
    `(object_ordinal, point_index_within_that_object's_own_pattern)` among their members, for
    a deterministic round-robin. `session.py`'s `show_targets` and `calibrate.js`'s
    `computeEpisodePlacements`/`maxEpisodeCount` now branch early on `episode_targets == "one"`
    into this entirely separate, `sequence_index`/`co_location`-free code path: episode
    `episode_index % len(sites)` shows exactly that site's full membership (leveled via the
    existing `level_order`/`stackLevelStrategy`, which DOES still apply unconditionally here —
    every multi-member site is a real stack), clearing every other object. `#episodes` is
    therefore exactly `len(occupied_sites(...))` — the number of DISTINCT occupied points, full
    stop — not multiplied by object count, and not dependent on any sweep. An object with
    several assigned points gets one dedicated turn per point (not a single "active turn" that
    sweeps across them), which is also the fix for "only 2 of 3 objects" in the stack: a stack
    member is never dropped because its own independent sweep happened to wander elsewhere that
    episode — there is no sweep to wander, the site's membership is fixed by assignment alone.
    `co_location`/`sequencing` are documented as having no effect under `episode_targets: "one"`
    (`config.py`'s field docstrings updated accordingly) — co-location there is exactly the
    static fact `occupied_sites` already encodes, not a per-episode choice to make.
  - **Confirmed end-to-end against `examples/stacking.example.yaml`** (mocked dataset/Rerun,
    real `run_session`): under `episode_targets: one`, the 3-block stack (all sharing one
    point) now shows as ONE site with all 3 members EVERY time its turn comes — never partial —
    while the distractor cube's 3 own (unshared) points each become their OWN separate site,
    for 4 total sites/episodes on this exact config, rather than the "2 groups" the old design's
    own example comment claimed (that comment was itself wrong about the new model and has been
    corrected).
  - **A second, separate, requested feature landed alongside the bug fix: `episode_targets:
    "all"` gained scene-level `combination_count: int | None`, decoupling the number of
    distinct SIMULTANEOUS multi-object combinations from any one object's own pattern length** —
    reported directly: "if i have 3 objects and 20 steps the combinations are limited to the
    nr of steps." Root cause: "all" mode's lockstep `sequence_index` is `episode_index %
    length`, so the distinct combination count can never exceed the longest object's own
    length, with no knob to ask for more. Fix: `pattern.combination_index(combination_i,
    length, object_ordinal, num_combinations)` (+ `calibrate.js`'s `combinationIndex`) — a
    "systematic spread": `combination_i` maps onto `[0, length)` by simple proportion
    (`combination_i * length // num_combinations`, pure integer floor division — deliberately
    NOT `round()`/`Math.round()`, whose half-rounding divergence between Python
    (round-half-to-even) and JS (always away from zero) was a real, separately-found bug in
    the now-removed "phase" sequencing mode this otherwise resembles), then phase-shifted by
    `object_ordinal` (a plain integer offset) so equal-length objects don't always pair the
    same way. `show_targets`/`computeEpisodePlacements` use `combination_index(episode_index %
    combination_count, ...)` per object instead of `sequence_index` whenever
    `config.combination_count`/`combinationCount` is set, leaving every other code path
    (`co_location`/grouping/stacking) completely unchanged downstream — `combination_count`
    only changes WHICH index each object resolves to, not what happens once they're resolved.
    `None`/`0` (default) is an exact byte-for-byte no-op, falling back to today's lockstep.
    Scoped to `episode_targets: "all"` only — `"one"` has no notion of "combination" at all
    (its sites are a static structural fact, not a per-episode index choice), so
    `combination_count` is documented as ignored there, not validated against it (setting both
    is harmless, just inert). Other spread strategies (seeded-random draws, coprime striding,
    the full Cartesian product of every object's points, or running exactly
    `LCM(length, ...)` combinations so every lockstep pairing appears once) were explicitly
    requested as future options but deliberately not built now — only the systematic spread
    shipped, since it alone already fully decouples the combination count from the step count.
  - The calibration tool's Objects card gained a `Combinations` number input
    (`combinationCountInput`, "all" mode only — disabled, alongside `coLocationSelect`, the
    instant `episodeTargetsSelect` is switched to "one"; `stackLevelStrategySelect`/
    `stackLevelSeedInput` stay enabled in BOTH modes, since level strategy applies
    unconditionally under "one" and only when `coLocation === "stack"` under "all" — a
    separate availability check, not tied to `episodeTargets` alone).
    `sceneLevelExportBlock`/export emits `combination_count: N` only when non-zero, sibling to
    the existing `episode_targets`/`co_location`/`level_strategy`/`level_seed` lines.
  - Verified via 7 new `pattern.py` tests for `combination_index` (range bounds, even spread,
    decoupling from `length`, phase-shift decorrelation, an exact integer-boundary check
    standing in for the round()/Math.round() divergence this formula deliberately avoids) and 6
    for `occupied_sites` (disjoint/shared/ordering/empty cases); rewrote the 2 `test_session.py`
    tests that had encoded the OLD dynamic-coincidence semantics for `episode_targets: "one"`
    into tests of the new static, point-driven rotation (an object's several own points each
    get a dedicated turn; objects sharing ANY point in their own lists ALWAYS stack together on
    that site's turn, not just on episodes where independent sweeps happen to coincide) plus 2
    new tests for `combination_count` (decoupling verified directly against
    `combination_index`; unset falls back to byte-identical lockstep) — 299 Python tests total.
    JS side verified via a 26-assertion ad-hoc Node harness (this project's established,
    not-committed convention) covering `combinationIndex`/`occupiedSites` parity against the
    Python values/properties above, `maxEpisodeCount` no longer multiplying by object count
    under "one" (the exact reported 60-vs-20 scenario, reproduced directly), `computeEpisodePlacements`
    for both modes, the new `combinationCountInput`/`episodeTargetsSelect`/`coLocationSelect`
    availability wiring driven through their REAL registered listeners (not called in
    isolation), export gating, and an end-to-end real-Assign-button-click scenario confirming
    two objects assigned to the same point via the actual UI flow form a guaranteed 2-member
    site. `examples/stacking.example.yaml`'s `episode_targets`/`co_location` comment block was
    corrected to describe the new 4-site (not 2-group) rotation, and gained a commented
    `combination_count` explanation.
- **`combination_count`'s "systematic spread" had a real, reported bug — "combinations repeat
  themselves if i set a number higher than the sample count" — and a deeper mathematical limit
  underneath it that the bug fix alone couldn't solve, which is why `combination_mode` now
  offers 4 additional strategies (`"random"`, `"coprime"`, `"cartesian"`, `"lcm"`), each from
  the TODO list the previous bullet deferred.** Root cause of the reported bug:
  `combination_index`'s proportional formula (`combination_i * length // num_combinations`)
  was applied UNCONDITIONALLY, even when `num_combinations > length` (oversampling) — the step
  size (`length / num_combinations`) then drops below 1, so many consecutive `combination_i`
  values floor to the SAME index before advancing (e.g. `length=4, num_combinations=20`
  repeated index 0 for `combination_i in [0, 5)`, then index 1 for `[5, 10)`, etc.) — correct
  in AGGREGATE count per index, but each index visibly "stalled" for several consecutive
  episodes instead of cycling. Fixed by branching: undersampling
  (`num_combinations <= length`) keeps the proportional spread (each combination lands on a
  DIFFERENT, evenly-spaced index); oversampling switches to plain `combination_i % length`
  cycling, which never repeats two consecutive episodes (for `length > 1`) and still visits
  each index `num_combinations // length` or one more times overall.
  - **The deeper limit, found while writing this fix's own justification (not from a test
    failure): when every object shares the SAME pattern length `L`, NO purely-deterministic
    function of `combination_i` ALONE, applied independently per object, can ever produce more
    than `L` distinct JOINT (multi-object) combinations — confirmed numerically, not just
    argued: 3 objects of length 5, `combination_count=50`, still only 5 distinct joint tuples
    appeared across all 50 episodes.** The reason: every object's index is `f(combination_i mod
    L)` for some periodic `f` — phase-shifting by `object_ordinal` (the existing decorrelation
    mechanism) changes WHICH point each object lands on, but never increases how many DISTINCT
    `combination_i mod L` values exist, so the joint state, being a deterministic function of
    that one shared value, is capped at `L` regardless of how the per-object mapping is phased
    or strided. This is exactly why the original bug report's "3 objects, 20 steps" scenario
    needed a structurally different mechanism, not just a bigger `combination_count` — escaping
    the cap needs either randomness (independent draws don't all derive from one shared `i mod
    L`) or a genuinely joint, multi-object formula (the Cartesian product, whose period is the
    PRODUCT of lengths, not any single one).
  - **`OverlayConfig.combination_mode: str = "systematic"`** (+ `combination_seed: int = 0` for
    `"random"`) selects among 5 strategies, implemented in `pattern.py`'s `combination_index`
    (the 3 PER-OBJECT modes — `"systematic"`/`"random"`/`"coprime"`, each only needing this
    object's own length/ordinal) and a new `resolve_combination_indices`/`combination_period`
    (the top-level dispatcher `session.py` actually calls, handling `"cartesian"`/`"lcm"` too,
    which need every object's length JOINTLY and can't go through the per-object function at
    all):
    - **`"random"`**: an independent seeded draw per `(combination_i, object_ordinal)` via the
      shared `_mulberry32` PRNG (the same one `level_order`/`sequence_index`'s `"shuffled"`
      mode already use, for JS/Python parity) — `int(rng() * length)`. The joint space is up to
      `length_1 * length_2 * ... * length_n`, escaping the per-object cap, at the cost of
      losing the even-coverage/no-repeat guarantee (a short run can revisit a draw by chance).
    - **`"coprime"`**: a deterministic, no-RNG alternative to `sequence_index`'s `"shuffled"`
      mode (re-added after the structurally similar `"stratified"` sequencing mode was removed
      for a DIFFERENT axis — see the much earlier "phase"/"stratified" bullets — revisited here
      deliberately, per direct request, since this is a different problem: combination spread,
      not per-object visit order). `_coprime_stride_for(length, object_ordinal)` finds a stride
      coprime to `length` (search starts from a per-ordinal offset, walks forward — `gcd(1,
      length) == 1` always, so it never fails to terminate); `(combination_i * stride +
      object_ordinal) % length` then visits all `length` points exactly once per period, like
      `"shuffled"` achieves via an actual seeded shuffle. Same per-object cap as
      `"systematic"` — it does NOT escape it, just offers a deterministic alternative.
    - **`"cartesian"`**: `_cartesian_indices` — a standard mixed-radix decomposition of
      `combination_i` into one index per object, with each object's own `length` as that
      digit's base. As `combination_i` ranges over `[0, prod(lengths))`, EVERY possible
      simultaneous combination appears EXACTLY once — the only mode that's both fully
      deterministic AND escapes the per-object cap. `combination_period` derives this natural
      product automatically when `combination_count` is left unset (the whole convenience of a
      named mode instead of requiring the operator to compute `length_1 * length_2 * ...` by
      hand); if `combination_count` IS set, it caps the natural product
      (`min(combination_count, natural)`) — the documented "can get very large, may need a
      cap" from the TODO list.
    - **`"lcm"`**: plain `combination_i % length` per object, deliberately with NO phase shift
      (unlike `"systematic"`) — this mode's entire point is exposing the TRUE periodicity of
      VANILLA lockstep cycling itself, not adding decorrelation on top of it. `combination_period`
      derives `lcm(every object's own length)` automatically when unset — "every lockstep
      pairing appears once" means every combination plain per-object lockstep naturally
      produces (NOT every possible combination — that's `"cartesian"`; the two only coincide
      when every length is pairwise coprime, since only then does `lcm(lengths) ==
      prod(lengths)`). This also fixes a related, smaller pre-existing gap: before this mode
      existed, NOTHING in this engine computed/reported the true LCM-based period for
      differing-length objects under plain lockstep at all — `maxEpisodeCount`'s legacy
      fallback (untouched, still default) only ever reports the LONGEST object's length, not
      the LCM, so a differing-length scene's true joint period was always under-reported unless
      `"lcm"` mode is explicitly selected.
  - **Validation**: `combination_mode` must be one of the 5 values; `"random"`/`"coprime"`
    have no natural period of their own and REQUIRE `combination_count` to be set (`ConfigError`
    at load time if not — consistent with this project's "a conflict in something deterministic
    is a config error: fail loudly" rule); `"cartesian"`/`"lcm"` tolerate it being unset (derive
    their own natural period) and `"systematic"` (the default) tolerates it being unset too
    (meaning the WHOLE combination subsystem stays inert — see `combination_active` in
    `session.py`, unchanged from the previous bullet: active whenever `combination_count is not
    None or combination_mode != "systematic"`).
  - **The calibration tool's `combinationSubsystemActive()` deliberately diverges from
    `session.py`'s `combination_active` for live-preview robustness**: Python's config is
    validated eagerly at load time, so `"random"`/`"coprime"` without a count can never reach
    `show_targets` — but `calibrate.js`'s dropdown/number-input can transiently be in exactly
    that invalid state while the operator is mid-edit (e.g. just switched to "Random" before
    typing a count), and `combinationPeriod()` throws for it. Rather than crash `renderAll()` on
    every keystroke, `"random"`/`"coprime"` only activate once a count is actually set in the
    JS preview specifically; `"cartesian"`/`"lcm"` are always active regardless, matching
    Python exactly (they tolerate an absent count by design, not just transiently).
  - Added a `Combination mode` dropdown + `Combination seed` number input to the calibration
    tool's Objects card, beside the existing `Combinations` count field — `combinationSeedInput`
    is disabled unless `combinationMode === "random"`; both `combinationModeSelect`/
    `combinationSeedInput` (like `combinationCountInput`/`coLocationSelect`) are disabled
    whenever `episodeTargets === "one"`, which has no notion of "combination" at all.
    `sceneLevelExportBlock` emits `combination_mode`/`combination_seed` only when non-default.
  - Verified via 27 new `pattern.py` tests (`combination_index`'s oversampling-no-longer-stalls
    regression directly reproducing the report; the undersampling branch's evenly-spread/no-
    duplicates property; `"random"`'s determinism-given-seed and full-range coverage;
    `_coprime_stride_for`'s actual coprimality across a battery of lengths/ordinals and
    `"coprime"`'s permutation/determinism properties; `_cartesian_indices`'s full-product
    enumeration and wraparound; `resolve_combination_indices`'s per-mode dispatch and required-
    count validation; `combination_period`'s natural-period derivation and capping for
    `"cartesian"`/`"lcm"`) and 5 rewritten ones (the old proportional-only expected values
    replaced with the new branch's correct ones) — 371 Python tests total. 5 new
    `test_config.py` tests (`combination_mode` default/parse/invalid, the
    `"random"`/`"coprime"`-require-`combination_count` validation in both directions) and 5
    new `test_session.py` end-to-end tests (the oversampling-stalling regression reproduced
    through `run_session` itself; `"random"` escaping the per-object cap with 3 equal-length
    objects measured directly; `"coprime"`'s determinism and same cap as systematic;
    `"cartesian"`'s full 9-combination product for 2 length-3 objects, including the
    period-wraps-correctly check; `"lcm"`'s exact 12-episode period for lengths 4 and 6 — not
    6, not 24). JS side: 30 new ad-hoc Node-harness assertions (the oversampling-bug regression
    reproduced directly in JS; bit-for-bit parity of every new function against the Python
    values/properties above; `"random"`/`"cartesian"`/`"lcm"` end-to-end through
    `computeEpisodePlacements`/`maxEpisodeCount`; the random/coprime-without-a-count
    non-crashing fallback; the new dropdown/input's real listener wiring and export gating) —
    56 JS assertions total. `examples/stacking.example.yaml`'s `combination_count` comment
    block was extended to describe all 5 modes, with `combination_mode`/`combination_seed`
    added as further commented-out fields.
- **`episode_targets: "all"` + `co_location: "stack"` had a real, reported design flaw: at one
  location holding a single object and another holding an authored stack, the rendered scene
  mixed "a point" with "a SLICE of the stack" rather than "a point and the full stack" — and
  the operator had no setting to control this.** Root cause: in `"all"` mode, each OBJECT swept
  its own assigned points independently every episode (`sequence_index`, or the combination
  subsystem), and objects were grouped into sites purely by PER-EPISODE coincidence — so an
  authored stack (2+ objects assigned to the same point) only rendered full on whichever
  episodes the objects' independently-decorrelated sweeps happened to land together, and
  rendered a partial slice every other episode. Investigating the fix surfaced (and the user
  directly confirmed) two existing redundancies worth resolving FIRST, rather than adding a
  third parallel concept on top of them:
  1. **`episode_targets: "all" | "one"` was secretly two collapsed axes** — `"all"` =
     (sweep positions, show all sites); `"one"` = (static/full positions, show exactly one
     site) — two diagonal corners of a 2×2 grid. The originally-requested `overlay` ("all sites,
     fixed & full") and `subsample` features were simply the OTHER corners of that same grid,
     not new parallel concepts.
  2. **Per-object `ObjectConfig.sequencing` (`"lockstep"`/`"shuffled"`) and the scene-level
     `combination_*` subsystem were the same axis — visit order — asked at two different
     scopes.** The code already made them mutually exclusive (the combination subsystem, when
     active, silently bypassed `sequencing` entirely); keeping both was redundant surface area,
     not complementary functionality.
- **Fix: `episode_targets` was replaced by two ORTHOGONAL scene-level axes —
  `OverlayConfig.position_mode: "sweep" | "overlay"` (default `"sweep"`) and
  `OverlayConfig.site_selection: "all" | "subset"` (default `"all"`, + `site_count`/
  `site_order`/`site_seed` for `"subset"`) — and `ObjectConfig.sequencing` was REMOVED entirely,
  folded into `OverlayConfig.combination_mode` gaining two new values, `"lockstep"` (the NEW
  default, replacing `"systematic"`) and `"shuffled"`.** `position_mode` decides what a
  scene's SITES are (swept per-episode positions, or every assigned point shown statically);
  `site_selection` decides how many of those sites are shown each episode — fully independent,
  so e.g. `sweep` + `subset` is a new, real corner ("a sampled subset of SWEEPING sites each
  episode") that neither old value could express. Old `"all"`/`"one"` are exactly the
  `(sweep, all)` / `(overlay, subset k=1)` corners.
  - **The atomic-stack fix itself**: `pattern.py` gained `group_into_units(object_point_lists,
    co_location="stack") -> list[list[int]]` — groups object ordinals into a maximal set
    sharing an IDENTICAL ORDERED point-list (not merely some overlapping points — that weaker
    case is deliberately left to `position_mode: "overlay"`/`occupied_sites` instead, which is
    exactly why the two position modes complement rather than duplicate each other) into one
    atomic STACK UNIT, ordered by smallest member ordinal. `co_location == "keep_apart"`
    disables grouping entirely (returns all singletons) — keep_apart exists to SEPARATE
    co-located objects, so bonding identical-point-list objects into one unit would directly
    contradict it. Under `position_mode: "sweep"`, `session.py`'s `show_targets` runs the
    visit-order axis over UNIT lengths/ordinals instead of per-object ones, then expands each
    unit's chosen index back to every member — a stack unit's members always resolve to the
    IDENTICAL point every episode, so a stack can never be partially sliced again, regardless of
    `combination_mode`. **The back-compat guarantee**: when no two objects share a point-list
    (the common, non-stacking case), `group_into_units` returns one singleton unit per object,
    in declaration order — unit ordinal equals object ordinal, unit count equals object count —
    so feeding those into the SAME `sequence_index`/`resolve_combination_indices` machinery the
    old per-object code already called produces BYTE-IDENTICAL output to the pre-unification
    `episode_targets: "all"` behavior, verified by a dedicated test. This holds unconditionally
    for the new `"lockstep"` default (which never even consults `seed`/`object_ordinal`), and
    holds for `"shuffled"` whenever seeds are left at their (overwhelmingly common) default of
    `0` — `"shuffled"` switching from a per-object seed to one shared scene-level
    `combination_seed` is the one deliberate, accepted loss of granularity from folding
    `sequencing` into `combination_mode` (an object that previously set a custom per-object seed
    specifically for `sequencing: "shuffled"` needs a one-time migration; this project is
    pre-release, so a clean, documented break is acceptable here — same precedent as every other
    hard-break field removal in this file).
  - **`pattern.py` gained `resolve_unit_indices(episode_index, lengths, mode="lockstep",
    combination_count=None, seed=0) -> list[int]`** — the single dispatcher unifying the old
    per-object `sequence_index` sweep and the old scene-level combination subsystem into ONE
    visit-order axis: `"lockstep"`/`"shuffled"` route to `sequence_index` per unit (unit ordinal
    as `object_ordinal`, unit count as `num_objects`); the other 5 modes
    (`"systematic"`/`"random"`/`"coprime"`/`"cartesian"`/`"lcm"`) route to
    `combination_period`/`resolve_combination_indices` exactly as the old scene-level subsystem
    did — no more `combination_active` bypass flag (every mode is now dispatched unconditionally
    by name, never silently inert). **A real, deliberate correction this surfaced**: the OLD
    `_COMBINATION_MODES_REQUIRING_COUNT = {"random", "coprime"}` excluded `"systematic"` only
    because the old `combination_active` flag (`combination_count is not None or
    combination_mode != "systematic"`) meant `"systematic"` (the old default) was NEVER actually
    invoked with `combination_count=None` in practice — `combination_period` itself already
    unconditionally raises for `"systematic"` + no count, that branch was just unreachable
    before. Now that `"lockstep"` is the new no-count default and every mode dispatches directly
    with no bypass, `"systematic"` genuinely needs a count if selected — `config.py`'s
    `_COMBINATION_MODES_REQUIRING_COUNT` now correctly includes it, validated eagerly at load
    time (a `ConfigError` naming the requirement, not a sometime-later runtime crash).
  - **`pattern.py` gained `select_sites(num_sites, selection, count, order, seed,
    episode_index) -> list[int]`** — the `site_selection` axis: `"all"` returns every site
    index; `"subset"` (`k = min(count, num_sites)`) returns either a deterministic sliding
    window (`"round_robin"`: `sorted((episode_index*k+j) % num_sites for j in range(k))` — `k=1`
    reduces to exactly the old `episode_targets: "one"` rotation, `episode_index % num_sites`)
    or a seeded subset (`"random"`: `sorted(_seeded_permutation(num_sites,
    _combine_seed(seed, episode_index))[:k])`, reusing the existing `_seeded_permutation` shuffle
    directly rather than re-deriving the math, for the same reason `level_order`/
    `sequence_index`'s seeded modes already do). `"systematic"`/`"coprime"` site orders are
    reserved (present in `config.py`'s allowed-value set, gated to raise `ConfigError(...not yet
    implemented...)` at load time if selected) — designed for a future pass, deliberately not
    built now (only `"round_robin"`/`"random"` ship), mirroring this project's established
    "design the enum, implement one now, gate the rest" pattern (see the `subsample_mode`-style
    precedent elsewhere in this file).
  - **`session.py`'s `show_targets` is now ONE unified function** (previously two
    near-duplicated branches keyed on `episode_targets`): builds `raw_sites` either from
    `occupied_sites` (static, `position_mode == "overlay"`) or from unit-grouping + the
    visit-order axis + `co_location`/`resolve_keep_apart` + exact-position grouping (dynamic,
    `"sweep"`) — THEN applies `select_sites` to both paths identically, so `site_selection` is
    genuinely orthogonal to `position_mode` in the implementation, not just in the config schema.
    Rotation's `visit_number` approximation (see `target_for_episode`'s existing "approximates
    how many times this position has been visited" caveat) generalizes to `episode_index //
    len(occupied_sites(...))` under `"overlay"` (matching the old `"one"` mode's formula exactly)
    and `episode_index // len(point_lists[ordinal])` under `"sweep"` (matching the old `"all"`
    mode's formula exactly, using the object's own FULL pattern length, not the unit's).
  - **`OverlayConfig.episode_targets` is GONE as a dataclass field — kept only as a deprecated
    YAML-level PARSING alias inside `load_config`**, not as a real field anything downstream
    reads: `episode_targets: "all"` sets the `position_mode`/`site_selection`/`site_count`
    DEFAULTS to `"sweep"`/`"all"`/`None` before the normal new-field parsing runs (a no-op if the
    file already matches today's defaults); `"one"` sets them to `"overlay"`/`"subset"`/`1`.
    Specifying `episode_targets` together with ANY of the new axis keys is a `ConfigError` (never
    two ways to say the same thing in one file) — verified by a dedicated test in both
    directions. Python test code that constructs `OverlayConfig` directly (bypassing
    `load_config`) must use the new field names — this is a hard, intentional break (consistent
    with this project's pre-release stance), not a silently-accepted old kwarg.
  - **A second, smaller dead-code finding surfaced while removing `sequencing`**:
    `PatternConfig.seed` under `shape: "points"` had its ONLY remaining consumer (the per-object
    `sequencing: "shuffled"` seed) removed by this same change, making it pure unused config
    surface — removed from `_POINTS_ALLOWED` entirely (a `seed:` key under `shape: points` is now
    a `ConfigError`, same treatment as the `StackConfig`/`jitter_px`/`presence` removals
    elsewhere in this file). `seed` remains on `PatternConfig` itself and fully functional for
    `sector`/`union` shapes, which have their own, unrelated sampling use for it.
  - **`rerun_client.log_target` gained an optional `extra_placements: list[tuple[Point, Point |
    None, int, int]] | None = None` parameter — needed because `position_mode: "overlay"` can
    make ONE object occupy several simultaneous points at once** (every point in its own
    pattern is its own site under `"overlay"`, unlike `"sweep"` where an object is always in
    exactly one site). All placements (the existing primary point/orientation_tip/level/
    stack_size args, plus every `extra_placements` tuple) are logged as N fanned positions/
    labels under ONE `rr.Points2D` entity (and N arrows, skipping placements with no tip, under
    one `rr.Arrows2D` child) — a SINGLE `rr.log` call per entity means re-logging always fully
    replaces every previous placement, even if a PREVIOUS episode had MORE of them, with no
    separate clearing logic needed for the shrinking case. Deliberately additive, not a
    signature overhaul: every pre-existing call site (the overwhelmingly common case — `"sweep"`,
    or `"overlay"` with single-point objects) omits `extra_placements` and is byte-for-byte
    unchanged, confirmed by every pre-existing `test_rerun_client.py` test passing with zero
    modifications.
  - **Calibration tool**: `episodeTargetsSelect`/the per-object `objectSequencingSelect`/
    `objectSeedInput` controls were removed outright (not deprecated-but-hidden) and replaced
    with `positionModeSelect`/`siteSelectionSelect`/`siteCountInput`/`siteOrderSelect`/
    `siteSeedInput` plus `combinationModeSelect` gaining "Lockstep"/"Shuffled" options —
    `calibrate.js` ported `groupIntoUnits`/`resolveUnitIndices`/`selectSites` bit-for-bit
    (adapted to this tool's shared `samplePointsCanonical` INDEX space: "identical point-list"
    there is just "identical sorted assigned-index array," no float-coordinate keying needed,
    unlike the Python original which must compare raw point lists since each object owns its
    own independent list) and rewrote `computeEpisodePlacements`/`maxEpisodeCount` as ONE
    unified function each, mirroring `session.py`'s own unification exactly — the return shape
    (`{sites, residual}`) is unchanged, so `renderObjectPlacementsCamera`/`Ortho`/
    `drawStackLegend`/`updateCollisionWarning` needed NO changes at all. The old `"episode_
    targets: 'one' auto-engages episode preview"` UX fix was generalized to fire whenever
    `siteSelection` becomes `"subset"` (the new, more general "fewer things shown" case the
    marginal/coverage view can't represent) rather than the old binary `episodeTargets ===
    "one"` check. `effectiveCombinationModeForPreview()` replaced `combinationSubsystemActive()`
    for the same live-preview-robustness reason as before (a transiently-incomplete
    systematic/random/coprime selection with no count yet falls back to `"lockstep"` rather than
    crashing `renderAll()` on every keystroke; lockstep/shuffled/cartesian/lcm never need this
    fallback). Verified via an ad-hoc Node harness (this project's established,
    not-committed-to-the-repo convention) — 113 assertions covering `groupIntoUnits`/
    `resolveUnitIndices`/`selectSites` parity (cross-checked directly against the Python
    functions' outputs for the seeded/grouping cases, not just internal JS self-consistency),
    the atomic-stack-never-slices behavior under `"shuffled"` end-to-end through
    `computeEpisodePlacements`, disjoint objects staying unaffected, both `"overlay"` cases
    (multi-point single object, static stack), `"sweep"` + `"subset"`, `maxEpisodeCount` parity,
    export gating, real DOM-event wiring for every new control plus the auto-engage behavior,
    a full `renderAll()` smoke test, and the back-compat byte-identical default-sweep anchor.
  - **`examples/stacking.example.yaml`** was migrated: the distractor cube's `sequencing:
    shuffled` + pattern-level `seed: 3` (both removed) became scene-level commentary describing
    the equivalent `combination_mode: shuffled` (commented out, since this scene's stack and
    distractor don't need decorrelating from each other to demonstrate the feature); the
    `episode_targets`/`combination_mode` commentary was rewritten for the new fields and the new
    `"lockstep"` default. Re-verified end-to-end via `load_config` plus a mocked `run_session`
    smoke for BOTH the default config (confirms the 3-block stack renders with `stack_size=3`
    every episode, levels rotating under `level_strategy: balanced`) and a
    `position_mode: overlay` + `site_selection: subset, site_count: 1` variant (confirms the
    documented 4-site rotation: the stack as one site, then each of the distractor's 3 own
    points in turn).
  - Verified via Python: `group_into_units`/`resolve_unit_indices`/`select_sites` unit tests in
    `test_pattern.py` (identical/partial-overlap/keep_apart/ordering for grouping; lockstep/
    shuffled/systematic dispatch correctness for the index resolver; all/subset round_robin/
    random correctness, capping, and empty-input handling for site selection); `test_config.py`
    coverage for `position_mode`/`site_selection`/`site_count`/`site_order`/`site_seed` defaults/
    parsing/validation, the `episode_targets` alias resolving to the correct new-field values in
    both directions and erroring when combined with a new axis key, the `sequencing`-key removal
    error, the points-shape `seed`-key removal error, and the new `"lockstep"` default/
    `"systematic"`-now-requires-count behavior; `test_session.py` coverage for the atomic-stack
    fix under `"shuffled"` end-to-end, disjoint-object non-interference, both `"overlay"` shapes,
    `"sweep"` + `"subset"`, and the `"shuffled"`-replaces-`sequencing` equivalence — 414 Python
    tests total (`pytest tests/ -q`).
- **The calibration tool's `positionModeSelect` (Sweep/Overlay) was removed outright and folded
  into the Preview dropdown itself, after a direct report that the two controls were
  redundant.** Verified the claim before acting on it: the Marginal preview's rendering
  (`objectStackAt`/`objectStacksAtPoint`) never reads `positionMode` at all — it always draws
  every assigned point of every object, regardless of sweep/overlay — so toggling Position mode
  while looking at Marginal had ZERO visible effect, a real, confusable dead control. Position
  mode only ever did anything once the operator was ALSO in the Episode preview (where sweep
  vs. overlay genuinely change what's shown). That makes the collapse lossless, not just
  shorter: `previewMode`'s `"episode"` value split into two — `"marginal"` | `"episode_sweep"` |
  `"episode_overlay"` — and a new `effectivePositionMode()` (`previewMode === "episode_overlay"
  ? "overlay" : "sweep"`) replaces the standalone `positionMode` variable everywhere it was read
  (`computeEpisodePlacements`, `maxEpisodeCount`, `updateSceneLevelControlsAvailability`,
  `sceneLevelExportBlock`) — `"marginal"` defaults to `"sweep"` for export purposes (an
  arbitrary but harmless choice, since Marginal never surfaces the difference either way).
  Every reachable, MEANINGFULLY DISTINCT preview state from before (today's coverage cloud;
  stepping through a sweeping scene; stepping through a static overlay scene to see level/
  rotation cycling) is still reachable — only the one truly-inert 4th combination (Marginal +
  Overlay, indistinguishable from Marginal + Sweep) was ever removed. `maybeAutoEngageEpisodePreview()`
  (the existing fix for `site_selection: "subset"` looking like a no-op under Marginal) now
  switches to `"episode_sweep"` specifically — the common-case default — rather than a bare
  `"episode"` value that no longer exists. This is a calibration-tool-only UI simplification:
  `OverlayConfig.position_mode` itself, and everything in `config.py`/`pattern.py`/`session.py`,
  is completely unaffected — only `calibrate.js`'s OWN internal preview-state representation
  changed. Verified via the ad-hoc Node harness: `effectivePositionMode()`'s derivation for all
  3 `previewMode` values; the real `previewModeSelect` "change" listener driving both
  `previewMode` and the combination/co-location controls' disabled state together; the
  subset-auto-engage now landing on `"episode_sweep"`; and that `positionModeSelect` is no
  longer a declared identifier anywhere in the script at all (a `ReferenceError` on access) --
  114 JS assertions total (up from 113), plus the unaffected 414 Python tests (no Python files
  touched in this pass).
- **The whole scene-level model was REDESIGNED from five co-equal, implementation-named axes
  (`position_mode`, `site_selection`+`site_count`+`site_order`+`site_seed`, `combination_mode`+
  `combination_count`+`combination_seed`, `co_location`, `level_strategy`+`level_seed`, plus the
  deprecated `episode_targets` alias) into a much smaller, intent-named set —
  `per_episode`/`combinations`/`order`/`stacking`/`level`/`seed` (+ an optional `preset:`
  shorthand) — after the project's own author reported being unable to express "I have 3
  objects and want to guide the operator to place exactly ONE of them per episode" without
  first working out which combination of those five axes did it.** Investigated directly: the
  case WAS already expressible (`site_selection: subset, site_count: 1`), but reaching it
  required understanding and discarding `combination_mode`'s 7 options, `co_location`, and
  `level_strategy` — none of which matter for it — which is exactly the kind of accreted
  surface area this project has repeatedly cut back before (jitter_px/presence, "phase"/
  "stratified" sequencing, the old `episode_targets` alias itself). The fix wasn't just
  renaming fields — it required a genuine mathematical reframe, worked out and confirmed with
  the user before writing any code: **what Rerun actually shows each episode is just a set of
  `{point, object, color, level}`, so the entire scene only needs to answer two questions: how
  many targets per episode, and how the distinct episodes are enumerated** — every other old
  axis turned out to be answering one of those two questions, just scoped or worded
  differently.
  - **`per_episode: "all" | "static" | <positive int>`** is the headline axis.
    `"all"` (default, today's original behavior) is the DYNAMIC case: every object's own
    position advances every episode (`pattern._episode_placements_sweep` — objects sharing an
    IDENTICAL point-list are grouped into one atomic stack unit via the pre-existing
    `group_into_units`, so a stack is always shown full, never sliced — this part is verbatim
    unchanged from the prior "sweep" design). `"static"` and `<int>` are both STATIC — no sweep
    at all, every point any object is assigned to is a fixed, permanent site
    (`pattern._episode_placements_static`) — `"static"` shows every one of them every episode
    (no rotation, a persistent coverage/reference view); `<int>` shows exactly that many,
    rotating via `order`. **`per_episode: 1` is the direct, one-field answer to the original
    report** — each object gets a one-point pattern, set `per_episode: 1`, done; see the new
    `examples/one_per_episode.example.yaml`, the canonical recipe for this case.
  - **`combinations: "synced" | "shuffled" | "all" | <positive int>`, only meaningful when
    `per_episode == "all"`** — a config-time `ConfigError` otherwise (`config.py`'s
    `'combinations' only applies when per_episode='all'`), rather than a silent no-op. This is
    the user's own explicitly-requested mathematical fix for the muddled relationship between
    a combination *count* and *mode*: a count only exists, and only ever means "sample N of the
    combination space," in this ONE case — every other `per_episode` value has no notion of a
    combination count at all, removing the entire old "does this mode need a count, and what
    happens if I set one anyway" decision tree. `"synced"` (default) is plain
    `episode_index % length` per unit — the OLD `"lockstep"` AND `"lcm"` modes turned out to be
    mathematically identical for this purpose (lockstep's per-unit modulo already has period
    `lcm(every unit's length)` with no extra machinery needed — `"lcm"` mode is gone, simply
    absorbed into `"synced"`, not replaced by it). `"shuffled"` and `"all"` (the old
    `"cartesian"`) are unchanged in meaning. `<positive int>` replaces the old
    `combination_count` + whichever of `"systematic"/"random"/"coprime"` was selected — which
    one now comes from the separate `order` axis below, instead of being baked into one
    7-valued `combination_mode` enum.
  - **`order: "even" | "random" | "coprime"`** is a SEPARATE axis from `combinations` on
    purpose — it answers "how to spread/sample," reused identically by two different things
    that both needed exactly this choice: sampling `combinations: <int>` (replacing
    `combination_mode: "systematic"/"random"/"coprime"`), AND selecting which static sites a
    `per_episode: <int>` shows each episode (replacing `site_order: "round_robin"/"random"` —
    `"even"` here maps internally to `"round_robin"`, since both mean the same
    deterministic-evenly-spread idea; reserved `"systematic"/"coprime"` site orders, never
    implemented, are dropped entirely rather than carried forward as dead enum values).
    `"coprime"` is config-time restricted to `combinations: <int>` only (`order='coprime'
    requires combinations to be a positive integer`) — it has no equivalent meaning for
    static-site selection.
  - **`stacking: "stack" | "keep_apart"`** replaces `co_location` directly (same two values,
    same meaning under `per_episode: "all"` — pile up vs. nudge apart via the unchanged
    `resolve_keep_apart`) but GAINS a real, new meaning under a static `per_episode`: `"stack"`
    merges objects sharing an exact point into one site (`occupied_sites`, unchanged); the new
    **`pattern._exploded_sites`** instead treats every individual `(object, point)` pair as its
    own, permanently un-stacked site. This is the literal, direct implementation of the user's
    own description of the desired behavior: *"if its in stack mode it should display one
    stack per episode but if its keep apart then it should display one point per episode while
    it cycles through all possible points within the scenario."* Confirmed by a dedicated test
    (`test_episode_placements_per_episode_one_keep_apart_cycles_every_individual_point`): two
    objects sharing one point plus a private point each — `"stack"` shows 2 distinct sites (the
    shared stack, then the private point); `"keep_apart"` shows 3 (every point individually,
    never merging the shared one).
  - **`level`/`seed`** replace `level_strategy`/`level_seed` (identical 4 values: `"fixed"` |
    `"shuffle"` | `"cycle"` | `"balanced"`) plus a NEW design decision: `seed` is now ONE shared
    field across every randomized aspect (`combinations: "shuffled"`/sampling `"random"`;
    `order: "random"`; `level: "shuffle"/"balanced"`) instead of three separate seeds
    (`combination_seed`/`site_seed`/`level_seed`) — a deliberate simplification matching "make
    the YAML schema as simple as possible," with the one real risk (different randomized
    aspects silently correlating because they share a literal seed number) handled by
    internally salting each aspect before use (`pattern._SEED_SALT_COMBINATIONS/_LEVEL/_SITES`,
    `seed XOR`-style combined via the pre-existing `_combine_seed`) — so one user-facing number
    still drives genuinely independent draws underneath.
  - **`pattern.episode_placements(point_lists, episode_index, per_episode, combinations, order,
    stacking, level, seed) -> (placements_per_ordinal, residual_collision_ordinals)`** is the
    single new orchestrator function `session.py`'s `show_targets` now calls directly, replacing
    ~100 lines of inline position_mode/site_selection/co_location branching with one call —
    dispatching to `_episode_placements_sweep` (`per_episode == "all"`) or
    `_episode_placements_static` (`"static"`/`<int>`), both of which are thin wrappers around
    the SAME unchanged building blocks (`group_into_units`, `resolve_unit_indices`,
    `resolve_keep_apart`, `occupied_sites`, `select_sites`, `level_order`) plus the one genuinely
    new function, `_exploded_sites`. `placements_per_ordinal[ordinal]` is now a list of
    `(point, level, stack_size, visit_number)` tuples — `visit_number` (for orientation-arrow
    cycling) is computed INSIDE this function now (per-branch: `episode_index //
    len(point_lists[ordinal])` under `"all"`, `episode_index // num_sites` under static),
    removing the `position_mode`-conditional visit-number formula that used to live directly in
    `session.py`.
  - **`ObjectConfig.variable: bool` was REMOVED entirely — every object now always has a
    `pattern`.** Investigated directly while redesigning the object schema for clarity: a
    `variable: false` object was filtered out by `session.py` (`variable_objects = [obj for obj
    in config.objects if obj.variable]`) and NEVER touched again — not logged once, not shown
    statically, nothing — pure dead config with no behavior at all. Removing it is a real
    simplification, not just a rename: `_parse_object` now always requires `pattern` (no more
    `variable: true` gating whether `pattern`/`orientation` are allowed), and a `variable:` key
    in a config is now a clean-break `ConfigError` naming the removal directly, same as the
    `sequencing` removal precedent.
  - **Clean break, no deprecated aliases** (explicit project-author decision: "make the names
    as clear as possible," reaffirming the existing pre-release stance) — every one of the 11
    removed top-level keys (`position_mode`, `site_selection`, `site_count`, `site_order`,
    `site_seed`, `combination_mode`, `combination_count`, `combination_seed`, `co_location`,
    `level_strategy`, `level_seed`) plus `episode_targets` itself is checked FIRST in
    `load_config` (`config.py`'s `_REMOVED_SCENE_KEYS` dict) and raises a `ConfigError` naming
    its replacement, before any other parsing runs — never silently reinterpreted, never a
    second accepted spelling.
  - **`preset:` is parse-time-only sugar — never a stored field, on either the Python or JS
    side.** `config.py`'s `_PRESETS` dict (`"sweep"` (default) | `"shuffled_sweep"` |
    `"one_at_a_time"` | `"cycle_through_points"` | `"show_everything"` | `"every_combination"` |
    `"sample_combinations"` | `"custom"`) supplies the DEFAULT value of each axis before any
    explicit axis in the same file overrides it — "advanced follows the preset but can be
    modified," per the user's explicit requirement. `OverlayConfig` itself only ever stores the
    resolved axes; `preset` is read, expanded, and discarded inside `load_config`.
  - **A capability genuinely would have been lost without the `"static"` value, found only by
    cross-checking every existing test against the new model before deleting anything:** the
    old `position_mode: "overlay"` + `site_selection: "all"` (the default) combination showed
    EVERY occupied site simultaneously, every episode, with NO rotation at all — a real,
    tested, persistent-coverage-view feature
    (`test_overlay_shows_every_occupied_point_full_every_episode_with_no_sweep`), distinct from
    both "all" (dynamic sweep) and any "<int>" (rotates a subset). Initially the new model only
    had `"all"`/`<int>`, which had no way to express "every static site, no rotation" without
    the operator hand-computing and hardcoding the exact current site count — fragile, and
    silently wrong the moment `exclude_zones` or the object count changed. Fixed by adding
    `per_episode: "static"` as a third value (`pattern.episode_placements`/
    `_episode_placements_static` branch on it directly) rather than dropping the capability —
    consistent with "without losing any of the functionality that is present now."
  - **Old → new mapping** (every old combination's new equivalent, used directly to migrate
    `examples/stacking.example.yaml`/`bimanual.example.yaml` and rewrite every affected test):
    `position_mode: sweep` + `site_selection: all` (the old default) → `per_episode: all`
    (the new default, byte-identical). `position_mode: overlay` + `site_selection: all` →
    `per_episode: static`. `episode_targets: one` (`site_selection: subset, site_count: 1`) →
    `per_episode: 1`. `combination_mode: lockstep`/`"lcm"` → `combinations: synced` (both
    absorbed into the same value, see above). `combination_mode: shuffled` →
    `combinations: shuffled`. `combination_mode: cartesian` → `combinations: all`.
    `combination_mode: systematic`/`"random"`/`"coprime"` + `combination_count: N` →
    `combinations: N` + `order: even`/`"random"`/`"coprime"` respectively. `co_location` →
    `stacking` (unchanged values). `level_strategy`/`level_seed` → `level`/`seed`.
  - **One old capability was deliberately DROPPED, not preserved under a new name: `position_mode:
    sweep` + `site_selection: subset` (dynamic per-object sweeping continuing underneath, with
    only a sampled subset actually revealed each episode).** This was the single most obscure
    corner of the old design — CLAUDE.md's own prior write-up already characterized it as "a
    new, real corner... that neither old value could express," i.e. it was discovered as an
    emergent consequence of two independent axes being orthogonal, not built because of a cited
    operational need. Under the new model, `per_episode: <int>` always means STATIC enumeration
    (matching the user's own stated mental model of "cycle through points/stacks one at a
    time"), so this exact combination is no longer directly reachable in the simplified schema
    — judged an acceptable loss given "as simple as possible," consistent with this project's
    history of cutting under-justified complexity (`jitter_px`/`presence`, "phase"/"stratified"
    sequencing). `test_site_selection_subset_under_sweep_shows_only_k_sites_per_episode` (the
    test that exercised it) was removed, not migrated.
  - **The calibration tool's Objects card was restructured around the same model**: a `Preset`
    dropdown (`calibrate.js`'s `PRESETS` table, byte-identical to `config.py`'s `_PRESETS`) plus
    five advanced controls (`perEpisodeSelect`+`perEpisodeCountInput`, `combinationsSelect`+
    `combinationsCountInput`, `orderSelect`, `stackingSelect`, `levelSelect`,
    `sceneSeedInput`) replacing the old 10 controls
    (`siteSelectionSelect`/`siteCountInput`/`siteOrderSelect`/`siteSeedInput`/
    `combinationModeSelect`/`combinationCountInput`/`combinationSeedInput`/`coLocationSelect`/
    `stackLevelStrategySelect`/`stackLevelSeedInput`) — editing any advanced control directly
    flips the preset selector to `"custom"` (`markCustomPreset()`), mirroring "advanced follows
    the preset but can be modified." The Preview dropdown SHRANK from 3 values
    (`"marginal"`/`"episode_sweep"`/`"episode_overlay"`) to 2 (`"marginal"`/`"episode"`) and lost
    its secret double-duty as the real scene's position-mode control entirely (see the
    `positionModeSelect` removal bullet above for why that fusion existed) — it is now PURELY a
    preview-view choice, with no bearing on what gets exported, now that `per_episode`'s
    open-ended `<int>` case can't be represented by a small fixed dropdown anyway.
    `objectsExportBlock` no longer emits a `variable: true` line per object (the removed field);
    `sceneLevelExportBlock` emits the new field names, gated identically (only when non-default,
    so an untouched scene still exports byte-for-byte to before this model existed).
  - **`_exploded_sites`/`explodedSites`, `episode_placements`/`computeEpisodePlacements`'s
    restructured dispatch, and the preset-expansion logic are all newly verified, in both
    languages** — `tests/overlay/test_pattern.py` gained direct tests for `_exploded_sites`
    (never merges) and `episode_placements` (the `"all"`/`"static"`/`<int>` × `"stack"`/
    `"keep_apart"` matrix, plus the residual-only-under-`"all"` and invalid-`per_episode`
    cases); `tests/overlay/test_config.py` and `test_session.py` were substantially rewritten
    (every `co_location`/`position_mode`/`site_selection`/`combination_mode` test ported to its
    new-field equivalent, plus new preset/removed-key/`"static"` tests) — 423 Python tests
    total (`pytest tests/ -q`), all passing, including a mocked end-to-end `run_session` smoke
    against both `examples/one_per_episode.example.yaml` (confirms the literal reported
    scenario: 3 single-point objects, `per_episode: 1`, episodes cycle red→green→blue→red) and
    the migrated `examples/stacking.example.yaml`. JS side reverified via the same ad-hoc Node
    harness convention (DOM-stub `document`/`window`, the extracted `<script>` body run in one
    `vm` context so test assertions share its top-level scope) — preset expansion for all 7
    presets, manual-edit-flips-to-custom, `explodedSites`/`occupiedSites` divergence, the
    `per_episode=1` cycling and `"keep_apart"` exploding behavior matching the Python tests
    above exactly, residual-only-under-`"all"` gating, export gating in both directions, and a
    full `renderAll()` smoke test with both zero and one object defined (covering the
    untouched-legacy-path and the new-model episode-preview path respectively).

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
as direct dependencies. `opencv-python-headless` and `pillow` are intentionally **not**
direct dependencies — nothing in this package opens/analyzes/encodes a camera frame's pixel
content; the overlay only watches file-arrival timestamps and logs synthetic target points,
and `overlay/calibrate.py` only copies a static HTML file (the camera feed it shows comes
from the operator's own browser, via `getUserMedia()`, never through Python). Add `cv2`/
`pillow` back to `pyproject.toml` if the deferred v2 auto-detect feature (or Phase 2's
`core/`) actually needs to touch pixels. `scripts/vizaudit_record.py` has zero dependency on
the `vizaudit` package at all (stdlib `multiprocessing` + `lerobot` only) and must be run
from the `lerobot` env, never `vizaudit`'s — it needs the `hardware` extra (robot/teleop
drivers) that `vizaudit`'s env
deliberately excludes.

## Planned architecture

```
src/vizaudit/
  core/                    # compute_frame_metrics(frame: np.ndarray) -> dict — placeholder until a later
                            # phase; pure vision only, no Rerun/dataset imports, ever
  overlay/                 # Phase 1: guided data-collection overlay
    config.py              # YAML pattern/object/exclude_zones/surface_calibration config — dataclasses + validation
    pattern.py             # pure pattern-generation functions (arc, line, sector; pixel or canonical space)
    perspective.py          # pure homography math (compute_homography/apply_homography) — no I/O
    dataset_watcher.py      # dataset-root resolution + EpisodeBoundaryWatcher (file-write-cadence polling)
    rerun_client.py         # rr.init(recording_id=...) + connect_grpc + target-marker logging
    session.py              # orchestrator wiring the above together
    cli.py                   # `vizaudit-overlay` entrypoint
    calibrate.py             # `vizaudit-calibrate` entrypoint — copies static/calibrate.html (camera-direct, no dataset/Rerun)
    static/calibrate.html     # standalone getUserMedia() coordinate-picker page
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
2. ✅ Done — Tier 1 no-hardware smoke tests, including the real native Rerun viewer (not just
   headless `.rrd` inspection): unit tests pass; `examples/fake_lerobot_session.py` +
   `vizaudit-overlay`/`examples/run_demo.sh` run end-to-end against a fake dataset;
   `EpisodeBoundaryWatcher` fires at the right cadence; the target marker visibly composites
   on the live feed in one merged Rerun view (the earlier GPU/WebGPU rendering problem was
   specific to the `--serve-web` browser path and the sandboxed test-tool shell — the native
   `rerun` viewer run from the user's own interactive terminal renders correctly).
3. ✅ Done — extended the pattern engine per
   `/home/bogdan/.claude/plans/rustling-riding-meadow.md`'s second plan: `sector` (filled-area,
   area-uniform sampling), `exclude_zones` (obstacle avoidance), `surface_calibration`
   (homography-based perspective correction for non-top-down cameras), and the
   `vizaudit-calibrate` HTML coordinate-picker tool. All covered by unit tests
   (`test_perspective.py`, extended `test_pattern.py`/`test_config.py`, `test_calibrate.py`).
4. Tier 2/3: validate the full pattern engine (sector/exclude_zones/surface_calibration) on
   real `lerobot-record` and SO-101 hardware, including running `vizaudit-calibrate` against
   a real tilted-camera frame. Then verify `minAreaRect` reliability on real `top` camera
   footage from `so101_cube_pick_place_50`, to start on the deferred v2 auto-detect addition
   and/or Phase 2's `core/`.
5. Design the Phase 2 sidecar artifact schema before writing the lerobot-dataset-visualizer
   panel component.
