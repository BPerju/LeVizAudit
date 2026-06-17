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
