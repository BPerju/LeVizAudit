# VizAudit

Audits and guides the **spatial diversity** of robot demonstration data collected with
`lerobot-record`, using purely visual (camera-frame) detection. See `CLAUDE.md` for the
full architecture and decision record, and `research_report.md` for the research behind it.

## Phase 1: guided data-collection overlay

For each episode, shows the operator where to place an object next (a point on a
configurable pixel-space pattern), directly composited on the live camera feed inside the
Rerun viewer -- so placement sweeps a diverse set of positions across a session instead of
being haphazard.

### Quickstart (three processes)

```bash
# 1. Shared Rerun gRPC server -- both other processes connect to this one server.
python -m rerun --serve-web --port 9876

# 2. In the `lerobot` conda env (needs lerobot's hardware/robot-driver extras), run this
#    wrapper INSTEAD OF plain `lerobot-record` -- same flags, otherwise identical:
conda activate lerobot
python scripts/vizaudit_record.py \
    --robot.type=so101 --dataset.repo_id=<you>/<dataset> \
    --display_data --display_ip 127.0.0.1 --display_port 9876

# 3. In the `vizaudit` conda env, start the guidance overlay -- before or at the same time
#    as step 2, so episode 0's target is visible from the start. --config defaults to
#    vizaudit_calibration.yaml (see "Calibration" below) if you omit it.
conda activate vizaudit
vizaudit-overlay --config examples/pattern.example.yaml \
    --connect 127.0.0.1:9876 --dataset.repo_id=<you>/<dataset>
```

Why a wrapper instead of plain `lerobot-record` in step 2: true on-feed compositing (the
target marker drawn directly on the live image, not a separate panel) requires both
processes to land in the same Rerun recording -- which, empirically, means matching *both*
`application_id` and `recording_id` (matching only one still produces two recordings, with
the marker silently rendered in its own empty view). `lerobot-record` never exposes a way to
set either directly, and patching its source is off the table. `scripts/vizaudit_record.py`
is a small, dependency-free shim that monkeypatches `rerun.init` (an in-memory
function-reference patch, not a file edit) to force both shared values before calling
lerobot's own unmodified `lerobot-record` entrypoint. See `CLAUDE.md` for the full rationale,
including why an earlier authkey-pinning approach turned out not to work.

### Config

See `examples/pattern.example.yaml`. Patterns are pixel coordinates on the named camera's
image plane -- no calibration, no physical units, unless you add `surface_calibration` (see
below). Objects with `variable: true` get a target marker each episode (one shared
pattern-index across all variable objects, but each has its own independent pattern);
`variable: false` objects are never targeted.

Three pattern shapes: `arc`/`line` (deterministic, boundary-only) and `sector` (a filled
pie-slice/annular-sector, evenly or randomly sampled -- genuine 2D coverage, not just a
curve). `sector` supports three `distribution`s: `grid` (the default -- a true Cartesian
lattice; cell spacing shrinks as the sample count grows, so coverage stays even at any
count), `radial` (a Fermat/Vogel spiral, the "sunflower seed" arrangement -- no two points
ever align radially or angularly, so there's no row/column/spoke structure, an alternative
look to `grid` rather than a replacement for it), or `random` (the original area-uniform
random sampling). Both `grid` and `radial` compute exactly `count` positions directly --
never more, never fewer, and never by over-generating a candidate pool and discarding the
surplus, which (especially at this tool's actual operating range of tens of points, not
thousands) thinned some regions of the pattern more than others and could look like points
were missing entirely. Any point a boundary would otherwise reject gets relocated to the
nearest valid spot instead of dropped. `border_width` keeps points away from every boundary
by some margin -- the circle's edge,
the workspace edge, *and* every `exclude_zones` cut -- in pixel or canonical units, matching
whatever space the pattern's `center`/`radius` are already in.

When an `exclude_zones` cut overlaps the pattern, both distributions reshape themselves around
it instead of just nudging individual points out of the way: `grid` splits a lattice column's
range around whatever cut crosses it and gives each resulting piece its own share of `count`
proportional to its (now correctly smaller) length; `radial` draws its radius from the true
available-area at each radius rather than the full disk's, so a radius band mostly covered by a
central cut gets proportionally fewer points up front. Both keep the *density* even across
whatever area is actually left, rather than bunching the points a cut displaces into a dense
ring right at its edge.

`exclude_zones` keep generated points off immovable scene obstacles (a robot base, a fixed
cup, an oddly-shaped cable clip) -- either `shape: circle` (`center`/`radius`, the default)
or `shape: polygon` (`vertices`, for obstacles a circle can't approximate well). `sector`
resamples around them automatically; `arc`/`line` raise a config error if a point lands in
one (they have no alternate point to substitute).

If your camera isn't perfectly top-down, add a top-level `surface_calibration` block (4
pixel-space corners marking a rectangle on the workspace surface) -- this corrects `sector`'s
sampling so it's uniform over the *real* workspace area, not just pixel area, and lines its
shape up with what the angled camera actually sees. Pick the corner coordinates without
guessing, *before you've recorded anything* -- calibration is fully standalone, with no
dependency on a dataset or recording session:
```bash
vizaudit-calibrate --output calibration.html
```
This writes the page, serves it on `http://127.0.0.1` (camera access needs a secure
context -- plain `file://` pages and editor preview panes typically block it), and opens it
in your default browser automatically. The page has two views, side by side:

- **Camera view**: with the square-with-corner-dots icon active (the default), drag the 4
  corners onto the real workspace surface -- a grid renders inside the quad live, so you can
  see it visually lying flat on the surface once aligned. Click "Mark extreme points" and,
  while physically moving the robot arm to its reach limits, click where it appears at each
  extreme (the feed stays live for this, since you're watching the real arm move). Once the
  corners are aligned, click the padlock icon to lock them -- this only disables corner
  *dragging*, so you can't nudge the surface by accident while marking points or drawing
  cuts; the reach circle stays editable either way.
- **Orthographic view**: the same surface shown undistorted. Once you have 3+ extreme
  points, click "Fit from points" -- this is where the actual reach-circle is computed
  (circle math only makes sense in undistorted space). Drag the circle's center or edge to
  fine-tune it, and watch the camera view update live to confirm it still makes sense against
  the real image.

Use the "Edit" toolbar's icon buttons to mark regions points shouldn't spawn in: rectangle
(▭) / circle (◯) (click-drag primitives) or polygon (⬠) (click to add vertices;
double-click, Esc, or the checkmark button ends it) on the camera view, where you can see
the actual obstacle -- the cursor turns into a knife while one of these is active. Each cut
keeps its own shape (a circle cut stays editable as a center+radius, not a vertex mess) and
is stored relative to the workspace surface, not the raw camera pixels, so it visually
follows the quad if you go back and adjust a corner. Click the pencil icon (✎) to drag any
existing cut's handles afterward -- on *either* view, camera or orthographic, same as the
fitted circle, which is likewise now draggable on both -- or click the active tool icon
again to stop drawing. Set "Steps" to your pattern's intended episode count, pick a
distribution (`grid`, `radial`, or `random` -- see below), and a border width -- the preview
always shows exactly that many points, confined to the circle, the workspace, and outside
every cut (with the border margin), in both views.

Click "Save to file..." once you're happy -- by default this writes straight to
`vizaudit_calibration.yaml` in the directory you ran `vizaudit-calibrate` from (no save
dialog, not your downloads folder), which is also where `vizaudit-overlay --config` looks by
default, so the two tools chain together with no flags. Ctrl-C the `vizaudit-calibrate`
command once you're done; pass `--no-serve` to just write the page without serving/opening
it (in that mode, "Save to file..." falls back to a native save dialog or a plain browser
download instead, since there's no server to save through). Calibrating, editing/saving the
config, and running the overlay are three independent steps -- nothing here needs to be
running at the same time as anything else.

### Development

```bash
pip install -e ".[dev]"
pytest
```
