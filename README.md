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
`variable: false` objects are never targeted. Each object can override the top-level
`marker:` block (e.g. just `marker: {color_rgba: [...]}`) -- any field it doesn't set falls
back to the top-level default. **Bimanual setups** (two arms in the same camera frame) are
just two `objects[]` entries with their own pattern regions and a distinct `marker` color
each, so the two simultaneous target markers stay visually tell-apart-able -- see
`examples/bimanual.example.yaml`. There's no separate schema concept for "arms": one shared
`surface_calibration`/`exclude_zones` already apply correctly to every object regardless of
how many there are (mark each arm's own base as an `exclude_zones` circle so neither arm's
target ever lands in the other's footprint).

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

`count_mode: variable` (sector only, `grid`/`radial` only, default `fixed`) skips relocation
entirely: it generates the same ideal positions and just drops whichever land outside the
valid area, so the final point count can be less than `count` -- simpler and relocation-
artifact-free, at the cost of not guaranteeing the exact count.

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

For tasks where the object's *orientation* matters too, not just its position, add an
`orientation:` block to an object (see `examples/pattern.example.yaml`'s `marble`) -- this
draws a target-rotation arrow from the target point alongside the position marker. `count`
is how many distinct target angles to generate (its own number, independent of the pattern's
own episode `count` -- the two cycle independently per episode, so they desync over a
session instead of always pairing the same position with the same rotation); `method` is
`uniform` (evenly spaced, the default) or `random` (seeded); `angle_start_deg`/
`angle_end_deg` are *relative to* `initial_angle_deg` (default `0`-`360`, i.e. a full
rotation around it) -- negative values are fine, e.g. `initial_angle_deg: 90,
angle_start_deg: -45, angle_end_deg: 45` spreads +-45 degrees around direction 90, instead of
requiring absolute angles computed by hand; `arrow_length` is in the same space as
the pattern's own `center`/`radius` (pixel space, or canonical space when
`surface_calibration` is set). The arrow itself is always computed by rotating in whichever
space the position pattern already samples in, then projected through the same homography --
a homography doesn't preserve angles, so this is what makes the arrow foreshorten/skew
correctly to match how a real rotated object would actually look under a tilted camera,
rather than just rotating it in raw pixel space. Omitting `orientation:` entirely (the
default) keeps an object exactly as before -- a position-only marker, no arrow.

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

- **Camera view**: with the square-with-corner-dots "Edit" icon active (the default), drag
  the 4 corners onto the real workspace surface -- a grid renders inside the quad live, so
  you can see it visually lying flat on the surface once aligned. Click "Mark extreme points"
  and, while physically moving the robot arm to its reach limits, click where it appears at
  each extreme (this is the one workflow that really needs the camera view specifically,
  since you're watching the real arm move against the live feed). Once the corners are
  aligned, click the padlock icon to lock them -- this only disables corner *dragging*
  ("the grid"); the reach circle, cuts, and extreme points all stay editable either way.
- **Orthographic view**: the same surface shown undistorted. Once you have 3+ extreme
  points, click "Fit from points" -- this is where the actual reach-circle is computed
  (circle math only makes sense in undistorted space). Drag the circle's center or edge to
  fine-tune it, and watch the camera view update live to confirm it still makes sense against
  the real image.

The "Edit" icon (square with 4 corner dots) is one tool for dragging *every* kind of existing
geometry -- the 4 corners, the fitted circle, any cut, and any extreme-reach point -- on
*either* view; pick whichever view is more convenient and the other updates to match. Use the
other "Edit" group icons to mark regions points shouldn't spawn in: rectangle (▭) / circle
(◯) (click-drag primitives) or polygon (⬠) (click to add vertices; double-click, Esc, or the
checkmark button ends it) -- starting on *either* view, same as "Mark extreme points"; the
camera view is where you'd normally do this (you can see the actual obstacle to trace it), but
drawing directly on the undistorted orthographic view works too, e.g. for a precise rectangle
or circle cut. The cursor turns into a knife on whichever view a cut tool is active on. Each
cut keeps its own shape (a circle cut stays editable as a center+radius, not a vertex mess)
and is stored relative to the workspace surface, not the raw camera pixels, so it visually
follows the quad if you go back and adjust a corner. The scattered-dots "Move sample points"
icon is a separate tool for
nudging one individual generated preview point by hand, overriding just that point without
touching the circle/cuts/distribution that produced the rest -- it's cleared if you change
"Steps", "Seed", or the distribution (a different point set), but survives a border-width
tweak. The &#10227; "toggle direction" tool drag-selects a rectangle and flips whether every
enclosed sample point shows an orientation arrow at all -- a calibration-preview-only aid for
deciding on a count/enable setting, with no effect on the exported config (the real engine
applies orientation uniformly, the same way `sampleOverrides` itself isn't exported either).
<kbd>Ctrl+Z</kbd> / <kbd>Ctrl+Shift+Z</kbd> (or the History toolbar's undo/redo icons)
step back/forward through any of this editing. Set "Steps" to your pattern's intended episode
count, pick a distribution (`grid`, `radial`, or `random` -- see below), and a border width --
the preview always shows exactly that many points, confined to the circle, the workspace, and
outside every cut (with the border margin), in both views.

For a **bimanual setup** (2+ arms sharing this one camera frame), use the "Arms" chips: each
chip is an independently-calibrated reach circle -- its own extreme-reach points and fitted
circle -- in its own color. Click "+" to add another arm (it gets a new color automatically
and becomes active), click a chip to switch which arm "Mark extreme points"/"Fit from points"
currently applies to, and use the name field/color swatch next to the chips to rename or
recolor whichever one is active. The 4 corners and any cuts stay shared by every arm (they're
facts about the camera frame itself, not about a specific arm).

A real bimanual rig's two reach circles typically *overlap* in the shared middle of the
workspace, so the "Independent" checkbox controls how multiple arms' circles combine for
sampling, **off by default**: with it off, every arm's circle is sampled together as their
*union*, so the overlap doesn't end up at roughly double the point density of the
non-overlapping parts -- the one shared "Steps"/"Seed"/"Distribution"/"Border width" preview
covers the whole union, and every arm's circle renders at full strength simultaneously (no
"other arms are dimmed" treatment, since none of them is more "active" than another for a
shared pattern). The export gets exactly one `pattern:` block (`shape: union`, listing every
arm's circle) ready to paste into one shared object. Turn "Independent" on if your two arms
actually work separate, non-overlapping zones and you want N fully independent patterns
instead: each arm gets back its own pattern-preview settings, renders at full detail only
while active (others dimmed), and the export gets one `pattern:` block per arm (+ a suggested
`marker: {color_rgba: ...}`) labeled with that arm's name, ready to paste into a separate
object. A single-arm session (the default: one arm, named "Arm 1") exports identically either
way -- this whole feature is purely additive and only matters once you add a second arm.

The "Orientation preview" toolbar group (off by default) previews the `orientation:` arrow
described above directly on both views, for whichever pattern is currently shown ("Count"/
"Method"/"Start°"/"End°"/"Seed"/"Arrow len" map 1:1 onto `orientation:`'s own fields). Like
"Pattern preview", it's per-arm with "Independent" on, shared otherwise. Enabling it adds an
`orientation:` block to the exported snippet automatically; leaving it off (the default)
exports nothing extra, so an existing single-arm/no-orientation session's export is
unaffected.
"Radial (beta)" -- labeled beta since it's had substantially more iterative correctness fixes
than grid/random -- isn't supported in combined/union mode yet (grid and random are): the
dropdown option disables itself automatically once 2+ circles are being combined, falling
back to "grid" if it was already selected. It stays available regardless of arm count with
"Independent" on, since that always samples one circle at a time.

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
TODO:
add episode band in which users can manually design each episode