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
#    as step 2, so episode 0's target is visible from the start:
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
image plane -- no calibration, no physical units. Objects with `variable: true` get a
target marker each episode (one shared pattern-index across all variable objects, but each
has its own independent pattern); `variable: false` objects are never targeted.

### Development

```bash
pip install -e ".[dev]"
pytest
```
