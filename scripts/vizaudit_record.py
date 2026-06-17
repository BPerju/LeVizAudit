#!/usr/bin/env python
"""Run lerobot-record with a Rerun recording shared with `vizaudit-overlay`.

Run this in the `lerobot` conda env, NOT `vizaudit` -- it needs lerobot's `hardware`
extra (real robot/teleop drivers) that the lighter `vizaudit` env intentionally lacks.
No installation step required; copy or symlink this file if running from elsewhere.

Usage: identical to `lerobot-record` -- all flags pass through unchanged.

    python vizaudit_record.py --robot.type=so101 --dataset.repo_id=... \
        --display_data --display_ip 127.0.0.1 --display_port 9876

Why this exists: true on-feed compositing (vizaudit's target marker drawn directly on
the live camera image, not a separate panel) requires this process and `vizaudit-overlay`
to land in the same Rerun recording. Empirically (via `rerun.dataframe.load_archive(...)
.all_recordings()`, not just the docstrings), a Rerun recording's actual identity is the
*pair* `(application_id, recording_id)` -- matching only one of the two still produces two
separate recordings, with the second process's data landing in its own, image-less view
(this is what bit us: the docstrings only mention recording_id). lerobot-record never
exposes either parameter directly, and patching its source is off the table (see CLAUDE.md
in the vizaudit repo). The fix: `lerobot.utils.visualization_utils.init_rerun()` calls
`rr.init(session_name)` via a lazily-imported `rr` (`import rerun as rr` inside the
function body) -- so monkeypatching `rerun.init` *before* lerobot is ever imported
intercepts that call cleanly, forcing both values to the shared constants regardless of
whatever lerobot happens to pass. This is a runtime patch of an in-memory function
reference in our own process, not a modification to any file in the lerobot checkout.

(An earlier version of this tried pinning `multiprocessing.current_process().authkey`
instead, per rr.init()'s docstring -- empirically that does NOT merge two independently
launched processes despite the docstring's wording; only real multiprocessing-spawned
children share the authkey-derived default. See CLAUDE.md's "Key decisions" for the full
story of both bugs found while building this.)

SHARED_APPLICATION_ID/SHARED_RECORDING_ID below must stay identical to the copies in
`vizaudit.overlay.rerun_client` -- a mismatch silently breaks compositing with no visible
error (the marker just lands in its own empty view instead of erroring).
"""

import rerun as rr

SHARED_APPLICATION_ID = "recording"
SHARED_RECORDING_ID = "8122e1bc-273e-4f00-a0ee-ab0b15c44107"

_original_rr_init = rr.init


def _init_with_shared_recording(application_id, **kwargs):
    kwargs["recording_id"] = SHARED_RECORDING_ID
    return _original_rr_init(SHARED_APPLICATION_ID, **kwargs)


rr.init = _init_with_shared_recording

from lerobot.scripts.lerobot_record import main  # noqa: E402

if __name__ == "__main__":
    main()
