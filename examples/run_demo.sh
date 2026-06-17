#!/usr/bin/env bash
# Runs the no-hardware vizaudit-overlay demo in the right order: cleans up any stale
# state first (the #1 cause of "it runs through all episodes instantly"), opens a Rerun
# viewer window, then streams a fake lerobot-record session into it while
# vizaudit-overlay draws target markers on top.
#
# Run this in your OWN interactive terminal (not via an agent's sandboxed shell) so the
# viewer window can actually open. Stop everything with Ctrl-C.
#
# Usage: bash examples/run_demo.sh
# Override any default via env var, e.g.: NUM_EPISODES=5 bash examples/run_demo.sh

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-vizaudit}"
DATASET_ROOT="${DATASET_ROOT:-/tmp/vizaudit_fake_dataset}"
RERUN_PORT="${RERUN_PORT:-9876}"
FPS="${FPS:-10}"
NUM_EPISODES="${NUM_EPISODES:-3}"
EPISODE_TIME_S="${EPISODE_TIME_S:-4}"
RESET_TIME_S="${RESET_TIME_S:-3}"
CONFIG="${CONFIG:-examples/pattern.example.yaml}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "==> Cleaning up stale state (this is what causes episodes to fly by instantly)"
pkill -f "rerun_cli" 2>/dev/null || true
pkill -f "vizaudit-overlay" 2>/dev/null || true
pkill -f "fake_lerobot_session.py" 2>/dev/null || true
rm -rf "$DATASET_ROOT"
sleep 1

OVERLAY_PID=""
cleanup() {
    echo "==> Stopping vizaudit-overlay (the Rerun viewer window is left open)"
    [[ -n "$OVERLAY_PID" ]] && kill "$OVERLAY_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "==> Opening the Rerun viewer window on port $RERUN_PORT"
rerun --port "$RERUN_PORT" &
sleep 3

echo "==> Starting vizaudit-overlay in the background (it waits for the dataset root)"
vizaudit-overlay --config "$CONFIG" --connect "127.0.0.1:$RERUN_PORT" \
    --dataset.root "$DATASET_ROOT" --dataset.repo_id ignored/unused --verbose &
OVERLAY_PID=$!

echo "==> Running the fake lerobot-record session in the foreground"
python examples/fake_lerobot_session.py --root "$DATASET_ROOT" --fps "$FPS" \
    --num-episodes "$NUM_EPISODES" --episode-time-s "$EPISODE_TIME_S" \
    --reset-time-s "$RESET_TIME_S" --rerun-port "$RERUN_PORT"

echo "==> Fake session done. Viewer window stays open -- Ctrl-C to stop the overlay."
wait "$OVERLAY_PID"
