# VizAudit Research Report

Pre-implementation research conducted June 16, 2026. All findings are from direct source
fetches unless flagged with ⚠️ UNVERIFIED.

---

## 1. huggingface/lerobot (v0.5.2)

### Visualization stack

Confirmed: **Rerun** (`rerun-sdk >= 0.24.0, < 0.27.0`), declared under the `viz` extra in
`pyproject.toml`. OpenCV is present (`opencv-python-headless >= 4.9.0`) but headless-only —
it handles video I/O, not GUI display. Matplotlib is an optional extra (`matplotlib-dep`),
not used for live display.

The migration from OpenCV GUI to Rerun happened in PR #903.

Relevant source files:
- `pyproject.toml` — version pinning and extras
- `src/lerobot/scripts/lerobot_record.py` — imports `init_rerun`, `log_rerun_data`
- `src/lerobot/utils/visualization_utils.py` — defines `init_rerun()` and `log_rerun_data()`

### How `--display_data` works

`display_data: bool = False` is a field in the `RecordConfig` dataclass
(`src/lerobot/scripts/lerobot_record.py`). When `True`:

1. At startup: `init_rerun(session_name="recording", ip=cfg.display_ip, port=cfg.display_port)`
2. `init_rerun()` either **spawns a local Rerun viewer** (no IP/port) or **connects to a
   remote gRPC server** (IP/port provided). It also sets environment variables for flush
   size and memory limits before initializing.
3. Per frame in the recording loop: `log_rerun_data(observation=obs_processed, action=action_values, compress_images=...)`
4. `log_rerun_data()` namespaces keys under `observation.` or `action.`, logs scalars via
   `rr.Scalars`, 3-channel arrays as images (CHW→HWC transpose), 1D arrays as indexed scalars.

Additional related config fields:
- `display_ip: Optional[str]` — IP of remote Rerun server
- `display_port: Optional[int]` — port of remote Rerun server
- `display_compressed_images: bool` — JPEG compression (auto-enabled when connecting remotely)

⚠️ UNVERIFIED: The exact default gRPC port used when spawning locally. Rerun changed its
port scheme between v0.19 and v0.24. Needs one live test to confirm. Likely 9876.

⚠️ UNVERIFIED: What the Rerun viewer actually renders — whether it shows camera feeds only
or also joint state/action plots. Requires running with real hardware to confirm.

### Attaching a second process to the same Rerun viewer

The spawned viewer accepts incoming gRPC connections from other processes — this is a
first-class Rerun use case. Two approaches:

**Option A — Attach to the spawned viewer (simplest UX):**
```
# Process 1
lerobot-record --display_data
# Process 2
rr.connect_grpc("rerun+grpc://127.0.0.1:9876")  # connects to same viewer
```
The two processes appear as separate recordings (different tabs) in the viewer since they
have different `application_id`s. Both visible in the same window.

**Option B — Shared server, single merged recording:**
```bash
# Start a Rerun server first
python -m rerun --serve-web --port 9876

# Both processes point at it
lerobot-record --display_data --display_ip 127.0.0.1 --display_port 9876
vizaudit overlay --connect 127.0.0.1:9876
```
Data from both processes merges into one timeline. The coverage heatmap and camera feed
share one timeline. This is the cleanest UX for the operator.

Recommendation: use Option B during development (explicit port, no guessing). Simplify to
Option A once confirmed working so users don't need to pre-start a server.

### LeRobotDataset API

File: `src/lerobot/common/datasets/lerobot_dataset.py`

**Reading data:**
```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

# Inspect without full download
meta = LeRobotDatasetMetadata("your/dataset")
print(meta.total_episodes, meta.fps, meta.robot_type)

# Load specific episodes
dataset = LeRobotDataset("your/dataset", episodes=[0, 1, 2])
frame = dataset[0]          # dict of torch.Tensor
frames = dataset[10:20]     # slicing supported
len(dataset)                # total frames

# Temporal windows (load past/future context per frame)
delta_timestamps = {
    "observation.state": [-0.5, -0.25, 0],
    "action": [0.0, 0.016, 0.033],
}
dataset = LeRobotDataset("your/dataset", delta_timestamps=delta_timestamps)
```

**Recording data:**
```python
dataset = LeRobotDataset.create(repo_id="your/dataset", fps=30, ...)
dataset.add_frame(frame_dict)
dataset.save_episode()
dataset.push_to_hub()
```

Key properties: `.meta.episodes`, `.fps`, `.robot_type`, `.meta.features`

PyTorch `DataLoader`-compatible (supports `num_workers`, `batch_size`, `shuffle`).

### Dataset format stability

| Format | Status |
|--------|--------|
| v2.0 | Stable — use this |
| v2.1 | Stable |
| v3.0 | **Actively breaking** — do not target yet |

v3.0 (PR #969) changes from single episode files to chunked multi-episode files. Known
active issues as of June 2026: NaN conversion error in `lerobot-edit-dataset` when
deleting/splitting episodes, no migration script from v2.0, Groot policy incompatible.

Porting guide exists at `docs/source/porting_datasets_v3.mdx` but v3.0 is not
production-ready.

**Other requirements:** Python >= 3.12, torch >= 2.7.

⚠️ UNVERIFIED: Whether the standard Parquet schema includes Cartesian end-effector pose
columns (e.g., `observation.eef_pos`, `observation.eef_rot`) or only joint angles. This is
load-bearing for the audit tool — if only joint angles are present, forward kinematics is
needed to get spatial positions. Verify by inspecting `dataset.meta.features` on a real
dataset before committing to a detection approach.

---

## 2. huggingface/leLab

### Architecture

React 18.3.1 + FastAPI web application (not Electron, not a desktop app).

- Frontend: Vite + TypeScript, runs on port **8080**
- Backend: FastAPI + uvicorn (Python 3.12+), runs on port **8000**
- Real-time data: single WebSocket at `/ws/joint-data` using a `ConnectionManager` with a
  queue-based broadcast system
- State persists at `~/.cache/huggingface/lerobot/`

Has a comprehensive `CLAUDE.md` at the repo root.

### Record/teleoperate flow

| File | Size | Responsibility |
|------|------|----------------|
| `lelab/teleoperate.py` | 13 KB | Leader→follower loop at ~20 FPS; broadcasts joint data via WebSocket |
| `lelab/record.py` | 37 KB | Multi-phase recording state machine (Preparing→Recording→Resetting→Completion) |
| `lelab/server.py` | 44 KB | All HTTP endpoints + WebSocket; wires features together |

HTTP endpoints (partial): `POST /start-recording`, `POST /stop-recording`,
`GET /recording-status`, `POST /upload-dataset`, `POST /start-teleoperation`,
`POST /move-arm`, `GET /joint-positions`, `POST /start-inference`, `GET /system/cuda-status`.

WebSocket message format: `{"type": "joint_update", "joints": {...}, "timestamp": ...}`
broadcast every 50 ms.

### Existing visualization

- **3D URDF viewer** (`frontend/src/components/UrdfViewer.tsx`): Three.js-based, loads
  SO-101 model, real-time joint highlighting, WebSocket status indicator.
- **MetricsPanel** (`frontend/src/components/control/MetricsPanel.tsx`): live webcam feed,
  voice activity visualization, 6 Recharts motor graphs.
- **Recording page** (`frontend/src/pages/Recording.tsx`, 20.7 KB): episode counter,
  phase-specific timer, animated progress bar. Status-focused, no camera feed.

No workspace overlay or per-episode spatial coverage display exists.

### Extension points

- `UrdfContext` (`frontend/src/contexts/UrdfContext.tsx`, 13.5 KB): callback registration
  via `onUrdfDetected()`, supports custom `UrdfProcessor` implementations, animatable state.
- `useRealTimeJoints` hook (`frontend/src/hooks/useRealTimeJoints.ts`, 3.6 KB): WebSocket
  connection to `/ws/joint-data` with auto-reconnect (exponential backoff 1 s → 30 s cap).
  Any component can subscribe to joint updates via this hook.
- `ApiContext` (`frontend/src/contexts/ApiContext.tsx`): provides `baseUrl`, `wsBaseUrl`,
  `fetchWithHeaders()`. Runtime-configurable via query params → localStorage → defaults.
- `apiClient.ts` (`frontend/src/lib/apiClient.ts`): generic `apiRequest()` + `ApiError` class.

Adding a new panel: create `frontend/src/pages/MyPanel.tsx`, wire into `App.tsx` routes,
use existing hooks (`useRealTimeJoints`, `useAvailableCameras`), optionally add a new
FastAPI endpoint in `server.py`.

### Critical constraint

**leLab is hardcoded for SO-101 arms throughout all feature modules.** This is documented
explicitly in its `CLAUDE.md`. Supporting other robots requires modifying every feature
module. This makes leLab unsuitable as the primary home for a robot-agnostic spatial
diversity tool.

---

## 3. huggingface/lerobot-dataset-visualizer

### Tech stack

Next.js 15.3.6 + React 19.0.0 + TypeScript 5 + Tailwind CSS 4 + Recharts 2.15.3 +
Three.js 0.182.0 + @react-three/fiber 9.5.0. Parquet files read via hyparquet 1.12.1.
HuggingFace Hub integration via @huggingface/hub 2.11.0. **Build tool: Bun** (not npm/yarn).
Optional FastAPI backend for annotation editing.

Actively maintained — last commit June 11, 2026 (PR #108, annotation features by Pepijn).
Apache 2.0 license. 8 open issues, 56 forks, 98 stars.

### Plugin/panel architecture

No formal plugin system. 16 panel components in `src/components/` are directly imported and
composed in page components. No panel registry, no config-driven layout, no plugin loader.

Data flows through React Context providers (`TimeProvider`, `AuthProvider`,
`FlaggedEpisodesContext`). Extension means: write a new `.tsx` component, manually import
it into the page component where other panels live.

### Action Insights panel

File: `src/components/action-insights-panel.tsx` (56.3 KB)

Computes **5 metrics** — all in action/state *time-series space*, none spatial/positional:

1. **Action Autocorrelation** — identifies natural chunk boundaries where autocorrelation
   drops below 0.5, showing how correlated each action dimension is with itself across lags.
2. **Action Velocity (Smoothness)** — frame-to-frame Δa: standard deviation, max absolute
   change, histogram. Classifies demos as smooth/moderate/jerky.
3. **Cross-Episode Variance Heatmap** — how action dimensions vary across episodes at each
   timestep; highlights consistency and multimodality.
4. **Demonstrator Speed Variance** — coefficient of variation in execution speeds across
   episodes; recommends velocity normalization when variance exceeds threshold.
5. **State–Action Temporal Alignment** — cross-correlation between action and state changes
   across lags; detects control delays and misalignment.

**No overlap with spatial diversity metrics.** A spatial diversity panel would be entirely
additive.

⚠️ UNVERIFIED: Whether the episode viewer page file (`episode-viewer.tsx`) exposes a clean
insertion point — the file returned 404 during research (likely path encoding issue with
brackets in the filename). Needs manual inspection of the actual file tree.

⚠️ UNVERIFIED: Whether Cartesian pose data is available in the standard Parquet schema. A
spatial panel in this frontend tool would need that data available without a separate Python
preprocessing step.

---

## 4. Object detection and orientation on GTX 1660 Ti (6 GB VRAM)

### Detection models

| Model | VRAM | 30 fps on 1660 Ti? | Notes |
|-------|------|--------------------|-------|
| YOLOv8n | ~0.5 GB | Yes | Recommended default |
| YOLOv11n | ~0.5–1.0 GB | Yes | Similar to YOLOv8n |
| YOLOv8s | ~1.0–1.5 GB | Marginal | May struggle at 640px |
| YOLOv8n INT8 | ~0.25 GB | Yes | Best VRAM efficiency |

pip-installable via `ultralytics`.

⚠️ UNVERIFIED: No published benchmarks for GTX 1660 Ti specifically. Estimates extrapolated
from RTX 3000/4000 series scaling. FP16 and INT8 quantization reduce VRAM ~50% and ~75%
respectively with minimal accuracy loss.

### Segmentation models

| Model | Params | VRAM estimate | Fits 6 GB? | Available via |
|-------|--------|---------------|------------|---------------|
| MobileSAM | <10M | ~0.5 GB | Yes | `pip install ultralytics` |
| EfficientSAM-S | 25M | ~0.8–1.2 GB | Yes | GitHub + HuggingFace Hub |
| FastSAM | 68M | ~1.2–1.8 GB | Yes | `facebookresearch/fast-segment-anything` |
| SAM ViT-B | 91M | ~3–4 GB | Yes (tight) | `pip install segment-anything` |
| SAM ViT-L | 308M | ~5–6 GB | Borderline | Not recommended |
| SAM ViT-H | 632M | ~7.0 GB | **No** | Exceeds budget |

YOLO + MobileSAM together: ~1 GB combined. Leaves 5 GB for OS and other processes.

⚠️ UNVERIFIED: EfficientSAM VRAM figures are estimated from parameter counts; no official
GB specifications published.

### Classical OpenCV orientation (contour → PCA → `minAreaRect`)

**Viable for controlled robot workspaces:**
- Fixed camera, solid background, good lighting, isolated elongated objects
- Latency: <5 ms, CPU-only, no GPU required
- Deterministic, fully interpretable (can visualize contours and PCA axes)

**Fails when:**
- Background has similar intensity/color to object
- Partial occlusion or overlapping objects
- Textured backgrounds or variable lighting
- `minAreaRect` has known edge-case bugs: opencv/opencv issue #11915

**Mitigation:** HSV color thresholding before contour detection, morphological preprocessing
(erosion/dilation), contour filtering by area and solidity.

### 6D pose estimation

**FoundPose** (ECCV 2024) is the only realistic option for 6 GB VRAM:
- Uses frozen DINOv2 features; model itself is 0.4 MB
- Estimated total VRAM (model + DINOv2 encoder): ~1–2 GB
- No training needed; works on unseen objects
- **Not pip-installable** — requires manual GitHub clone and conda environment setup

No mainstream lightweight 6D pose model exists as a simple `pip install`. Do not target
6D pose in the initial version.

### Recommended approach

1. **Start with classical CV** (OpenCV contour → PCA) for clean workspaces. It's zero-cost,
   deterministic, and sufficient for simple elongated tools on a solid background.
2. **Fall back to YOLOv8n + MobileSAM** when robustness is needed (multiple objects,
   variable lighting, partial occlusion). Total overhead ~1 GB VRAM.
3. **Do not implement 6D pose** initially — no pip-installable solution exists.

Hybrid pipeline: use learned segmentation to get the object mask, then run OpenCV PCA on
the masked contour for orientation. Best of both: semantic robustness + lightweight orientation.

---

## 5. Where the tool should live

**Recommendation: standalone Python package (`vizaudit`), not a contribution to leLab or
lerobot-dataset-visualizer.**

| Option | Why it doesn't fit |
|--------|-------------------|
| leLab contribution | Hardcoded for SO-101 throughout; robot-agnostic tool would require touching all feature modules |
| lerobot-dataset-visualizer | TypeScript/Bun stack; no plugin system; Cartesian pose data not available at frontend layer without a Python backend step |
| lerobot core | Dataset API is in flux (v3.0 breaking); premature to contribute until format stabilizes |

**A standalone package wins because:**
- Integrates with Rerun — same SDK lerobot already uses, same viewer the operator has open
- Reads `LeRobotDataset` directly — pure Python, works with any robot type
- Can be contributed back to lerobot later (`lerobot/utils/spatial_audit.py`) once v3.0
  stabilizes, since the interface will already be clean
- The dataset visualizer's spatial panel would need a Python computation backend anyway —
  that backend computation step *is* this tool

---

## 6. Open questions requiring manual testing

| Question | Why it matters |
|----------|----------------|
| Default Rerun gRPC port after `rr.spawn()` in lerobot ≥ 0.24 | Required for Option A (attach to spawned viewer) |
| What `--display_data` actually renders in the Rerun viewer | Determines how to integrate the overlay visually |
| Does `dataset.meta.features` include `eef_pos`/`eef_rot` for SO-101 datasets? | If not, FK or external detection needed before audit can extract spatial positions |
| `episode-viewer.tsx` file structure in lerobot-dataset-visualizer | Needed if a panel contribution is considered later |
| `minAreaRect` reliability on your specific objects under your lighting | Determines whether classical CV path is viable without a learned fallback |
