# ALPR Production — Multi-Camera Licence Plate Recognition

Production-ready Automatic Licence Plate Recognition system supporting multiple parallel camera streams, real-time MJPEG streaming, and PostgreSQL persistence with a live web dashboard.

## Features

- **Multi-camera support** — unlimited cameras, each in its own thread
- **Primary cameras** — full pipeline: plate + vehicle colour + brand (CLIP zero-shot)
- **Secondary cameras** — lightweight pipeline: plate detection only (~3× faster)
- **Live dashboard** — MJPEG streams, per-camera detection tables, live DB feed panel
- **PostgreSQL persistence** — upsert logic: one row per (plate, camera), highest-confidence colour/brand always kept
- **Graceful track finalization** — records written when vehicle leaves frame or crosses entry line
- **RTSP auto-reconnect** — streams retry automatically after drops
- **File/video source support** — point cameras at local `.mkv`/`.mp4` files for testing
- **Detection zone filtering** — configurable upper/entry lines to ignore distant or already-passed vehicles

## Architecture

```
ALPR_Production/
├── models.py        # ML model loading + inference (YOLO, EfficientNet-B0, CLIP, fast_alpr)
├── tracker.py       # ByteTrack wrapper, detection zone logic, TrackStore
├── database.py      # PostgreSQL connection pool, schema, upsert logic
├── pipeline.py      # Per-camera processing loop (primary / secondary)
├── server.py        # FastAPI: camera management, streaming, status API
├── cameras.json     # Camera definitions loaded at startup
├── bytetrack.yaml   # Tuned ByteTrack config (track_buffer=90)
├── models/          # Place model weight files here
│   ├── yolo11n.pt
│   ├── color_classifier.pth
│   └── color_classes.json
└── static/
    └── index.html   # Live dashboard UI
```

## ML Models

| Model | Purpose | Framework |
|---|---|---|
| YOLOv11-nano | Vehicle detection + ByteTrack tracking | PyTorch / Ultralytics |
| EfficientNet-B0 | Vehicle colour classification (15 classes) | PyTorch |
| CLIP (ViT-B/32) | Zero-shot car brand classification | HuggingFace Transformers |
| fast-alpr (YOLOv9) | Licence plate detection + OCR | ONNX Runtime |

## Setup

### Requirements

- Python 3.10–3.12
- PostgreSQL 14+
- `torch >= 2.2`, `transformers == 4.40.x` (CLIP requires this exact range with torch 2.2)

### 1. Create a virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Note:** `transformers==4.40.0` is pinned in `requirements.txt`. Do **not** upgrade to 5.x — it requires torch ≥ 2.4 which breaks CLIP loading on torch 2.2.

### 2. Set up PostgreSQL

```bash
# Create database and user
psql -U postgres -c "CREATE DATABASE alpr;"
psql -U postgres -c "CREATE USER alpr_user WITH PASSWORD 'yourpassword';"
psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE alpr TO alpr_user;"

export DATABASE_URL=postgresql://alpr_user:yourpassword@localhost:5432/alpr
```

The `detections` table and indexes are created automatically on first startup.  
**The table is truncated (cleared) on every server restart** — detections are session-scoped by design.

### 3. Configure cameras

Edit `cameras.json`:

```json
[
  {
    "name":       "Gate-A-Primary",
    "type":       "primary",
    "source":     "rtsp://192.168.1.10/stream",
    "upper_line": 0.20,
    "entry_line": 0.70
  },
  {
    "name":       "Gate-B-Secondary",
    "type":       "secondary",
    "source":     "rtsp://192.168.1.11/stream"
  }
]
```

| Field | Description |
|---|---|
| `name` | Unique camera identifier shown in the dashboard |
| `type` | `primary` (full pipeline) or `secondary` (plate only) |
| `source` | RTSP URL or absolute/relative path to a video file |
| `upper_line` | 0.0–1.0 fraction of frame height — vehicles above this are ignored. Default: `0.0` |
| `entry_line` | 0.0–1.0 fraction of frame height — crossing this finalizes the record. Default: `1.0` |

For **secondary cameras** (exit/overview), set `upper_line: 0.0` and `entry_line: 1.0` to use the full frame with no zone filtering. Records are written when vehicles disappear from frame.

### 4. Start the server

```bash
DATABASE_URL=postgresql://alpr_user:yourpassword@localhost:5432/alpr \
  uvicorn server:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` for the live dashboard.

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Dashboard UI |
| POST | `/cameras` | Register + start a new camera |
| DELETE | `/cameras/{name}` | Stop + remove a camera |
| GET | `/cameras` | List all cameras with status and progress |
| GET | `/status` | Live detection rows for all cameras |
| GET | `/status/{name}` | Live rows for one camera |
| GET | `/stream/{name}` | MJPEG live stream (annotated) |
| GET | `/detections` | Recent DB records (`?camera=X&limit=N`) |
| GET | `/stats` | Per-camera aggregate stats |
| GET | `/health` | Liveness check |

## Camera types

### Primary camera
- Pipeline: YOLO tracking → plate reader → colour classifier → CLIP brand
- DB: plate, colour, colour_conf, brand, brand_conf
- Use for: entrance gates, full-vehicle visible angles

### Secondary camera
- Pipeline: YOLO tracking → plate reader only
- DB: plate only (colour/brand stored as NULL); always records even without a plate match
- Use for: exit gates, secondary lanes, overview cameras
- ~3× faster than primary — suitable for high-throughput lanes

## Database schema

```sql
CREATE TABLE detections (
    id           SERIAL PRIMARY KEY,
    camera_name  TEXT        NOT NULL,
    camera_type  TEXT        NOT NULL,
    plate        TEXT,
    colour       TEXT,
    colour_conf  REAL,
    brand        TEXT,
    brand_conf   REAL,
    detected_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    video_time   TEXT
);
-- Unique: one row per (plate, camera_name) when plate IS NOT NULL
-- On conflict: highest colour_conf and brand_conf values are kept (upsert)
```

## Detection zone

```
┌──────────────────────┐
│   ignored (distant)  │
│ ═══ UPPER_LINE ════  │  ← vehicle centre must be below this to be tracked
│                      │
│   classify zone      │  ← colour / brand / plate accumulate here
│                      │
│ ═══ ENTRY_LINE ════  │  ← bottom edge crossing this finalizes the record
│   ignored (close)    │
└──────────────────────┘
```

Both lines are drawn on the MJPEG stream as coloured overlays.

## Notes

- Each camera gets its **own YOLO instance** — ByteTrack state is not shared between threads
- Tracks that disappear from frame (without crossing the entry line) are auto-finalized and written to DB
- Remaining tracks at video-end are flushed to DB before the pipeline exits
- RTSP streams reconnect automatically after a drop (3s delay)
- ByteTrack `track_buffer=90` frames — reduces ID switches during occlusion
- Colour: probability accumulation across frames — resistant to single bad frames
- Brand: strict-max confidence — a high-confidence reading is never overwritten by a lower one
- DB upsert: `ON CONFLICT (plate, camera_name)` — only the best colour and brand confidences are kept per plate per camera

## Production Checklist

> Changes required before deploying to production:

- [ ] **Database URL** — set `DATABASE_URL` via environment variable or secrets manager; never hardcode credentials
- [ ] **Do not truncate on startup** — remove the `TRUNCATE TABLE detections RESTART IDENTITY` line in `database.py:init_db()` to persist data across restarts
- [ ] **RTSP sources** — replace video file paths in `cameras.json` with actual `rtsp://` stream URLs
- [ ] **HTTPS / reverse proxy** — run behind Nginx or Caddy with TLS; do not expose port 8000 directly
- [ ] **Authentication** — the dashboard and API have no authentication; add OAuth2 / API key middleware for production
- [ ] **MJPEG stream security** — MJPEG streams are unauthenticated by default; restrict at the proxy layer
- [ ] **GPU acceleration** — set `device=cuda` in `models.py` for YOLO and EfficientNet; use a CUDA-enabled torch build
- [ ] **uvicorn workers** — use `gunicorn -k uvicorn.workers.UvicornWorker` with multiple workers, or run behind a process manager (systemd, supervisor)
- [ ] **Log rotation** — configure logging handlers with rotation; default logging goes to stdout only
- [ ] **Model storage** — store model weights in a volume-mounted path, not inside the container/repo
- [ ] **transformers version** — keep `transformers==4.40.x` unless upgrading torch to ≥ 2.4 at the same time
- [ ] **ByteTrack buffer** — tune `track_buffer` in `bytetrack.yaml` for your camera FPS and expected vehicle dwell time
