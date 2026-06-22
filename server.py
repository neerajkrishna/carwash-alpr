"""
server.py — FastAPI multi-camera ALPR server
=============================================
Manages multiple camera pipelines running in parallel.
Each camera is registered via POST /cameras or loaded from cameras.json at startup.

Endpoints:
  GET  /                        → dashboard UI
  POST /cameras                 → register a new camera and start its pipeline
  DELETE /cameras/{name}        → stop and remove a camera
  GET  /cameras                 → list all registered cameras and their status
  GET  /status/{name}           → live detection rows + progress for one camera
  GET  /status                  → combined status for all cameras
  GET  /stream/{name}           → MJPEG live stream for one camera
  GET  /detections              → recent rows from PostgreSQL (optional ?camera=X&limit=N)
  GET  /stats                   → per-camera aggregate stats from PostgreSQL
  GET  /health                  → liveness check

Camera config format (POST /cameras body or cameras.json entry):
  {
    "name":       "Gate-A-Primary",   required
    "type":       "primary",          "primary" or "secondary", default "primary"
    "source":     "rtsp://...",       required — RTSP URL or file path
    "upper_line": 0.20,               optional, default 0.0 (full frame)
    "entry_line": 0.70,               optional, default 1.0 (no tripwire)
  }
"""

import os
import json
import asyncio
import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request

from database import init_db, fetch_recent, fetch_stats
from pipeline import CameraPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("server")

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="ALPR Production Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_BASE        = os.path.dirname(os.path.abspath(__file__))
CAMERAS_JSON = os.path.join(_BASE, "cameras.json")

# Active pipelines: name → CameraPipeline
_cameras: dict[str, CameraPipeline] = {}


# ── Startup / shutdown ─────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """
    On startup:
      1. Connect to PostgreSQL and ensure schema exists.
      2. Load cameras.json if it exists and start all defined cameras.
    """
    try:
        init_db()
    except RuntimeError as e:
        logger.error(f"Database init failed: {e}")

    if os.path.exists(CAMERAS_JSON):
        with open(CAMERAS_JSON) as f:
            camera_configs = json.load(f)
        for cfg in camera_configs:
            _start_camera(cfg)
        logger.info(f"Loaded {len(camera_configs)} camera(s) from cameras.json")


@app.on_event("shutdown")
async def shutdown():
    """Stop all running camera pipelines cleanly on server shutdown."""
    for cam in _cameras.values():
        cam.stop()


# ── Camera management ──────────────────────────────────────────────────────────

def _start_camera(config: dict) -> CameraPipeline:
    """Instantiate, register and start a CameraPipeline."""
    name = config["name"]
    if name in _cameras:
        raise ValueError(f"Camera '{name}' is already registered.")
    cam = CameraPipeline(config)
    _cameras[name] = cam
    cam.start()
    return cam


@app.post("/cameras", summary="Register and start a new camera pipeline")
async def add_camera(config: dict[str, Any]):
    """
    Register a new camera and immediately start processing.
    Returns the camera name and type on success.
    """
    if "name" not in config or "source" not in config:
        raise HTTPException(400, "config must include 'name' and 'source'")
    name = config["name"]
    if name in _cameras:
        raise HTTPException(409, f"Camera '{name}' already exists")
    try:
        _start_camera(config)
    except Exception as e:
        raise HTTPException(500, str(e))
    return {"ok": True, "name": name, "type": config.get("type", "primary")}


@app.delete("/cameras/{name}", summary="Stop and remove a camera pipeline")
async def remove_camera(name: str):
    if name not in _cameras:
        raise HTTPException(404, f"Camera '{name}' not found")
    _cameras[name].stop()
    del _cameras[name]
    return {"ok": True, "name": name}


@app.get("/cameras", summary="List all registered cameras")
async def list_cameras():
    return [
        {
            "name":       name,
            "type":       cam.cam_type,
            "source":     cam.source,
            "upper_line": cam.upper_line,
            "entry_line": cam.entry_line,
            "running":    cam.running,
            "done":       cam.done,
            "progress":   cam.progress,
        }
        for name, cam in _cameras.items()
    ]


# ── Status / live data ─────────────────────────────────────────────────────────

@app.get("/status/{name}", summary="Live detection rows for one camera")
async def camera_status(name: str):
    if name not in _cameras:
        raise HTTPException(404, f"Camera '{name}' not found")
    cam = _cameras[name]
    return {
        "name":     name,
        "type":     cam.cam_type,
        "rows":     cam.log_rows,
        "progress": cam.progress,
        "done":     cam.done,
    }


@app.get("/status", summary="Combined live status for all cameras")
async def all_status():
    return {
        name: {
            "type":     cam.cam_type,
            "rows":     cam.log_rows,
            "progress": cam.progress,
            "done":     cam.done,
        }
        for name, cam in _cameras.items()
    }


# ── MJPEG streaming ────────────────────────────────────────────────────────────

@app.get("/stream/{name}", summary="MJPEG live stream for one camera")
async def stream(name: str):
    """
    Returns a multipart/x-mixed-replace MJPEG stream.
    The browser <img> tag can point directly to this URL.
    Frames are pushed as fast as the pipeline encodes them (~every 2nd frame).
    """
    if name not in _cameras:
        raise HTTPException(404, f"Camera '{name}' not found")

    cam = _cameras[name]

    async def frame_generator():
        last_jpeg = b""
        while cam.running:
            jpeg = cam.latest_jpeg
            if jpeg and jpeg != last_jpeg:
                last_jpeg = jpeg
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
            await asyncio.sleep(0.04)  # ~25 fps max push rate

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── Database queries ───────────────────────────────────────────────────────────

@app.get("/detections", summary="Recent detections from the database")
async def get_detections(camera: str | None = None, limit: int = 100):
    """Fetch recent finalized detections. Optionally filter by camera name."""
    try:
        rows = fetch_recent(camera_name=camera, limit=limit)
        # Convert datetime objects to ISO strings for JSON serialisation
        for r in rows:
            if r.get("detected_at"):
                r["detected_at"] = r["detected_at"].isoformat()
        return rows
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/stats", summary="Per-camera detection statistics")
async def get_stats():
    """Aggregate stats per camera: total detections and unique plates."""
    try:
        return fetch_stats()
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Health check ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "cameras": len(_cameras)}


# ── Static files + UI ──────────────────────────────────────────────────────────

app.mount(
    "/static",
    StaticFiles(directory=os.path.join(_BASE, "static")),
    name="static",
)


@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(_BASE, "static", "index.html")) as f:
        return f.read()
