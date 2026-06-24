"""
pipeline.py — Per-camera video processing pipeline
====================================================
Each camera runs its own CameraPipeline instance in a dedicated thread.

Camera types:
  PRIMARY   — Full pipeline: plate + colour + brand detection.
              All fields written to the database.
  SECONDARY — Plate-only pipeline. Colour/brand skipped entirely.
              Much faster — useful for exit gates or secondary lanes.

Frame processing rate:
  - Vehicle tracking: every frame
  - Plate detection:  every PLATE_EVERY_N frames
  - Colour + brand:   every CLASSIFY_EVERY_N frames (primary only)

RTSP reconnect:
  If the video source drops (common with IP cameras), the pipeline
  automatically attempts to reconnect after RECONNECT_DELAY_S seconds.

Zone lines:
  upper_line and entry_line are optional. If not provided:
    - upper_line defaults to 0.0 (no upper boundary — full frame)
    - entry_line defaults to 1.0 (no tripwire — records never auto-finalize)
"""

import os
import cv2
import time
import logging
import threading
import numpy as np
from datetime import datetime

from ultralytics import YOLO
from models import detect_colour, detect_brand, read_plates, YOLO_PATH
from tracker import get_tracks, TrackStore, match_plate_to_vehicle, draw_overlay
from database import insert_detection

logger = logging.getLogger("pipeline")

# ── Tunable constants ──────────────────────────────────────────────────────────
CLASSIFY_EVERY_N   = 5    # Run colour/brand every N frames
PLATE_EVERY_N      = 5    # Run plate reader every N frames
MAX_VIDEO_WIDTH    = 1280  # Downscale frames wider than this (saves CPU)
JPEG_QUALITY       = 70    # MJPEG stream quality
RECONNECT_DELAY_S  = 3.0   # Seconds to wait before reconnecting a dropped stream


class CameraPipeline:
    """
    Manages the full detection pipeline for a single camera.

    Usage:
        cam = CameraPipeline(config={
            "name": "Gate-A-Primary",
            "type": "primary",
            "source": "rtsp://192.168.1.10/stream",
            "upper_line": 0.20,
            "entry_line": 0.70,
        })
        cam.start()
        # cam.latest_jpeg  → current annotated JPEG bytes
        # cam.log_rows     → list of detection dicts for the UI
        cam.stop()
    """

    def __init__(self, config: dict):
        # Camera identity
        self.name        = config["name"]
        self.cam_type    = config.get("type", "primary").lower()  # "primary" or "secondary"
        _src = config["source"]
        if not str(_src).startswith("rtsp://") and not os.path.isabs(_src):
            _src = os.path.join(os.path.dirname(os.path.abspath(__file__)), _src)
        self.source      = _src

        # Detection zone — defaults mean full frame (no filtering)
        self.upper_line  = float(config.get("upper_line", 0.0))
        self.entry_line  = float(config.get("entry_line", 1.0))

        # Per-camera YOLO detector (own instance so ByteTrack state is not shared)
        self._detector   = YOLO(YOLO_PATH)

        # Runtime state
        self.running     = False
        self._thread     = None
        self._lock       = threading.Lock()

        # Shared output — read by server.py for streaming and status API
        self.latest_jpeg: bytes       = b""
        self.log_rows:    list[dict]  = []  # live detection records for UI
        self.progress:    int         = 0   # 0-100, only meaningful for file sources
        self.done:        bool        = False

    # ── Public control ─────────────────────────────────────────────────────────

    def start(self):
        """Start the pipeline in a background thread."""
        self.running = True
        self.done    = False
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"cam-{self.name}")
        self._thread.start()
        logger.info(f"[{self.name}] Pipeline started (type={self.cam_type})")

    def stop(self):
        """Signal the pipeline to stop and wait for the thread to exit."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info(f"[{self.name}] Pipeline stopped.")

    # ── Internal processing loop ───────────────────────────────────────────────

    def _run(self):
        """
        Main loop. Opens the video source, processes frames, handles reconnects.
        For file sources (mp4/avi), sets progress and marks done when finished.
        For RTSP streams, loops indefinitely with reconnect on failure.
        """
        is_file = not str(self.source).startswith("rtsp://")

        while self.running:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                logger.warning(f"[{self.name}] Could not open source: {self.source}. Retrying in {RECONNECT_DELAY_S}s...")
                time.sleep(RECONNECT_DELAY_S)
                continue

            fps       = cap.get(cv2.CAP_PROP_FPS) or 25.0
            raw_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            scale     = min(1.0, MAX_VIDEO_WIDTH / raw_w) if raw_w > MAX_VIDEO_WIDTH else 1.0
            total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            frame_idx = 0

            track_store = TrackStore()
            last_plates: list[dict] = []
            active_tids: set[int]   = set()   # track IDs seen in previous frame

            logger.info(f"[{self.name}] Stream opened. fps={fps:.1f} scale={scale:.2f}")

            try:
                while self.running:
                    ret, frame = cap.read()
                    if not ret:
                        if is_file:
                            logger.info(f"[{self.name}] Video file ended.")
                            self.done = True
                        else:
                            logger.warning(f"[{self.name}] Stream dropped. Reconnecting...")
                        break

                    frame_idx += 1
                    if scale < 1.0:
                        frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

                    # Compute human-readable timestamp within the video
                    ts   = frame_idx / fps
                    time_str = f"{int(ts)//60:02d}:{int(ts)%60:02d}"

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # ── Plate detection (every PLATE_EVERY_N frames) ────────────
                    if frame_idx % PLATE_EVERY_N == 0:
                        last_plates = read_plates(frame_rgb)

                    # ── Vehicle tracking ────────────────────────────────────────
                    tracks = get_tracks(frame, self.upper_line, self.entry_line, self._detector)
                    current_tids = {t["track_id"] for t in tracks}

                    # ── Finalize tracks that disappeared (left frame) ────────────
                    for gone_tid in active_tids - current_tids:
                        if not track_store.is_finalized(gone_tid):
                            track_store.finalize(gone_tid)
                            gone_snap = track_store.snapshot(gone_tid)
                            self._finalize_log_row(gone_tid, gone_snap)
                            self._write_to_db(gone_tid, gone_snap, time_str, require_plate=False)
                    active_tids = current_tids

                    # ── Process each track ──────────────────────────────────────
                    if frame_idx % CLASSIFY_EVERY_N == 1:
                        for t in tracks:
                            tid     = t["track_id"]
                            crossed = t["crossed"]

                            # Detect ByteTrack ID reuse — new vehicle, recycled ID
                            if track_store.is_finalized(tid):
                                if crossed:
                                    continue  # Already past line — skip
                                # ID reused by genuinely new vehicle
                                track_store.handle_reuse(tid)
                                self._remove_log_row(tid)

                            if crossed:
                                # Vehicle crossed entry line — lock the record
                                if not track_store.is_finalized(tid):
                                    track_store.finalize(tid)
                                    snap = track_store.snapshot(tid)
                                    self._finalize_log_row(tid, snap)
                                    self._write_to_db(tid, snap, time_str)
                                continue

                            # ── Classification (primary cameras only) ──────────
                            if self.cam_type == "primary":
                                colour, colour_conf = detect_colour(t["crop"])
                                brand,  brand_conf  = detect_brand(t["crop"])
                                track_store.update_colour(tid, colour, colour_conf, time_str)
                                track_store.update_brand(tid, brand, brand_conf, time_str)

                            # ── Plate matching ──────────────────────────────────
                            matched = match_plate_to_vehicle(last_plates, t["bbox"])
                            if matched:
                                track_store.update_plate(tid, matched["plate"], matched["confidence"])

                            # ── Update live log row ─────────────────────────────
                            snap = track_store.snapshot(tid)
                            self._upsert_log_row(tid, snap, time_str)

                    # ── Update progress (file sources) ──────────────────────────
                    if is_file:
                        self.progress = min(round(frame_idx / total * 100), 99)

                    # ── Encode annotated JPEG for streaming ─────────────────────
                    if frame_idx % 2 == 0:
                        annotated = draw_overlay(
                            frame_rgb, tracks, last_plates, track_store,
                            self.upper_line, self.entry_line, self.name,
                        )
                        ok, buf = cv2.imencode(
                            ".jpg", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
                        )
                        if ok:
                            with self._lock:
                                self.latest_jpeg = buf.tobytes()

            finally:
                cap.release()

            if is_file:
                self.progress = 100
                self.done = True
                break

            if self.running:
                logger.info(f"[{self.name}] Reconnecting in {RECONNECT_DELAY_S}s...")
                time.sleep(RECONNECT_DELAY_S)

    # ── Log row helpers ────────────────────────────────────────────────────────

    def _make_row(self, tid: int, snap: dict, time_str: str, finalized: bool) -> dict:
        return {
            "id":          tid,
            "camera":      self.name,
            "plate":       snap["plate"],
            "colour":      snap["colour"],
            "colour_conf": f"{snap['colour_conf']:.0%}" if snap["colour_conf"] > 0 else "—",
            "brand":       snap["brand"],
            "brand_conf":  f"{snap['brand_conf']:.0%}" if snap["brand_conf"] > 0 else "—",
            "time":        time_str,
            "finalized":   finalized,
        }

    def _upsert_log_row(self, tid: int, snap: dict, time_str: str):
        """Insert or update the live log row for a track."""
        new_row = self._make_row(tid, snap, time_str, finalized=False)
        with self._lock:
            for i, r in enumerate(self.log_rows):
                if r["id"] == tid:
                    # Only update colour/brand if new confidence is higher
                    _cc = r.get("colour_conf", "0%")
                    _bc = r.get("brand_conf",  "0%")
                    old_cc = float(_cc.replace("%", "") or 0) / 100 if isinstance(_cc, str) and _cc not in ("—", "") else 0.0
                    old_bc = float(_bc.replace("%", "") or 0) / 100 if isinstance(_bc, str) and _bc not in ("—", "") else 0.0
                    if snap["colour_conf"] <= old_cc:
                        new_row["colour"]      = r["colour"]
                        new_row["colour_conf"] = r["colour_conf"]
                    if snap["brand_conf"] <= old_bc:
                        new_row["brand"]      = r["brand"]
                        new_row["brand_conf"] = r["brand_conf"]
                    self.log_rows[i] = new_row
                    return
            self.log_rows.append(new_row)

    def _finalize_log_row(self, tid: int, snap: dict):
        """Mark a log row as finalized (locked)."""
        with self._lock:
            for r in self.log_rows:
                if r["id"] == tid:
                    r["finalized"] = True
                    return

    def _remove_log_row(self, tid: int):
        """Remove a log row (called when ByteTrack reuses an ID)."""
        with self._lock:
            self.log_rows = [r for r in self.log_rows if r["id"] != tid]

    # ── Database write ─────────────────────────────────────────────────────────

    def _write_to_db(self, tid: int, snap: dict, time_str: str, require_plate: bool = True):
        """
        Write a finalized detection to PostgreSQL.
        Secondary cameras only write plate — colour/brand are stored as NULL.
        require_plate=False allows saving vehicles that left frame without a plate read,
        but still requires at least a plate or brand to avoid all-empty rows.
        """
        plate = snap.get("plate", "—")
        brand = snap.get("brand", "—")
        if require_plate and (not plate or plate == "—"):
            return
        if not require_plate and (not plate or plate == "—") and (not brand or brand == "—"):
            return
        try:
            insert_detection(
                camera_name=self.name,
                camera_type=self.cam_type,
                plate=snap["plate"],
                colour=snap["colour"]      if self.cam_type == "primary" else None,
                colour_conf=snap["colour_conf"] if self.cam_type == "primary" else None,
                brand=snap["brand"]        if self.cam_type == "primary" else None,
                brand_conf=snap["brand_conf"]  if self.cam_type == "primary" else None,
                video_time=time_str,
            )
        except Exception as e:
            logger.error(f"[{self.name}] DB write failed for track {tid}: {e}")
