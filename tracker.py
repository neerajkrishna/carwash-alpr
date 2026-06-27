"""
tracker.py — Vehicle tracking and detection zone logic
=======================================================
Wraps YOLO ByteTrack to provide:
  - Detection zone filtering (UPPER_LINE / ENTRY_LINE)
  - Per-track state management (colour votes, brand best, plate best)
  - Tripwire finalization: once a vehicle's bottom edge crosses ENTRY_LINE, its record is locked
  - Track ID reuse detection: ByteTrack recycles integer IDs; we detect and handle this correctly

Detection zone diagram:
  ┌────────────────────┐
  │  ignored (distant) │
  │ ══ UPPER_LINE ════ │  ← vehicle centre must be below this to be tracked
  │                    │
  │  classify zone     │  ← colour / brand / plate accumulate here
  │                    │
  │ ══ ENTRY_LINE ════ │  ← bottom edge crossing this locks the record
  │  ignored (close)   │
  └────────────────────┘
"""

import os
import cv2
import numpy as np
from collections import Counter
from models import YOLO_PATH
from ultralytics import YOLO

# ── Constants ──────────────────────────────────────────────────────────────────
VEHICLE_CLASS_IDS = [2, 5, 7]   # COCO: car, bus, truck
MIN_CONFIDENCE    = 0.45         # YOLO detection confidence threshold
MIN_VEHICLE_AREA  = 0.04         # Minimum fraction of frame area — filters tiny/distant vehicles
BYTETRACK_CFG     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bytetrack.yaml")


def get_tracks(frame_bgr: np.ndarray, upper_line: float = 0.0, entry_line: float = 1.0,
               detector: "YOLO" = None, entry_axis: str = "horizontal", lower_line: float = 1.0) -> list[dict]:
    """
    Run YOLO + ByteTrack on a frame and return active vehicle tracks.

    Filters:
      - YOLO confidence < MIN_CONFIDENCE
      - Bounding box area < MIN_VEHICLE_AREA (too small / distant)
      - Vehicle centre above UPPER_LINE (outside detection zone, horizontal axis only)
      - Vehicle already past ENTRY_LINE (already finalized — not returned)

    entry_axis="horizontal": entry_line is a Y fraction — crossed when bottom edge (y2) passes it.
    entry_axis="vertical":   entry_line is an X fraction — crossed when right edge (x2) passes it.
                             Used when cars move left-to-right across the frame.

    Returns list of track dicts:
      {"track_id": int, "bbox": (x1,y1,x2,y2), "crop": ndarray,
       "confidence": float, "area": int, "primary": bool, "crossed": bool}
    """
    fh, fw     = frame_bgr.shape[:2]
    frame_area = fh * fw

    results = detector.track(
        frame_bgr,
        persist=True,
        tracker=BYTETRACK_CFG,
        classes=VEHICLE_CLASS_IDS,
        verbose=False,
    )

    tracks = []
    if not results or results[0].boxes is None or results[0].boxes.id is None:
        return tracks

    for box, tid, conf in zip(results[0].boxes.xyxy,
                               results[0].boxes.id,
                               results[0].boxes.conf):
        if conf.item() < MIN_CONFIDENCE:
            continue

        x1, y1, x2, y2 = map(int, box.tolist())

        # Filter by detection zone boundaries
        if entry_axis == "vertical":
            # upper_line = left X boundary, lower_line = right X boundary
            cx = (x1 + x2) / 2
            if upper_line > 0.0 and (cx / fw) < upper_line:
                continue
            if lower_line < 1.0 and (cx / fw) > lower_line:
                continue
        else:
            cy = (y1 + y2) / 2
            if upper_line > 0.0 and (cy / fh) < upper_line:
                continue
            if lower_line < 1.0 and (cy / fh) > lower_line:
                continue

        # Filter by minimum area
        norm_area = ((x2 - x1) * (y2 - y1)) / frame_area
        if norm_area < MIN_VEHICLE_AREA:
            continue

        crop = frame_bgr[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            continue

        if entry_axis == "vertical":
            crossed = (x1 / fw) >= entry_line  # entire vehicle has passed the line
        else:
            crossed = (y2 / fh) >= entry_line

        tracks.append({
            "track_id":   int(tid.item()),
            "bbox":       (x1, y1, x2, y2),
            "crop":       crop,
            "confidence": conf.item(),
            "area":       (x2 - x1) * (y2 - y1),
            "crossed":    crossed,
        })

    # Mark the largest visible vehicle as "primary" for overlay label
    if tracks:
        primary_idx = max(range(len(tracks)), key=lambda i: tracks[i]["area"])
        for i, t in enumerate(tracks):
            t["primary"] = (i == primary_idx)

    return tracks


class TrackStore:
    """
    Per-camera in-memory state for all active tracks.

    Stores colour vote history, best brand, best plate, and finalization flag
    for each track_id. Handles ByteTrack ID reuse by detecting when a new
    vehicle is assigned an already-finalized ID.
    """

    def __init__(self):
        self._store: dict[int, dict] = {}

    def get(self, tid: int) -> dict:
        return self._store.get(tid, {})

    def is_finalized(self, tid: int) -> bool:
        return self._store.get(tid, {}).get("finalized", False)

    def update_colour(self, tid: int, colour: str, conf: float, time_str: str):
        """
        Accumulate a colour vote for this track.
        Uses probability accumulation: sum confidence weights per label,
        winner = label with highest total weight.
        This prevents one overconfident bad frame from permanently locking a wrong colour.
        """
        entry = self._store.setdefault(tid, self._new_entry())
        entry["colour_votes"].append((colour, conf))

        tally: dict[str, float] = {}
        for lbl, c in entry["colour_votes"]:
            tally[lbl] = tally.get(lbl, 0.0) + c

        best = max(tally, key=tally.__getitem__)
        entry["colour"]      = best
        entry["colour_conf"] = tally[best] / len(entry["colour_votes"])
        entry["colour_time"] = time_str

    def update_brand(self, tid: int, brand: str, conf: float, time_str: str):
        """
        Update brand only if new confidence is strictly higher.
        Brand doesn't flicker like colour so strict-max is appropriate.
        """
        entry = self._store.setdefault(tid, self._new_entry())
        if conf > entry.get("brand_conf", 0.0):
            entry["brand"]      = brand
            entry["brand_conf"] = conf
            entry["brand_time"] = time_str

    def update_plate(self, tid: int, plate: str, conf: float):
        """Update best plate only if new confidence is strictly higher."""
        entry = self._store.setdefault(tid, self._new_entry())
        if conf > entry.get("plate_conf", 0.0):
            entry["plate"]      = plate
            entry["plate_conf"] = conf

    def finalize(self, tid: int):
        """Lock this track — no further updates after this point."""
        entry = self._store.setdefault(tid, self._new_entry())
        entry["finalized"] = True

    def handle_reuse(self, tid: int) -> bool:
        """
        Check if this tid was previously finalized (ByteTrack ID reuse).
        If so, remove the old entry so a fresh track can start.
        Returns True if reuse was detected and cleaned up.
        """
        if self.is_finalized(tid):
            del self._store[tid]
            return True
        return False

    def snapshot(self, tid: int) -> dict:
        """Return a copy of the current best values for a track."""
        e = self._store.get(tid, {})
        return {
            "colour":      e.get("colour", "?"),
            "colour_conf": e.get("colour_conf", 0.0),
            "brand":       e.get("brand", "—"),
            "brand_conf":  e.get("brand_conf", 0.0),
            "plate":       e.get("plate", "—"),
            "plate_conf":  e.get("plate_conf", 0.0),
            "finalized":   e.get("finalized", False),
        }

    @staticmethod
    def _new_entry() -> dict:
        import time
        return {
            "colour": "?", "colour_conf": 0.0, "colour_votes": [],
            "brand": "—",  "brand_conf": 0.0,
            "plate": "—",  "plate_conf": 0.0,
            "finalized": False,
            "first_seen": time.time(),
        }


def match_plate_to_vehicle(plate_results: list[dict], vehicle_bbox: tuple) -> dict | None:
    """
    Find the plate detection that best matches a given vehicle bounding box.

    Priority: overlapping bbox first, then nearest centre-to-centre distance.
    Returns the best matching plate dict or None.
    """
    if not plate_results or vehicle_bbox is None:
        return None

    vx1, vy1, vx2, vy2 = vehicle_bbox
    best, best_dist = None, float("inf")

    for p in plate_results:
        px1, py1, px2, py2 = p["bbox"]
        ox = max(0, min(vx2, px2) - max(vx1, px1))
        oy = max(0, min(vy2, py2) - max(vy1, py1))
        if ox * oy > 0:
            return p  # overlapping bbox — direct match
        cx, cy   = (px1 + px2) / 2, (py1 + py2) / 2
        vcx, vcy = (vx1 + vx2) / 2, (vy1 + vy2) / 2
        dist = ((cx - vcx) ** 2 + (cy - vcy) ** 2) ** 0.5
        if dist < best_dist:
            best_dist, best = dist, p

    return best


def draw_overlay(frame_rgb: np.ndarray, tracks: list[dict], plate_results: list[dict],
                 track_store: "TrackStore", upper_line: float = 0.0, entry_line: float = 1.0,
                 camera_name: str = "", entry_axis: str = "horizontal", lower_line: float = 1.0) -> np.ndarray:
    """
    Draw detection zone lines, vehicle bounding boxes, plate boxes and labels
    onto a copy of the frame. Returns the annotated frame.
    """
    img = frame_rgb.copy()
    fh, fw = img.shape[:2]

    # Draw detection zone lines
    uy = int(upper_line * fh) if upper_line > 0.0 else 0
    ly = int(lower_line * fh) if lower_line < 1.0 else fh

    if entry_axis == "vertical":
        # Vertical left/right boundaries define the detection zone
        lx = int(upper_line * fw) if upper_line > 0.0 else 0
        rx = int(lower_line * fw) if lower_line < 1.0 else fw
        if upper_line > 0.0:
            cv2.line(img, (lx, 0), (lx, fh), (0, 255, 180), 2)
            cv2.putText(img, "ZONE", (lx + 4, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 2)
        if lower_line < 1.0:
            cv2.line(img, (rx, 0), (rx, fh), (0, 255, 180), 2)
        # Vertical entry line spans full height within the zone
        if entry_line < 1.0:
            ev = int(entry_line * fw)
            cv2.line(img, (ev, 0), (ev, fh), (0, 200, 255), 2)
            cv2.putText(img, "ENTRY LINE", (ev + 6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    else:
        if upper_line > 0.0:
            cv2.line(img, (0, uy), (fw, uy), (0, 255, 180), 2)
            cv2.putText(img, "DETECTION ZONE", (8, uy + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 2)
        if entry_line < 1.0:
            ey = int(entry_line * fh)
            cv2.line(img, (0, ey), (fw, ey), (0, 200, 255), 2)
            cv2.putText(img, "ENTRY LINE", (8, ey - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    # Camera name overlay
    if camera_name:
        cv2.putText(img, camera_name, (fw - 200, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Vehicle bounding boxes
    for t in tracks:
        x1, y1, x2, y2 = t["bbox"]
        tid     = t["track_id"]
        info    = track_store.snapshot(tid)
        primary = t.get("primary", False)

        if primary:
            colour_box = (0, 220, 0)
            label      = f"#{tid} | {info['brand']} | {info['colour']}"
            thickness  = 3
        else:
            colour_box = (120, 120, 120)
            label      = f"#{tid}"
            thickness  = 1

        cv2.rectangle(img, (x1, y1), (x2, y2), colour_box, thickness)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x1, max(y1 - th - 6, 0)), (x1 + tw + 4, y1), colour_box, -1)
        cv2.putText(img, label, (x1 + 2, max(y1 - 4, th)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    # Plate bounding boxes
    for p in plate_results:
        px1, py1, px2, py2 = p["bbox"]
        cv2.rectangle(img, (px1, py1), (px2, py2), (255, 140, 0), 2)
        cv2.putText(img, p["plate"], (px1, max(py1 - 6, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 140, 0), 2)

    return img
