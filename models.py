"""
models.py — ML model loading and inference
===========================================
All models are loaded once at module import time and shared across all camera
pipelines. Thread-safe for read-only inference (PyTorch inference_mode).

Models:
  - YOLO (yolo11n.pt)            : vehicle detection + ByteTrack tracking
  - EfficientNet-B0               : vehicle colour classification (15 classes)
  - CLIP (clip-vit-base-patch32)  : zero-shot car brand classification
  - fast_alpr                     : licence plate detection + OCR
"""

import os
import json
import cv2
import torch
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO
from fast_alpr import ALPR
from transformers import CLIPProcessor, CLIPModel

# ── Paths ──────────────────────────────────────────────────────────────────────
_BASE        = os.path.dirname(os.path.abspath(__file__))
YOLO_PATH    = os.path.join(_BASE, "models", "yolo11n.pt")
COLOR_PATH   = os.path.join(_BASE, "models", "color_classifier.pth")
CLASSES_PATH = os.path.join(_BASE, "models", "color_classes.json")

# ── Device ─────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[models] Using device: {DEVICE}")

# ── YOLO vehicle detector ──────────────────────────────────────────────────────
# yolo11n is the nano variant — fastest inference, ~30-50ms/frame on CPU.
# Used in .track() mode with ByteTrack to assign persistent track IDs.
print("[models] Loading YOLO detector...")
DETECTOR = YOLO(YOLO_PATH)
print("[models] YOLO loaded.")

# ── Licence plate reader (fast_alpr) ──────────────────────────────────────────
# Two-stage pipeline: yolo-v9-t detects the plate bounding box,
# cct-s-v1 OCR reads the text. Run on full frame every N frames.
print("[models] Loading ALPR...")
PLATE_READER = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="cct-s-v1-global-model",
)
print("[models] ALPR loaded.")

# ── Colour classifier (EfficientNet-B0) ───────────────────────────────────────
# Fine-tuned on merged vehicle dataset (VCoR + seebicb + dataclusterlabs).
# Training used body-patch cropping aligned with inference-time cropping.
print("[models] Loading colour classifier...")
_COLOR_MODEL   = None
_COLOR_CLASSES = []
_COLOR_TFM     = None

try:
    ckpt = torch.load(COLOR_PATH, map_location=DEVICE, weights_only=False)
    _COLOR_CLASSES = ckpt.get("classes", [])
    arch = ckpt.get("arch", "efficientnet_b0")

    if arch == "efficientnet_b2":
        _m = models.efficientnet_b2(weights=None)
    elif arch == "efficientnet_b0":
        _m = models.efficientnet_b0(weights=None)
    else:
        _m = models.mobilenet_v3_small(weights=None)

    _m.classifier[-1] = torch.nn.Linear(_m.classifier[-1].in_features, len(_COLOR_CLASSES))
    _m.load_state_dict(ckpt["model_state"])
    _m.to(DEVICE).eval()
    _COLOR_MODEL = _m

    _COLOR_TFM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    print(f"[models] Colour classifier loaded ({arch}, {len(_COLOR_CLASSES)} classes).")
except Exception as e:
    print(f"[models] WARNING: Colour classifier failed to load — {e}")

# ── Brand classifier (CLIP zero-shot) ─────────────────────────────────────────
# No brand-specific training needed. At startup, text embeddings for all brand
# prompts are pre-computed and cached. Inference: cosine similarity between
# vehicle image embedding and brand text embeddings.
CAR_BRANDS = [
    "Dacia", "Renault", "Peugeot", "Volkswagen", "Toyota", "Honda", "Ford",
    "BMW", "Mercedes-Benz", "Audi", "Hyundai", "Kia", "Nissan", "Mazda",
    "Skoda", "Seat", "Opel", "Fiat", "Citroën", "Volvo", "Mitsubishi",
    "Suzuki", "Subaru", "Jeep", "Land Rover", "Porsche", "Tesla",
]
_BRAND_PROMPTS = [f"a photo of a {b} car" for b in CAR_BRANDS]

_CLIP_OK        = False
_clip_model     = None
_clip_processor = None
_text_feats     = None

print("[models] Loading CLIP brand classifier...")
try:
    _clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    _text_inputs    = _clip_processor(text=_BRAND_PROMPTS, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        text_out    = _clip_model.text_model(**_text_inputs)
        _tf         = text_out.pooler_output @ _clip_model.text_projection.weight.T
        _text_feats = torch.nn.functional.normalize(_tf, p=2, dim=-1).detach()
    _CLIP_OK = True
    print("[models] CLIP loaded.")
except Exception as e:
    print(f"[models] WARNING: CLIP failed to load — {e}")


# ── Public inference functions ─────────────────────────────────────────────────

def detect_colour(crop_bgr: "np.ndarray") -> tuple[str, float]:
    """
    Classify vehicle colour from a BGR crop.

    Uses body-region patch voting:
      1. Crop to body region (top 15-70%, inner 10-90%) — removes roof glare and wheels.
      2. Split into 3 horizontal patches (left / centre / right).
      3. Classify each patch independently.
      4. Return majority-vote colour and its average confidence.

    Returns: (colour_label, confidence)  e.g. ("White", 0.91)
    """
    try:
        if _COLOR_MODEL is None:
            return "Unknown", 0.0

        h, w = crop_bgr.shape[:2]
        body = crop_bgr[int(h * 0.15):int(h * 0.70),
                        int(w * 0.10):int(w * 0.90)]
        bh, bw = body.shape[:2]
        if bh < 10 or bw < 10:
            return "Unknown", 0.0

        patches = [
            body[:, :bw // 3],
            body[:, bw // 3: 2 * bw // 3],
            body[:, 2 * bw // 3:],
        ]

        votes = []
        for patch in patches:
            if patch.size == 0:
                continue
            pil = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
            with torch.inference_mode():
                logits = _COLOR_MODEL(_COLOR_TFM(pil).unsqueeze(0).to(DEVICE))
            probs = torch.softmax(logits, dim=1)[0]
            idx   = probs.argmax().item()
            votes.append((_COLOR_CLASSES[idx], probs[idx].item()))

        if not votes:
            return "Unknown", 0.0

        from collections import Counter
        counts   = Counter(v[0] for v in votes)
        best_lbl = counts.most_common(1)[0][0]
        avg_conf = sum(v[1] for v in votes if v[0] == best_lbl) / counts[best_lbl]
        return best_lbl.capitalize(), avg_conf

    except Exception:
        return "Unknown", 0.0


def detect_brand(crop_bgr: "np.ndarray") -> tuple[str, float]:
    """
    Classify car brand using CLIP zero-shot similarity.

    Encodes the vehicle crop as an image embedding, computes cosine similarity
    against pre-cached text embeddings for all brand prompts, returns top match.

    Returns: (brand_name, confidence)  e.g. ("Toyota", 0.61)
    """
    if not _CLIP_OK or _clip_model is None:
        return "—", 0.0
    try:
        pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        inp = _clip_processor(images=pil, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            img_out   = _clip_model.vision_model(**inp)
            img_feats = img_out.pooler_output @ _clip_model.visual_projection.weight.T
            img_feats = torch.nn.functional.normalize(img_feats, p=2, dim=-1)
            sims      = (img_feats @ _text_feats.T).squeeze(0)
        idx        = int(sims.argmax())
        brand      = CAR_BRANDS[idx]
        brand_conf = float(torch.softmax(sims * 100, dim=0)[idx])
        return brand, brand_conf
    except Exception:
        return "—", 0.0


def read_plates(frame_rgb: "np.ndarray") -> list[dict]:
    """
    Run the ALPR pipeline on a full RGB frame.

    Returns a list of dicts: [{"plate": str, "confidence": float, "bbox": (x1,y1,x2,y2)}]
    Plates shorter than 3 characters or made of all-identical characters are filtered out
    (common OCR artefacts like "000" or "---").
    """
    results = []
    try:
        detections = PLATE_READER.predict(frame_rgb)
        for r in (detections or []):
            conf = r.ocr.confidence if r.ocr else 0.0
            if isinstance(conf, list):
                conf = conf[0] if conf else 0.0
            bb   = r.detection.bounding_box if r.detection else None
            bbox = (int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2)) if bb else None
            text = r.ocr.text.upper().strip() if r.ocr and r.ocr.text else None
            if text and bbox and len(text) >= 3 and not all(c == text[0] for c in text):
                results.append({"plate": text, "confidence": float(conf), "bbox": bbox})
    except Exception:
        pass
    return results
