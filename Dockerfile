FROM python:3.11-slim

# OpenCV runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and bundled model weights
COPY bytetrack.yaml .
COPY cameras.json .
COPY database.py models.py pipeline.py server.py tracker.py .
COPY models/ models/
COPY static/ static/

# Export YOLO11n to ONNX for 2-3x faster CPU inference via ONNXRuntime.
# Done at build time so the .onnx file is baked into the image layer.
RUN python -c "from ultralytics import YOLO; YOLO('models/yolo11n.pt').export(format='onnx', imgsz=480, dynamic=False, simplify=False)"

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
