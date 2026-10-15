"""
VisionLens AI - Production Backend Server (FastAPI)
---------------------------------------------------
Provides real-time REST APIs for deep learning image classification,
explainable AI (Grad-CAM), model switching, camera capture, and batch inference.
"""

import os
import io
import base64
import requests
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from model_engine import (
    classify_and_explain,
    MODEL_INFO,
    get_model
)

app = FastAPI(
    title="VisionLens AI Studio",
    description="Real-world Deep Learning Image Classification & Explainable AI Studio",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files mount
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "samples"), exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "css"), exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "js"), exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>VisionLens AI Studio Server Running. Frontend loading...</h1>")

@app.get("/api/health")
async def health_check():
    import torch
    return {
        "status": "online",
        "torch_version": torch.__version__,
        "device": "CUDA (GPU)" if torch.cuda.is_available() else "CPU",
        "models_available": list(MODEL_INFO.keys())
    }

@app.get("/api/models")
async def get_models():
    """Returns available vision models and their architecture specs."""
    models_list = []
    for model_id, info in MODEL_INFO.items():
        models_list.append({
            "id": model_id,
            "name": info["name"],
            "description": info["description"],
            "params": info["params"],
            "input_size": info["input_size"],
            "speed": info["speed"],
            "accuracy": info["accuracy"]
        })
    return {"models": models_list}

@app.get("/api/samples")
async def list_sample_images():
    """Lists available sample images for instant demo testing."""
    samples_dir = os.path.join(STATIC_DIR, "samples")
    files = []
    if os.path.exists(samples_dir):
        for f in os.listdir(samples_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                files.append({
                    "filename": f,
                    "url": f"/static/samples/{f}",
                    "title": f.replace("_", " ").split(".")[0].title()
                })
    return {"samples": files}

@app.post("/api/classify")
async def classify_image_endpoint(
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    image_base64: Optional[str] = Form(None),
    model_id: str = Form("mobilenet_v2"),
    generate_heatmap: bool = Form(True)
):
    """
    Classifies an image from file upload, direct URL, or base64 stream.
    """
    pil_img = None
    try:
        if file is not None:
            contents = await file.read()
            pil_img = Image.open(io.BytesIO(contents))
        elif image_url:
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(image_url, headers=headers, timeout=10)
            if resp.status_code != 200:
                raise HTTPException(status_code=400, detail="Could not download image from provided URL")
            pil_img = Image.open(io.BytesIO(resp.content))
        elif image_base64:
            if "," in image_base64:
                image_base64 = image_base64.split(",", 1)[1]
            raw_bytes = base64.b64decode(image_base64)
            pil_img = Image.open(io.BytesIO(raw_bytes))
        else:
            raise HTTPException(status_code=400, detail="No image provided (upload a file, URL, or webcam capture).")

        # Run model inference and Grad-CAM
        results = classify_and_explain(
            pil_img=pil_img,
            model_id=model_id,
            top_k=5,
            generate_heatmap=generate_heatmap
        )

        # Convert original image to thumbnail base64 for history / response
        thumb_buf = io.BytesIO()
        pil_img.convert("RGB").save(thumb_buf, format="JPEG", quality=80)
        img_preview_b64 = base64.b64encode(thumb_buf.getvalue()).decode("utf-8")

        results["image_preview"] = f"data:image/jpeg;base64,{img_preview_b64}"
        results["image_dimensions"] = f"{pil_img.width}x{pil_img.height}"
        return results

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/batch-classify")
async def batch_classify_endpoint(
    files: List[UploadFile] = File(...),
    model_id: str = Form("mobilenet_v2")
):
    """Batch classify multiple images in a single request."""
    batch_results = []
    for file in files:
        try:
            contents = await file.read()
            pil_img = Image.open(io.BytesIO(contents))
            res = classify_and_explain(pil_img, model_id=model_id, top_k=3, generate_heatmap=False)
            batch_results.append({
                "filename": file.filename,
                "top_prediction": res["predictions"][0],
                "latency_ms": res["latency_ms"],
                "success": True
            })
        except Exception as e:
            batch_results.append({
                "filename": file.filename,
                "error": str(e),
                "success": False
            })
    return {"model_id": model_id, "results": batch_results}

if __name__ == "__main__":
    import uvicorn
    # Pre-warm default model
    print("[*] Pre-warming default vision model (MobileNetV2)...")
    get_model("mobilenet_v2")
    print("[*] Starting VisionLens AI Studio Server on http://localhost:8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
