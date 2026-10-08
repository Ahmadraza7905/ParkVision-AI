from typing import Any
import io

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .ai.pipeline.parking_pipeline import ParkingPipeline


app = FastAPI(
    title="ParkVision AI",
    version="1.0.0",
)



app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
pipeline = ParkingPipeline()
pipeline.load()


# Serve frontend
app.mount(
    "/static",
    StaticFiles(directory="backend/app/static"),
    name="static",
)


@app.get("/")
def root():
    return FileResponse(
        "backend/app/static/index.html"
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ai": "ready",
    }


@app.get("/api/status")
def status():
    return {
        "status": "online",
        "model": "RT-DETR",
        "device": str(pipeline.detector.device),
        "ready": True,
    }


@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...),
) -> dict[str, Any]:

    if not file.content_type or not file.content_type.startswith(
        "image/"
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file.",
        )

    try:
        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

        frame = np.array(image)

        result = pipeline.analyze(frame)

        return {
            "filename": file.filename,
            "width": image.width,
            "height": image.height,
            **result,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc
