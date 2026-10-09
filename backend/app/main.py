from typing import Any
import io

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator
from PIL import Image

from .ai.parking.calibration import ParkingCalibration
from .ai.parking.occupancy_mapper import CalibrationOccupancyMapper
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

calibration = ParkingCalibration()
occupancy_mapper = CalibrationOccupancyMapper()


class CalibrationSpace(BaseModel):
    id: str = Field(min_length=1, max_length=32)
    bbox: list[float] = Field(min_length=4, max_length=4)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Space ID cannot be empty.")

        return value

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value: list[float]) -> list[float]:
        x, y, width, height = value

        if x < 0 or y < 0:
            raise ValueError(
                "Bounding-box x and y coordinates cannot be negative."
            )

        if width <= 0 or height <= 0:
            raise ValueError(
                "Bounding-box width and height must be greater than zero."
            )

        return value


class CalibrationRequest(BaseModel):
    spaces: list[CalibrationSpace] = Field(
        min_length=1,
        max_length=1000,
    )

    @model_validator(mode="after")
    def validate_unique_ids(self):
        ids = [space.id for space in self.spaces]

        if len(ids) != len(set(ids)):
            raise ValueError(
                "Parking-space IDs must be unique."
            )

        return self


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


@app.get("/api/calibration/{camera_id}")
def get_calibration(camera_id: str):
    spaces = calibration.load(
        camera_id=camera_id,
    )

    return {
        "camera_id": camera_id,
        "spaces": spaces,
        "configured": bool(spaces),
    }


@app.post("/api/calibration/{camera_id}")
def save_calibration(
    camera_id: str,
    request: CalibrationRequest,
):
    calibration.save(
        spaces=[
            space.model_dump()
            for space in request.spaces
        ],
        camera_id=camera_id,
    )

    return {
        "status": "saved",
        "camera_id": camera_id,
        "spaces": [
            space.model_dump()
            for space in request.spaces
        ],
        "space_count": len(request.spaces),
    }


@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    camera_id: str = Query(
        default="default-camera",
        min_length=1,
        max_length=64,
    ),
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

        calibrated_spaces = calibration.load(
            camera_id=camera_id,
        )

        if calibrated_spaces:
            mapped_spaces = occupancy_mapper.map_spaces(
                calibrated_spaces,
                result["space_detections"],
            )

            occupied_count = sum(
                1
                for space in mapped_spaces
                if space.get("occupied") is True
            )

            detected_count = sum(
                1
                for space in mapped_spaces
                if space.get("detected") is True
            )

            empty_count = sum(
                1
                for space in mapped_spaces
                if (
                    space.get("detected") is True
                    and space.get("occupied") is False
                )
            )

            unknown_count = sum(
                1
                for space in mapped_spaces
                if space.get("detected") is not True
            )

            result.update(
                {
                    "spaces": mapped_spaces,
                    "space_detections": mapped_spaces,
                    "total_spaces": len(mapped_spaces),
                    "occupied_spaces": occupied_count,
                    "available_spaces": empty_count,
                    "detected_spaces": detected_count,
                    "unknown_spaces": unknown_count,
                    "calibrated": True,
                    "camera_id": camera_id,
                }
            )

        else:
            result.update(
                {
                    "calibrated": False,
                    "camera_id": camera_id,
                    "detected_spaces": len(
                        result["space_detections"]
                    ),
                    "unknown_spaces": 0,
                }
            )

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
