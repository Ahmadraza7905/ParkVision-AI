from pathlib import Path

import numpy as np
from PIL import Image

from backend.app.ai.pipeline.parking_pipeline import ParkingPipeline


IMAGE_PATH = Path(
    "datasets/pklot/valid/"
    "2012-09-14_15_31_27_jpg.rf."
    "02946e33a9c6f6a9ccd2d1959c79b7b8.jpg"
)


def main() -> None:
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Test image not found: {IMAGE_PATH}")

    pipeline = ParkingPipeline()
    pipeline.load()

    image = np.array(Image.open(IMAGE_PATH).convert("RGB"))
    result = pipeline.analyze(image)

    detections = result["space_detections"]

    assert len(detections) > 0, "Pipeline returned zero parking detections."

    assert len(detections) <= 105, (
        f"Unexpectedly high detection count: {len(detections)}"
    )

    for detection in detections:
        assert "id" in detection
        assert "label" in detection
        assert "score" in detection
        assert "box" in detection
        assert "occupied" in detection
        assert len(detection["box"]) == 4

    ids = [detection["id"] for detection in detections]

    assert len(ids) == len(set(ids)), "Duplicate parking IDs detected."

    assert result["total_spaces"] == len(detections)

    print("PRODUCTION PIPELINE REGRESSION TEST: PASSED")
    print(f"Detections: {len(detections)}")
    print(f"First ID: {detections[0]['id']}")
    print(f"First label: {detections[0]['label']}")


if __name__ == "__main__":
    main()
