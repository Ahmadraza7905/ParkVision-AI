from typing import Any


VEHICLE_CLASSES = {
    "car",
    "motorcycle",
    "bus",
    "truck",
}


def filter_vehicles(
    detections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return only detections representing vehicles."""

    return [
        detection
        for detection in detections
        if detection.get("class_name") in VEHICLE_CLASSES
    ]
