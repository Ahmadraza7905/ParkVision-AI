from typing import Any

from .space import ParkingSpace


def calculate_box_overlap(
    box: list[float],
    polygon: tuple[tuple[float, float], ...],
) -> float:
    """Calculate the fraction of a detection box covered by a parking-space polygon."""

    box_x1, box_y1, box_x2, box_y2 = box

    polygon_x1 = min(point[0] for point in polygon)
    polygon_y1 = min(point[1] for point in polygon)
    polygon_x2 = max(point[0] for point in polygon)
    polygon_y2 = max(point[1] for point in polygon)

    intersection_x1 = max(box_x1, polygon_x1)
    intersection_y1 = max(box_y1, polygon_y1)
    intersection_x2 = min(box_x2, polygon_x2)
    intersection_y2 = min(box_y2, polygon_y2)

    if intersection_x2 <= intersection_x1:
        return 0.0

    if intersection_y2 <= intersection_y1:
        return 0.0

    intersection_area = (
        (intersection_x2 - intersection_x1)
        * (intersection_y2 - intersection_y1)
    )

    box_area = (box_x2 - box_x1) * (box_y2 - box_y1)

    if box_area <= 0:
        return 0.0

    return intersection_area / box_area


def is_space_occupied(
    space: ParkingSpace,
    detections: list[dict[str, Any]],
    overlap_threshold: float = 0.25,
) -> bool:
    """Determine whether a parking space is occupied."""

    for detection in detections:
        overlap = calculate_box_overlap(
            detection["box"],
            space.polygon,
        )

        if overlap >= overlap_threshold:
            return True

    return False
