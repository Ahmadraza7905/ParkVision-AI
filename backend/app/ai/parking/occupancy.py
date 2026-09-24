from typing import Any

import cv2
import numpy as np

from .space import ParkingSpace


def calculate_box_overlap(
    box: list[float],
    polygon: tuple[tuple[float, float], ...],
) -> float:
    """Calculate the fraction of a detection box inside a parking polygon."""

    box_x1, box_y1, box_x2, box_y2 = box

    if box_x2 <= box_x1 or box_y2 <= box_y1:
        return 0.0

    polygon_array = np.array(
        polygon,
        dtype=np.int32,
    )

    mask = np.zeros(
        (
            int(max(box_y2, polygon_array[:, 1].max())) + 1,
            int(max(box_x2, polygon_array[:, 0].max())) + 1,
        ),
        dtype=np.uint8,
    )

    cv2.fillPoly(
        mask,
        [polygon_array],
        1,
    )

    box_mask = np.zeros_like(mask)

    cv2.rectangle(
        box_mask,
        (int(box_x1), int(box_y1)),
        (int(box_x2), int(box_y2)),
        1,
        -1,
    )

    intersection_area = np.logical_and(
        mask,
        box_mask,
    ).sum()

    box_area = np.count_nonzero(box_mask)

    if box_area == 0:
        return 0.0

    return float(intersection_area / box_area)


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
