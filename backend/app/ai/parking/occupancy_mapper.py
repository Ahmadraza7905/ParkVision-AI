from typing import Any


class CalibrationOccupancyMapper:
    """Map parking detections onto stable calibrated parking IDs."""

    def __init__(
        self,
        overlap_threshold: float = 0.30,
    ) -> None:
        self.overlap_threshold = overlap_threshold

    @staticmethod
    def calibration_to_box(
        bbox: list[float],
    ) -> list[float]:
        """Convert [x, y, width, height] to [x1, y1, x2, y2]."""
        if len(bbox) != 4:
            raise ValueError(
                "Calibration bbox must contain exactly 4 values."
            )

        x, y, width, height = bbox

        if width <= 0 or height <= 0:
            raise ValueError(
                "Calibration bbox width and height must be greater than zero."
            )

        return [
            x,
            y,
            x + width,
            y + height,
        ]

    @staticmethod
    def intersection_area(
        box_a: list[float],
        box_b: list[float],
    ) -> float:
        """Calculate intersection area between two xyxy boxes."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        x1 = max(ax1, bx1)
        y1 = max(ay1, by1)
        x2 = min(ax2, bx2)
        y2 = min(ay2, by2)

        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)

        return width * height

    @staticmethod
    def box_area(
        box: list[float],
    ) -> float:
        """Calculate area of an xyxy box."""
        x1, y1, x2, y2 = box

        return max(0.0, x2 - x1) * max(
            0.0,
            y2 - y1,
        )

    def overlap_ratio(
        self,
        calibration_box: list[float],
        detection_box: list[float],
    ) -> float:
        """Calculate detection coverage of a calibrated space."""
        calibration_area = self.box_area(
            calibration_box,
        )

        if calibration_area <= 0:
            return 0.0

        return self.intersection_area(
            calibration_box,
            detection_box,
        ) / calibration_area

    def map_spaces(
        self,
        calibrated_spaces: list[dict[str, Any]],
        detections: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Map each calibrated space to its best detection."""

        best_matches: list[dict[str, Any]] = []

        for space_index, space in enumerate(
            calibrated_spaces
        ):
            bbox = space.get("bbox")

            if (
                not isinstance(bbox, list)
                or len(bbox) != 4
            ):
                best_matches.append(
                    {
                        "space_index": space_index,
                        "detection_index": None,
                        "overlap": 0.0,
                    }
                )
                continue

            calibration_box = self.calibration_to_box(
                bbox,
            )

            best_detection_index: int | None = None
            best_overlap = 0.0

            for detection_index, detection in enumerate(
                detections
            ):
                detection_box = detection.get("box")

                if (
                    not isinstance(detection_box, list)
                    or len(detection_box) != 4
                ):
                    continue

                overlap = self.overlap_ratio(
                    calibration_box,
                    detection_box,
                )

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_detection_index = (
                        detection_index
                    )

            if (
                best_detection_index is not None
                and best_overlap >= self.overlap_threshold
            ):
                best_matches.append(
                    {
                        "space_index": space_index,
                        "detection_index": best_detection_index,
                        "overlap": best_overlap,
                    }
                )
            else:
                best_matches.append(
                    {
                        "space_index": space_index,
                        "detection_index": None,
                        "overlap": 0.0,
                    }
                )

        # Resolve collisions: if two calibrated spaces select
        # the same detection, keep the space with the stronger match.
        detection_winners: dict[int, dict[str, Any]] = {}

        for match in best_matches:
            detection_index = match["detection_index"]

            if detection_index is None:
                continue

            existing = detection_winners.get(
                detection_index
            )

            if (
                existing is None
                or match["overlap"] > existing["overlap"]
            ):
                detection_winners[detection_index] = match

        winning_spaces = {
            match["space_index"]
            for match in detection_winners.values()
        }

        winning_detections = {
            detection_index: match
            for detection_index, match
            in detection_winners.items()
        }

        results: list[dict[str, Any]] = []

        for space_index, space in enumerate(
            calibrated_spaces
        ):
            selected_match = None

            for detection_index, match in (
                winning_detections.items()
            ):
                if match["space_index"] == space_index:
                    selected_match = match
                    break

            if (
                selected_match is None
                or space_index not in winning_spaces
            ):
                results.append(
                    {
                        **space,
                        "status": "unknown",
                        "occupied": False,
                        "score": 0.0,
                        "overlap": 0.0,
                        "detected": False,
                    }
                )
                continue

            detection = detections[
                selected_match["detection_index"]
            ]

            occupied = (
                detection.get("class_name")
                == "space-occupied"
            )

            results.append(
                {
                    **space,
                    "status": (
                        "occupied"
                        if occupied
                        else "empty"
                    ),
                    "occupied": occupied,
                    "score": float(
                        detection.get(
                            "score",
                            0.0,
                        )
                    ),
                    "overlap": selected_match["overlap"],
                    "detected": True,
                }
            )

        return results
