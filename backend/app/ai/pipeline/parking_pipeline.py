from typing import Any

from ..detectors.rtdetr import RTDETRDetector


class ParkingPipeline:
    """Converts RT-DETR parking-space detections into clean parking results."""

    def __init__(
        self,
        detector: RTDETRDetector | None = None,
    ) -> None:
        self.detector = detector or RTDETRDetector()

    def load(self) -> None:
        """Load the parking-space detector."""
        self.detector.load()

    @staticmethod
    def _iou(
        box_a: list[float],
        box_b: list[float],
    ) -> float:
        """Calculate intersection-over-union for two boxes."""

        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_width = max(0.0, inter_x2 - inter_x1)
        inter_height = max(0.0, inter_y2 - inter_y1)
        intersection = inter_width * inter_height

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

        union = area_a + area_b - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    @classmethod
    def _remove_duplicates(
        cls,
        detections: list[dict[str, Any]],
        iou_threshold: float = 0.70,
    ) -> list[dict[str, Any]]:
        """Keep only the highest-confidence detection for overlapping spaces."""

        sorted_detections = sorted(
            detections,
            key=lambda detection: float(
                detection.get("score", 0.0)
            ),
            reverse=True,
        )

        kept: list[dict[str, Any]] = []

        for detection in sorted_detections:
            box = detection.get("box")

            if not isinstance(box, list) or len(box) != 4:
                continue

            duplicate = False

            for existing in kept:
                if cls._iou(box, existing["box"]) >= iou_threshold:
                    duplicate = True
                    break

            if not duplicate:
                kept.append(detection)

        return kept

    def analyze(self, frame: Any) -> dict[str, Any]:
        """Detect and classify unique parking spaces."""

        detections = self.detector.detect(frame)

        parking_detections = [
            detection
            for detection in detections
            if detection.get("class_name")
            in {"space-empty", "space-occupied"}
            and isinstance(detection.get("box"), list)
            and len(detection["box"]) == 4
        ]

        parking_detections = self._remove_duplicates(
            parking_detections,
            iou_threshold=0.70,
        )

        # Stable numbering:
        # top-to-bottom, then left-to-right.
        def sort_key(
            detection: dict[str, Any],
        ) -> tuple[float, float]:

            x1, y1, x2, y2 = detection["box"]

            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0

            return center_y, center_x

        parking_detections.sort(key=sort_key)

        space_detections = []

        for index, detection in enumerate(
            parking_detections,
            start=1,
        ):
            occupied = (
                detection["class_name"]
                == "space-occupied"
            )

            space_detections.append(
                {
                    "id": f"P{index:03d}",
                    "status": (
                        "occupied"
                        if occupied
                        else "empty"
                    ),
                    "occupied": occupied,
                    "label": detection["class_name"],
                    "score": detection["score"],
                    "box": detection["box"],
                }
            )

        occupied_count = sum(
            1
            for space in space_detections
            if space["occupied"]
        )

        empty_count = (
            len(space_detections) - occupied_count
        )

        return {
            "space_detections": space_detections,
            "spaces": space_detections,
            "total_spaces": len(space_detections),
            "occupied_spaces": occupied_count,
            "available_spaces": empty_count,
            "parking_detections": len(space_detections),
            "occupied_detections": occupied_count,
            "empty_detections": empty_count,
        }
