from typing import Any

from ..detectors.rtdetr import RTDETRDetector
from ..parking.occupancy import is_space_occupied
from ..parking.space import ParkingSpace
from ..parking.vehicle_filter import filter_vehicles


class ParkingPipeline:
    """Coordinates vehicle detection and parking occupancy analysis."""

    def __init__(
        self,
        detector: RTDETRDetector | None = None,
    ) -> None:
        self.detector = detector or RTDETRDetector()

    def load(self) -> None:
        """Load the AI detection model."""
        self.detector.load()

    def analyze(
        self,
        frame: Any,
        parking_spaces: list[ParkingSpace],
    ) -> dict[str, Any]:
        """Analyze a frame and return parking occupancy results."""

        detections = self.detector.detect(frame)

        vehicles = filter_vehicles(detections)

        spaces = []

        for space in parking_spaces:
            occupied = is_space_occupied(
                space,
                vehicles,
            )

            spaces.append(
                {
                    "id": space.id,
                    "occupied": occupied,
                }
            )

        occupied_count = sum(
            1
            for space in spaces
            if space["occupied"]
        )

        total_spaces = len(spaces)

        return {
            "spaces": spaces,
            "total_spaces": total_spaces,
            "occupied_spaces": occupied_count,
            "available_spaces": total_spaces - occupied_count,
            "vehicles_detected": len(vehicles),
        }
