from typing import Any


class ParkingGeometryEngine:
    """Maps detected vehicles to parking spaces."""

    def __init__(
        self,
        overlap_threshold: float = 0.50,
    ) -> None:
        self.overlap_threshold = overlap_threshold

    @staticmethod
    def calculate_vehicle_coverage(
        vehicle_box: list[float],
        parking_box: list[float],
    ) -> float:
        vx1, vy1, vx2, vy2 = vehicle_box
        px1, py1, px2, py2 = parking_box

        intersection_x1 = max(vx1, px1)
        intersection_y1 = max(vy1, py1)
        intersection_x2 = min(vx2, px2)
        intersection_y2 = min(vy2, py2)

        intersection_width = max(
            0.0,
            intersection_x2 - intersection_x1,
        )

        intersection_height = max(
            0.0,
            intersection_y2 - intersection_y1,
        )

        intersection_area = (
            intersection_width
            * intersection_height
        )

        vehicle_width = max(
            0.0,
            vx2 - vx1,
        )

        vehicle_height = max(
            0.0,
            vy2 - vy1,
        )

        vehicle_area = (
            vehicle_width
            * vehicle_height
        )

        if vehicle_area <= 0:
            return 0.0

        return intersection_area / vehicle_area

    def is_occupied(
        self,
        parking_box: list[float],
        vehicles: list[dict[str, Any]],
    ) -> bool:
        for vehicle in vehicles:
            box = vehicle.get("box")

            if not isinstance(box, list):
                continue

            if len(box) != 4:
                continue

            coverage = self.calculate_vehicle_coverage(
                box,
                parking_box,
            )

            if coverage >= self.overlap_threshold:
                return True

        return False

    def analyze_spaces(
        self,
        spaces: list[dict[str, Any]],
        vehicles: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        results = []

        for space in spaces:
            x, y, width, height = space["bbox"]

            parking_box = [
                x,
                y,
                x + width,
                y + height,
            ]

            occupied = self.is_occupied(
                parking_box,
                vehicles,
            )

            result = {
                **space,
                "detected_status": (
                    "occupied"
                    if occupied
                    else "empty"
                ),
                "occupied": occupied,
            }

            results.append(result)

        return results
