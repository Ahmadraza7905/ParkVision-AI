from backend.app.ai.parking.occupancy_mapper import (
    CalibrationOccupancyMapper,
)


def main():
    mapper = CalibrationOccupancyMapper(
        overlap_threshold=0.30,
    )

    calibrated_spaces = [
        {
            "id": "P001",
            "bbox": [10, 10, 100, 100],
        },
        {
            "id": "P002",
            "bbox": [130, 10, 100, 100],
        },
    ]

    detections = [
        {
            "class_name": "space-occupied",
            "score": 0.95,
            "box": [15, 15, 105, 105],
        },
        {
            "class_name": "space-empty",
            "score": 0.91,
            "box": [135, 15, 225, 105],
        },
    ]

    results = mapper.map_spaces(
        calibrated_spaces,
        detections,
    )

    assert len(results) == 2

    assert results[0]["id"] == "P001"
    assert results[0]["status"] == "occupied"
    assert results[0]["occupied"] is True
    assert results[0]["detected"] is True

    assert results[1]["id"] == "P002"
    assert results[1]["status"] == "empty"
    assert results[1]["occupied"] is False
    assert results[1]["detected"] is True

    print("OCCUPANCY MAPPER TEST: PASSED")


if __name__ == "__main__":
    main()
