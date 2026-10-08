from pathlib import Path
import json

from backend.app.ai.parking.geometry import ParkingGeometryEngine


GEOMETRY_FILE = Path(
    "runs/pklot_geometry_sample.json"
)


def main():
    with GEOMETRY_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    image = data[0]
    spaces = image["spaces"]

    print()
    print("PARKING GEOMETRY ENGINE TEST")
    print("=" * 70)

    print(
        "Image:",
        image["image"],
    )

    print(
        "Spaces:",
        len(spaces),
    )

    print()

    # Use the first three parking spaces.
    space_1 = spaces[0]
    space_2 = spaces[1]
    space_3 = spaces[2]

    def bbox_to_xyxy(space):
        x, y, width, height = space["bbox"]

        return [
            x,
            y,
            x + width,
            y + height,
        ]

    box_1 = bbox_to_xyxy(space_1)
    box_2 = bbox_to_xyxy(space_2)
    box_3 = bbox_to_xyxy(space_3)

    # Simulated vehicles:
    #
    # Vehicle 1 is completely inside P001.
    vehicle_1 = {
        "class": "vehicle",
        "score": 0.95,
        "box": [
            box_1[0] + 5,
            box_1[1] + 5,
            box_1[2] - 5,
            box_1[3] - 5,
        ],
    }

    # Vehicle 2 is completely inside P002.
    vehicle_2 = {
        "class": "vehicle",
        "score": 0.92,
        "box": [
            box_2[0] + 5,
            box_2[1] + 5,
            box_2[2] - 5,
            box_2[3] - 5,
        ],
    }

    # Vehicle 3 is outside all three spaces.
    vehicle_3 = {
        "class": "vehicle",
        "score": 0.88,
        "box": [
            10,
            10,
            60,
            60,
        ],
    }

    vehicles = [
        vehicle_1,
        vehicle_2,
        vehicle_3,
    ]

    engine = ParkingGeometryEngine(
        overlap_threshold=0.50,
    )

    results = engine.analyze_spaces(
        spaces=spaces[:3],
        vehicles=vehicles,
    )

    print(
        "Simulated vehicles:",
        len(vehicles),
    )

    print()

    for result in results:
        print(
            f'{result["id"]}: '
            f'{result["detected_status"].upper()} '
            f'| Ground truth: '
            f'{result["status"].upper()}'
        )

    print()
    print("=" * 70)

    occupied = sum(
        1
        for result in results
        if result["occupied"]
    )

    empty = len(results) - occupied

    print(
        "Detected occupied:",
        occupied,
    )

    print(
        "Detected empty:",
        empty,
    )


if __name__ == "__main__":
    main()
