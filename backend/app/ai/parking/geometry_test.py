from pathlib import Path
import json


GEOMETRY_FILE = Path(
    "runs/pklot_geometry_sample.json"
)


def box_to_polygon(box):
    x1, y1, x2, y2 = box

    return [
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2],
    ]


def calculate_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

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

    area_a = max(
        0.0,
        ax2 - ax1,
    ) * max(
        0.0,
        ay2 - ay1,
    )

    area_b = max(
        0.0,
        bx2 - bx1,
    ) * max(
        0.0,
        by2 - by1,
    )

    union_area = (
        area_a
        + area_b
        - intersection_area
    )

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def calculate_overlap_percentage(
    vehicle_box,
    parking_box,
):
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

    vehicle_area = max(
        0.0,
        vx2 - vx1,
    ) * max(
        0.0,
        vy2 - vy1,
    )

    if vehicle_area <= 0:
        return 0.0

    return (
        intersection_area
        / vehicle_area
    )


def vehicle_belongs_to_space(
    vehicle_box,
    parking_box,
    overlap_threshold=0.50,
):
    overlap = calculate_overlap_percentage(
        vehicle_box,
        parking_box,
    )

    return overlap >= overlap_threshold


def main():
    with GEOMETRY_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    first_image = data[0]

    first_space = first_image["spaces"][0]

    parking_box = first_space["bbox"]

    px, py, width, height = parking_box

    parking_box_xyxy = [
        px,
        py,
        px + width,
        py + height,
    ]

    print()
    print("PARKING GEOMETRY TEST")
    print("=" * 60)

    print("Space:", first_space["id"])
    print("Ground-truth status:", first_space["status"])
    print("Parking box:", parking_box_xyxy)
    print()

    test_cases = {
        "vehicle_inside": [
            px + 10,
            py + 10,
            px + width - 10,
            py + height - 10,
        ],
        "vehicle_partial": [
            px + width * 0.50,
            py + height * 0.50,
            px + width + 30,
            py + height + 30,
        ],
        "vehicle_outside": [
            px + width + 50,
            py + height + 50,
            px + width + 100,
            py + height + 100,
        ],
    }

    for name, vehicle_box in test_cases.items():

        overlap = calculate_overlap_percentage(
            vehicle_box,
            parking_box_xyxy,
        )

        iou = calculate_iou(
            vehicle_box,
            parking_box_xyxy,
        )

        occupied = vehicle_belongs_to_space(
            vehicle_box,
            parking_box_xyxy,
        )

        print(name)
        print("  Vehicle:", vehicle_box)
        print(
            f"  Vehicle covered by space: "
            f"{overlap * 100:.2f}%"
        )
        print(
            f"  IoU: {iou:.4f}"
        )
        print(
            f"  Occupied: {occupied}"
        )
        print("-" * 60)


if __name__ == "__main__":
    main()
