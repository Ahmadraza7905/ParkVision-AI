from pathlib import Path
import json


ANNOTATION_FILE = Path(
    "datasets/pklot/valid/_annotations.coco.json"
)


def bbox_to_polygon(bbox):
    x, y, width, height = bbox

    return [
        [x, y],
        [x + width, y],
        [x + width, y + height],
        [x, y + height],
    ]


def main():
    with ANNOTATION_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        dataset = json.load(file)

    categories = {
        item["id"]: item["name"]
        for item in dataset["categories"]
    }

    images = {
        image["id"]: image
        for image in dataset["images"]
    }

    print("Building parking-space geometry...")
    print()

    count = 0

    for annotation in dataset["annotations"][:10]:
        category = categories.get(
            annotation["category_id"],
            "unknown",
        )

        bbox = annotation["bbox"]

        polygon = bbox_to_polygon(bbox)

        image = images[
            annotation["image_id"]
        ]

        print(
            f'Image: {image["file_name"]}'
        )

        print(
            f"Type: {category}"
        )

        print(
            f"BBox: {bbox}"
        )

        print(
            f"Polygon: {polygon}"
        )

        print("-" * 60)

        count += 1

    print()
    print("Example spaces:", count)


if __name__ == "__main__":
    main()
