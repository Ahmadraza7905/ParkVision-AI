from pathlib import Path
import json


ANNOTATION_FILE = Path(
    "datasets/pklot/valid/_annotations.coco.json"
)

OUTPUT_FILE = Path(
    "runs/pklot_geometry_sample.json"
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

    grouped = {}

    for annotation in dataset["annotations"]:
        category = categories.get(
            annotation["category_id"]
        )

        if category not in {
            "space-empty",
            "space-occupied",
        }:
            continue

        image_id = annotation["image_id"]

        grouped.setdefault(
            image_id,
            []
        ).append(annotation)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = []

    for image_id, annotations in list(
        grouped.items()
    )[:5]:

        image = images[image_id]

        spaces = []

        for index, annotation in enumerate(
            annotations,
            start=1,
        ):
            category = categories[
                annotation["category_id"]
            ]

            spaces.append(
                {
                    "id": f"P{index:03d}",
                    "status": (
                        "occupied"
                        if category == "space-occupied"
                        else "empty"
                    ),
                    "bbox": annotation["bbox"],
                    "polygon": bbox_to_polygon(
                        annotation["bbox"]
                    ),
                }
            )

        output.append(
            {
                "image_id": image_id,
                "image": image["file_name"],
                "width": image["width"],
                "height": image["height"],
                "spaces": spaces,
            }
        )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print()
    print("Created:", OUTPUT_FILE)
    print("Images:", len(output))

    if output:
        print("First image:", output[0]["image"])
        print(
            "Parking spaces:",
            len(output[0]["spaces"])
        )

        print()
        print("First 5 spaces:")

        for space in output[0]["spaces"][:5]:
            print(
                space["id"],
                space["status"],
                space["bbox"],
            )


if __name__ == "__main__":
    main()
