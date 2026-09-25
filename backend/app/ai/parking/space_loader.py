import json
from pathlib import Path

from .space import ParkingSpace


def load_parking_spaces(
    path: str | Path,
) -> list[ParkingSpace]:
    """Load parking-space definitions from a JSON file."""

    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        data = json.load(file)

    spaces = []

    for item in data:
        polygon = tuple(
            tuple(point)
            for point in item["polygon"]
        )

        spaces.append(
            ParkingSpace(
                id=item["id"],
                polygon=polygon,
            )
        )

    return spaces
