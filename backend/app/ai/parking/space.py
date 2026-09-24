from dataclasses import dataclass
from typing import Tuple


@dataclass
class ParkingSpace:
    """Represents a parking space in an image."""

    id: str
    polygon: Tuple[Tuple[float, float], ...]
