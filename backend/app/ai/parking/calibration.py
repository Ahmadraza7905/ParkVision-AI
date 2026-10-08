from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ParkingCalibration:
    """Store and load fixed parking-space layouts for a camera."""

    def __init__(
        self,
        path: str = "runs/parking_calibration.json",
    ) -> None:
        self.path = Path(path)

    def save(
        self,
        spaces: list[dict[str, Any]],
        camera_id: str = "default-camera",
    ) -> None:
        """Save a calibrated parking layout."""

        payload = {
            "version": 1,
            "camera_id": camera_id,
            "spaces": spaces,
        }

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.path.write_text(
            json.dumps(
                payload,
                indent=2,
            ),
            encoding="utf-8",
        )

    def load(
        self,
        camera_id: str = "default-camera",
    ) -> list[dict[str, Any]]:
        """Load the calibrated parking layout."""

        if not self.path.exists():
            return []

        payload = json.loads(
            self.path.read_text(
                encoding="utf-8",
            )
        )

        if payload.get("camera_id") != camera_id:
            return []

        spaces = payload.get("spaces", [])

        if not isinstance(spaces, list):
            return []

        return spaces

    def exists(self) -> bool:
        """Return whether a calibration file exists."""
        return self.path.exists()
