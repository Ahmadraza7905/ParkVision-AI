from pathlib import Path
from tempfile import TemporaryDirectory

from backend.app.ai.parking.calibration import ParkingCalibration


def main() -> None:
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "calibration.json"

        calibration = ParkingCalibration(
            path=str(path),
        )

        spaces = [
            {
                "id": "P001",
                "bbox": [10, 20, 30, 40],
                "status": "unknown",
            },
            {
                "id": "P002",
                "bbox": [50, 60, 30, 40],
                "status": "unknown",
            },
        ]

        calibration.save(
            spaces,
            camera_id="camera-001",
        )

        loaded = calibration.load(
            camera_id="camera-001",
        )

        assert loaded == spaces
        assert calibration.exists()

        wrong_camera = calibration.load(
            camera_id="camera-002",
        )

        assert wrong_camera == []

        print("CALIBRATION UNIT TEST: PASSED")


if __name__ == "__main__":
    main()

