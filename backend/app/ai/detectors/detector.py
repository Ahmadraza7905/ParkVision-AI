from abc import ABC, abstractmethod


class Detector(ABC):
    """Base interface for all ParkVision object detectors."""

    @abstractmethod
    def load(self) -> None:
        """Load the detection model."""
        raise NotImplementedError

    @abstractmethod
    def detect(self, frame):
        """Run detection on a single image frame."""
        raise NotImplementedError
