from typing import Any

import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from .detector import Detector


class RTDETRDetector(Detector):
    """RT-DETR object detector used by ParkVision AI."""

    def __init__(
        self,
        model_name: str = "PekingU/rtdetr_r18vd",
        confidence_threshold: float = 0.5,
    ) -> None:
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.processor = None
        self.model = None

    def load(self) -> None:
        """Load the processor and RT-DETR model."""
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_name
        )

        self.model = AutoModelForObjectDetection.from_pretrained(
            self.model_name
        ).to(self.device)

        self.model.eval()

    def detect(self, frame: Any) -> list[dict[str, Any]]:
        """Run object detection on a single image frame."""

        if self.processor is None or self.model is None:
            raise RuntimeError(
                "RT-DETR model is not loaded. Call load() first."
            )

        inputs = self.processor(
            images=frame,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor(
            [frame.shape[:2]],
            device=self.device,
        )

        results = self.processor.post_process_object_detection(
            outputs,
            threshold=self.confidence_threshold,
            target_sizes=target_sizes,
        )[0]

        detections = []

        for score, label, box in zip(
            results["scores"],
            results["labels"],
            results["boxes"],
        ):
            detections.append(
                {
                    "label": int(label.item()),
                    "score": float(score.item()),
                    "box": [
                        float(coordinate.item())
                        for coordinate in box
                    ],
                }
            )

        return detections