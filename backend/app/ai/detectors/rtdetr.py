from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
)

from .detector import Detector


class RTDETRDetector(Detector):
    """RT-DETR detector using the locally trained ParkVision model."""

    def __init__(
        self,
        model_name: str = "backend/app/ai/models/rtdetr_parking",
        confidence_threshold: float = 0.50,
    ) -> None:
        self.model_name = Path(model_name)
        self.confidence_threshold = confidence_threshold

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.processor = None
        self.model = None

    def load(self) -> None:
        """Load the locally trained processor and RT-DETR model."""

        if not self.model_name.exists():
            raise FileNotFoundError(
                f"Trained RT-DETR checkpoint not found: "
                f"{self.model_name}"
            )

        print(
            f"Loading trained RT-DETR checkpoint: "
            f"{self.model_name}"
        )

        print(f"Device: {self.device}")

        self.processor = AutoImageProcessor.from_pretrained(
            self.model_name
        )

        self.model = AutoModelForObjectDetection.from_pretrained(
            self.model_name
        ).to(self.device)

        self.model.eval()

        print("Trained RT-DETR model loaded.")

        print(
            "Classes:",
            self.model.config.id2label,
        )

    def detect(
        self,
        frame: Any,
    ) -> list[dict[str, Any]]:
        """Run trained RT-DETR inference on one image."""

        if self.processor is None or self.model is None:
            raise RuntimeError(
                "RT-DETR model is not loaded. "
                "Call load() first."
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

        height, width = frame.shape[:2]

        target_sizes = torch.tensor(
            [[height, width]],
            device=self.device,
        )

        results = (
            self.processor.post_process_object_detection(
                outputs,
                threshold=self.confidence_threshold,
                target_sizes=target_sizes,
            )[0]
        )

        detections = []

        for score, label, box in zip(
            results["scores"],
            results["labels"],
            results["boxes"],
        ):
            label_id = int(label.item())

            class_name = self.model.config.id2label.get(
                label_id,
                str(label_id),
            )

            detections.append(
                {
                    "label": label_id,
                    "class_name": class_name,
                    "score": float(score.item()),
                    "box": [
                        float(coordinate.item())
                        for coordinate in box
                    ],
                }
            )

        return detections
