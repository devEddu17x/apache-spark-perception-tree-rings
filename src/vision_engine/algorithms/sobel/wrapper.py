import numpy as np
from typing import Dict, Tuple

from vision_engine.core import BaseVisionAlgorithm
from .detector import SobelRingDetector


class SobelRingDetection(BaseVisionAlgorithm):
    def __init__(
        self,
        min_radius: int = 20,
        edge_threshold: float = 0.50,
        min_ring_distance: int = 35,
    ):
        self._detector = SobelRingDetector(
            min_radius=min_radius,
            edge_threshold=edge_threshold,
            min_ring_distance=min_ring_distance,
        )

    def process(self, image_numpy: np.ndarray, coordinates: Tuple[float, float]) -> Dict:
        try:
            center = (int(coordinates[0]), int(coordinates[1]))
            result = self._detector.detect(image_numpy, center, draw_output=True)

            response = {
                "ring_count": result["ring_count"],
                "radii": result["radii"],
                "method": "sobel",
                "metrics": {"max_radius": result["max_radius"]},
            }

            if "_visual_output" in result:
                response["_visual_output"] = result["_visual_output"]

            return response

        except Exception as e:
            return {
                "ring_count": 0,
                "radii": [],
                "method": "sobel",
                "error": str(e),
            }
