import cv2
import numpy as np
from typing import List, Tuple, Dict

from .kernels import apply_sobel


class SobelRingDetector:
    def __init__(
        self,
        min_radius: int = 20,
        edge_threshold: float = 0.50,
        min_ring_distance: int = 35,
    ):
        self.min_radius = min_radius
        self.edge_threshold = edge_threshold
        self.min_ring_distance = min_ring_distance

    def detect(
        self, image: np.ndarray, center: Tuple[int, int], draw_output: bool = True
    ) -> Dict:
        h, w = image.shape[:2]
        center_x, center_y = int(center[0]), int(center[1])

        # 1. Escala de grises
        gray = self._to_grayscale(image)

        # 2. Sobel (magnitud del gradiente)
        magnitude = apply_sobel(gray)

        # 3. Calcular max_radius (distancia al borde más cercano)
        max_radius = min(center_x, w - center_x, center_y, h - center_y) - 10

        # 4. Detectar anillos en perfil radial promedio
        radii = self._detect_rings_radial(magnitude, center_x, center_y, max_radius)

        result = {
            "ring_count": len(radii),
            "radii": [int(r) for r in radii],
            "max_radius": max_radius,
        }

        # 5. Generar imagen de salida si se solicita
        if draw_output:
            result["_visual_output"] = self._draw_results(
                image, radii, center_x, center_y
            )

        return result

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return image.astype(np.float32)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    def _detect_rings_radial(
        self, magnitude: np.ndarray, center_x: int, center_y: int, max_radius: int
    ) -> List[int]:
        num_angles = 360
        radii_range = np.arange(self.min_radius, max_radius)

        if len(radii_range) == 0:
            return []

        # Crear perfil radial promediando todos los ángulos
        profile = np.zeros(len(radii_range), dtype=np.float32)
        h, w = magnitude.shape
        angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)

        for angle in angles:
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            for i, r in enumerate(radii_range):
                x = int(center_x + r * cos_a)
                y = int(center_y + r * sin_a)
                if 0 <= x < w and 0 <= y < h:
                    profile[i] += magnitude[y, x]

        # Promediar (siempre hay 360 muestras por radio)
        profile = profile / num_angles

        # Normalizar
        if np.max(profile) > 0:
            profile = profile / np.max(profile)

        # Encontrar picos
        peaks = self._find_peaks(profile)

        # Convertir índices a radios
        return [radii_range[p] for p in peaks]

    def _find_peaks(self, profile: np.ndarray) -> List[int]:
        if len(profile) < 3:
            return []

        peaks = []

        for i in range(1, len(profile) - 1):
            # Es máximo local?
            if profile[i] > profile[i - 1] and profile[i] > profile[i + 1]:
                # Supera umbral?
                if profile[i] >= self.edge_threshold:
                    # Distancia suficiente al pico anterior?
                    if not peaks or (i - peaks[-1]) >= self.min_ring_distance:
                        peaks.append(i)

        return peaks

    def _draw_results(
        self,
        image: np.ndarray,
        radii: List[int],
        center_x: int,
        center_y: int,
    ) -> np.ndarray:
        output = image.copy()

        # Centro (punto rojo)
        cv2.circle(output, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # Radio mínimo (círculo amarillo)
        cv2.circle(output, (center_x, center_y), self.min_radius, (0, 255, 255), 1)

        # Anillos detectados (gradiente de color)
        for i, radius in enumerate(radii):
            ratio = i / max(len(radii), 1)
            color = (int(255 * ratio), int(255 * (1 - ratio)), 0)
            cv2.circle(output, (center_x, center_y), radius, color, 2)

        # Texto informativo
        cv2.putText(
            output,
            f"Sobel: {len(radii)} rings",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        return output
