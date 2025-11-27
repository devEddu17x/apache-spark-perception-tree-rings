import cv2
import numpy as np
from typing import List, Tuple, Dict

from vision_engine.core import BaseVisionAlgorithm


class SecondDerivativeRingDetection(BaseVisionAlgorithm):
    def __init__(
        self,
        min_radius: int = 20,
        min_ring_distance: int = 35,
        derivative_threshold: float = 0.0,
    ):
        self.min_radius = min_radius
        self.min_ring_distance = min_ring_distance
        self.derivative_threshold = derivative_threshold

    def process(
        self, image_numpy: np.ndarray, coordinates: Tuple[float, float]
    ) -> Dict:
        try:
            h, w = image_numpy.shape[:2]
            center_x, center_y = int(coordinates[0]), int(coordinates[1])

            # 1. Escala de grises (único pre-procesamiento)
            if len(image_numpy.shape) == 3:
                gray = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2GRAY).astype(np.float32)
            else:
                gray = image_numpy.astype(np.float32)

            # 2. Calcular radio máximo
            max_radius = min(center_x, w - center_x, center_y, h - center_y) - 10

            # 3. Construir perfil radial de INTENSIDAD f(r)
            radii, profile = self._build_intensity_profile(
                gray, center_x, center_y, max_radius
            )

            if len(profile) < 3:
                return self._empty_result("Perfil radial muy corto")

            # 4. Calcular primera derivada f'(r)
            first_derivative = self._compute_derivative(profile)

            # 5. Calcular segunda derivada f''(r)
            second_derivative = self._compute_derivative(first_derivative)

            # 6. Detectar máximos locales en f''(r) → valles en f(r) → anillos
            ring_radii = self._detect_local_maxima(radii[1:-1], second_derivative)

            # 7. Generar visualización
            visual_output = self._draw_results(
                image_numpy, ring_radii, center_x, center_y
            )

            return {
                "ring_count": len(ring_radii),
                "radii": [int(r) for r in ring_radii],
                "method": "second_derivative",
                "metrics": {
                    "max_radius": max_radius,
                    "profile_length": len(profile),
                },
                "_visual_output": visual_output,
            }

        except Exception as e:
            return self._empty_result(str(e))

    def _empty_result(self, error: str) -> Dict:
        """Retorna resultado vacío con mensaje de error."""
        return {
            "ring_count": 0,
            "radii": [],
            "method": "second_derivative",
            "error": error,
        }

    def _build_intensity_profile(
        self,
        gray: np.ndarray,
        center_x: int,
        center_y: int,
        max_radius: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construye el perfil radial de intensidad f(r).
        Promedia la intensidad en 360 ángulos para cada radio.
        """
        num_angles = 360
        radii = np.arange(self.min_radius, max_radius)

        if len(radii) == 0:
            return np.array([]), np.array([])

        profile = np.zeros(len(radii), dtype=np.float32)
        h, w = gray.shape
        angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)

        for angle in angles:
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            for i, r in enumerate(radii):
                x = int(center_x + r * cos_a)
                y = int(center_y + r * sin_a)
                if 0 <= x < w and 0 <= y < h:
                    profile[i] += gray[y, x]

        profile = profile / num_angles
        return radii, profile

    def _compute_derivative(self, signal: np.ndarray) -> np.ndarray:
        """
        Calcula la derivada discreta de una señal 1D.
        Usa diferencias centrales: f'(i) = (f(i+1) - f(i-1)) / 2
        """
        if len(signal) < 3:
            return np.array([])

        derivative = np.zeros(len(signal) - 2, dtype=np.float32)
        for i in range(1, len(signal) - 1):
            derivative[i - 1] = (signal[i + 1] - signal[i - 1]) / 2.0

        return derivative

    def _detect_local_maxima(
        self, radii: np.ndarray, second_derivative: np.ndarray
    ) -> List[int]:
        """
        Detecta máximos locales en la segunda derivada.
        Un máximo en f''(r) corresponde a un valle en f(r) → anillo oscuro.
        """
        if len(second_derivative) < 3:
            return []

        maxima = []
        last_maximum_idx = -self.min_ring_distance

        for i in range(1, len(second_derivative) - 1):
            # Verificar si es máximo local
            is_maximum = (
                second_derivative[i] > second_derivative[i - 1]
                and second_derivative[i] > second_derivative[i + 1]
            )

            # Verificar umbral (solo máximos positivos significativos)
            above_threshold = second_derivative[i] > self.derivative_threshold

            # Verificar distancia mínima entre anillos
            sufficient_distance = (i - last_maximum_idx) >= self.min_ring_distance

            if is_maximum and above_threshold and sufficient_distance:
                maxima.append(int(radii[i]))
                last_maximum_idx = i

        return maxima

    def _draw_results(
        self,
        image: np.ndarray,
        radii: List[int],
        center_x: int,
        center_y: int,
    ) -> np.ndarray:
        """Dibuja los anillos detectados sobre la imagen."""
        output = image.copy()

        # Centro
        cv2.circle(output, (center_x, center_y), 5, (0, 0, 255), -1)

        # Radio mínimo
        cv2.circle(output, (center_x, center_y), self.min_radius, (0, 255, 255), 1)

        # Anillos detectados
        for i, radius in enumerate(radii):
            ratio = i / max(len(radii), 1)
            color = (int(255 * ratio), int(255 * (1 - ratio)), 0)
            cv2.circle(output, (center_x, center_y), radius, color, 2)

        # Texto
        cv2.putText(
            output,
            f"2nd Derivative: {len(radii)} rings",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        return output
