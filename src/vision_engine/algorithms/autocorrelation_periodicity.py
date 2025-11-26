from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

from vision_engine.core import BaseVisionAlgorithm


class AutocorrelationPeriodicity(BaseVisionAlgorithm):
    """
    Algoritmo de conteo de anillos usando autocorrelación del perfil radial.

    Pasos:
    1. Realza el canal azul con CLAHE.
    2. Calcula un mapa de bordes (gradiente de Sobel + magnitud).
    3. Convierte el mapa de bordes a coordenadas polares.
    4. Calcula el perfil radial (promedio angular de la magnitud de bordes).
    5. Suaviza el perfil, calcula autocorrelación y busca un periodo.
    6. Si la autocorrelación falla, usa FFT como fallback.
    7. Escala el número de anillos mediante un factor (ring_scale) calibrable.
    """

    def __init__(
        self,
        num_angles: int = 360,
        smooth_sigma: float = 1.5,
        min_radius: int = 40,
        max_radius: int | None = None,
        prominence: float = 0.02,
        min_expected_rings: int = 12,
        max_expected_rings: int = 30,
        ring_scale: float = 1.4,
    ):
        """
        Args:
            num_angles: número de muestras angulares para la transformación polar.
            smooth_sigma: sigma del filtro gaussiano sobre el perfil radial.
            min_radius: radio mínimo (en píxeles) para evitar ruido cerca de la médula.
            max_radius: radio máximo a considerar. Si es None, se usa el máximo posible.
            prominence: prominencia mínima (NORMALIZADA) para detectar picos
                        en la autocorrelación.
            min_expected_rings: mínimo de anillos esperados (sin escalar).
            max_expected_rings: máximo de anillos esperados (sin escalar).
            ring_scale: factor de escala global para calibrar el conteo de anillos.
                        En tu entorno real lo ajustarás desde fuera (si lo deseas).
        """
        print("  📈 Initializing AutocorrelationPeriodicity...")
        self.num_angles = num_angles
        self.smooth_sigma = smooth_sigma
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.prominence = prominence
        self.min_expected_rings = min_expected_rings
        self.max_expected_rings = max_expected_rings
        self.ring_scale = ring_scale
        print("  ✅ AutocorrelationPeriodicity initialized")

    # ------------------------------------------------------------------ #
    # API principal
    # ------------------------------------------------------------------ #
    def process(self, image_numpy: np.ndarray, coordinates: tuple) -> dict:
        """
        Procesa una imagen para estimar el número de anillos.

        Args:
            image_numpy: imagen BGR (np.ndarray).
            coordinates: tupla (x, y) del centro del tronco/médula.

        Returns:
            dict con:
                - ring_count: número de anillos estimado.
                - radii: lista de radios de los anillos dibujados.
                - metrics: métricas internas para análisis.
                - _visual_output: imagen BGR con círculos (si se detectan anillos).
        """
        # Kafka/Spark pueden mandar floats; OpenCV exige int
        center_x = int(coordinates[0])
        center_y = int(coordinates[1])

        # 1. Mapa de bordes a partir del canal azul realzado
        edge_image = self._edge_enhanced_channel(image_numpy)

        # 2. Transformación polar del mapa de bordes
        polar_image, max_radius = self._cartesian_to_polar(edge_image, center_x, center_y)

        # 3. Perfil radial de bordes (promedio angular)
        radial_profile = np.mean(polar_image, axis=0).astype(np.float32)

        # 3.1 Rango radial útil
        radius_max = max_radius if self.max_radius is None else min(self.max_radius, max_radius)
        radius_max = min(radius_max, len(radial_profile) - 1)

        if radius_max <= self.min_radius + 5:
            return {
                "ring_count": 0,
                "radii": [],
                "metrics": {
                    "reason": "Perfil radial demasiado corto para el rango pedido",
                    "min_radius": int(self.min_radius),
                    "radius_max": int(radius_max),
                    "profile_length": int(len(radial_profile)),
                },
            }

        # Región de interés donde realmente hay anillos
        profile_roi = radial_profile[self.min_radius: radius_max]
        effective_range = len(profile_roi)

        # 4. Suavizado del perfil
        smoothed_profile = gaussian_filter1d(profile_roi, sigma=self.smooth_sigma)

        # 5. Estimar periodo (spacing)
        spacing, method_used, extra_info = self._estimate_spacing(smoothed_profile, effective_range)

        if spacing is None or spacing <= 0:
            metrics = {
                "reason": "No se encontró periodo dominante",
                "min_radius": int(self.min_radius),
                "radius_max": int(radius_max),
                "profile_length": int(len(radial_profile)),
                "effective_range": int(effective_range),
                "ring_scale": float(self.ring_scale),
            }
            metrics.update(extra_info)
            return {
                "ring_count": 0,
                "radii": [],
                "metrics": metrics,
            }

        # 6. Conteo aproximado de anillos (con factor de escala)
        raw_rings = effective_range / spacing
        scaled_rings = raw_rings * self.ring_scale
        estimated_rings = int(round(scaled_rings))

        # Radii en píxeles desde el centro
        radii = [
            int(self.min_radius + spacing * k)
            for k in range(1, estimated_rings + 1)
            if self.min_radius + spacing * k <= radius_max
        ]

        visual_output = None
        if radii:
            visual_output = self._draw_rings(image_numpy, radii, center_x, center_y)

        metrics: Dict = {
            "estimated_rings": len(radii),
            "estimated_spacing": float(spacing),
            "method_used": method_used,
            "min_radius": int(self.min_radius),
            "radius_max": int(radius_max),
            "profile_length": int(len(radial_profile)),
            "effective_range": int(effective_range),
            "raw_rings": float(raw_rings),
            "scaled_rings": float(scaled_rings),
            "ring_scale": float(self.ring_scale),
        }
        metrics.update(extra_info)

        result = {
            "ring_count": len(radii),
            "radii": radii,
            "metrics": metrics,
        }
        if visual_output is not None:
            # VisionPipeline detecta esta clave y sube la imagen a R2
            result["_visual_output"] = visual_output

        return result

    # ------------------------------------------------------------------ #
    # Métodos auxiliares
    # ------------------------------------------------------------------ #
    def _edge_enhanced_channel(self, image: np.ndarray) -> np.ndarray:
        """
        Extrae un canal realzado y calcula un mapa de bordes:
        - Canal azul + normalización + CLAHE
        - Suavizado
        - Gradiente de Sobel y magnitud de gradiente
        """
        if image.ndim == 3:
            blue = image[:, :, 0].astype(np.float32)
        else:
            blue = image.astype(np.float32)

        # Normalizar y convertir a 8 bits
        blue_u8 = cv2.normalize(blue, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Realce de contraste local
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blue_u8)

        # Suavizado suave
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

        # Gradientes
        gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)

        # Normalizar magnitud a [0, 255]
        mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return mag_norm.astype(np.float32)

    def _cartesian_to_polar(
        self, image: np.ndarray, center_x: int, center_y: int
    ) -> Tuple[np.ndarray, int]:
        """
        Convierte la imagen a coordenadas polares centradas en (center_x, center_y).
        """
        height, width = image.shape
        max_radius = int(
            np.sqrt(
                max(center_x, width - center_x) ** 2
                + max(center_y, height - center_y) ** 2
            )
        )

        polar = cv2.warpPolar(
            image,
            dsize=(max_radius, self.num_angles),
            center=(center_x, center_y),
            maxRadius=max_radius,
            flags=cv2.WARP_POLAR_LINEAR,
        )
        return polar, max_radius

    def _estimate_spacing(
        self, profile: np.ndarray, effective_range: int
    ) -> Tuple[int | None, str, Dict]:
        """
        Estima el ancho promedio de anillo (spacing) combinando:

        1. Autocorrelación + búsqueda de picos (preferida)
        2. FFT como fallback

        Se sanea el perfil para evitar NaNs que disparen warnings en la FFT.
        """
        profile = np.nan_to_num(profile, nan=0.0, posinf=0.0, neginf=0.0)
        n = len(profile)
        extra_info: Dict = {"autocorr_used": False, "fft_used": False}

        if n < 10 or np.all(profile == 0):
            return None, "none", extra_info

        profile_centered = profile - np.mean(profile)
        profile_centered = np.nan_to_num(profile_centered, nan=0.0, posinf=0.0, neginf=0.0)

        # --- 1) Autocorrelación ---
        autocorr_full = np.correlate(profile_centered, profile_centered, mode="full")
        autocorr = autocorr_full[len(autocorr_full) // 2 :]

        if autocorr[0] != 0:
            autocorr = autocorr / autocorr[0]

        lags = np.arange(len(autocorr))

        lag_min = max(1, int(effective_range / self.max_expected_rings))
        lag_max = max(lag_min + 1, int(effective_range / self.min_expected_rings))
        lag_max = min(lag_max, n - 1)

        extra_info["lag_window"] = (int(lag_min), int(lag_max))
        extra_info["autocorr_length"] = int(len(autocorr))

        spacing_autocorr: int | None = None

        if lag_min < lag_max:
            window_mask = (lags >= lag_min) & (lags <= lag_max)
            if np.any(window_mask):
                lags_window = lags[window_mask]
                autocorr_window = autocorr[window_mask]

                peaks, _ = find_peaks(autocorr_window, prominence=self.prominence)

                if len(peaks) > 0:
                    target = 0.5 * (self.min_expected_rings + self.max_expected_rings)
                    best_cost = float("inf")
                    best_lag = None
                    first_lag = int(lags_window[peaks[0]])

                    for idx in peaks:
                        lag = int(lags_window[idx])
                        rings_est = effective_range / lag
                        cost = abs(rings_est - target)

                        if rings_est < self.min_expected_rings or rings_est > self.max_expected_rings:
                            cost += 10.0

                        if cost < best_cost:
                            best_cost = cost
                            best_lag = lag

                    if best_lag is not None:
                        spacing_autocorr = best_lag
                        extra_info["autocorr_used"] = True
                        extra_info["first_peak_lag"] = first_lag
                        extra_info["best_peak_lag"] = int(best_lag)

        if spacing_autocorr is not None and spacing_autocorr > 0:
            return spacing_autocorr, "autocorr", extra_info

        # --- 2) FFT fallback ---
        fft_vals = np.fft.rfft(profile_centered)
        fft_vals = np.nan_to_num(fft_vals, nan=0.0, posinf=0.0, neginf=0.0)
        power = np.abs(fft_vals) ** 2

        k = np.arange(len(power))
        k_min = max(1, int(self.min_expected_rings))
        k_max = min(int(self.max_expected_rings), len(power) - 1)

        if k_min >= k_max:
            return None, "none", extra_info

        k_window = np.arange(k_min, k_max + 1)
        power_window = power[k_window]

        best_k = int(k_window[np.argmax(power_window)])
        spacing_fft = int(round(effective_range / best_k))

        extra_info["fft_used"] = True
        extra_info["fft_best_k"] = int(best_k)
        extra_info["fft_spacing_raw"] = float(effective_range / best_k)

        if spacing_fft <= 0:
            return None, "none", extra_info

        return spacing_fft, "fft", extra_info

    def _draw_rings(
        self, image: np.ndarray, radii: List[int], center_x: int, center_y: int
    ) -> np.ndarray:
        """
        Dibuja los anillos estimados sobre la imagen original.
        """
        output = image.copy()
        cv2.circle(output, (center_x, center_y), 5, (0, 0, 255), -1)

        for r in radii:
            cv2.circle(output, (center_x, center_y), int(r), (0, 255, 0), 2)

        return output
