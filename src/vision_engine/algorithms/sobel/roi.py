import cv2
import numpy as np
from typing import Tuple


def create_trunk_mask(
    image: np.ndarray,
    center_x: int,
    center_y: int,
    min_radius: int = 20
) -> Tuple[np.ndarray, int]:
    h, w = image.shape[:2]
    
    # Convertir a HSV - la saturación separa bien el tronco del fondo gris
    if len(image.shape) == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
    else:
        saturation = image.copy()
    
    # Umbralizar por saturación (tronco = saturado, fondo gris = no saturado)
    _, mask = cv2.threshold(saturation, 25, 255, cv2.THRESH_BINARY)
    
    # Si el centro no está en la máscara, intentar con Otsu en escala de grises
    if mask[center_y, center_x] == 0:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invertir si el centro sigue en negro
        if mask[center_y, center_x] == 0:
            mask = 255 - mask
    
    # Limpiar máscara con operaciones morfológicas
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    
    # Encontrar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return _fallback_circular_mask(h, w, center_x, center_y, min_radius)
    
    # Encontrar el contorno que contiene el centro
    trunk_contour = None
    for contour in contours:
        if cv2.pointPolygonTest(contour, (float(center_x), float(center_y)), False) >= 0:
            trunk_contour = contour
            break
    
    # Si ninguno contiene el centro, usar el más grande
    if trunk_contour is None:
        trunk_contour = max(contours, key=cv2.contourArea)
    
    # Crear máscara final
    final_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(final_mask, [trunk_contour], -1, 255, -1)
    
    # Calcular radio máximo (mediana de distancias al contorno)
    distances = []
    for point in trunk_contour.reshape(-1, 2):
        dist = np.sqrt((point[0] - center_x)**2 + (point[1] - center_y)**2)
        distances.append(dist)
    max_radius = int(np.median(distances)) if distances else min(h, w) // 2
    
    # Excluir el centro (médula)
    cv2.circle(final_mask, (center_x, center_y), min_radius, 0, -1)
    
    return final_mask, max_radius


def _fallback_circular_mask(
    h: int, w: int, 
    center_x: int, center_y: int,
    min_radius: int
) -> Tuple[np.ndarray, int]:
    max_radius = int(min(center_x, w - center_x, center_y, h - center_y) * 0.95)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), max_radius, 255, -1)
    cv2.circle(mask, (center_x, center_y), min_radius, 0, -1)
    return mask, max_radius
