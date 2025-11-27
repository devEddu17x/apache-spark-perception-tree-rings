"""
K-Means Ring Counter Algorithm - VERSIÓN FINAL
Compatible con BaseVisionAlgorithm para integración con Apache Spark

CARACTERÍSTICAS:
- Hereda de BaseVisionAlgorithm para compatibilidad con Spark
- Usa coordenadas del centro desde CSV o parámetro coordinates
- Máscara conservadora para evitar fondo
- Separación mínima 0.6%
- K adaptativo hasta 55 clusters
- Peso del radio 6x sobre intensidad
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod
from vision_engine.core import BaseVisionAlgorithm


# ============================================================================
# IMPLEMENTACIÓN K-MEANS
# ============================================================================
class KMeansFinal:
    """Implementación K-Means optimizada."""
    
    def __init__(self, n_clusters=8, max_iter=150, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.labels = None
        self.centroids = None
    
    def fit(self, X):
        if len(X) == 0:
            self.labels = np.array([])
            self.centroids = np.array([])
            return self
        
        if len(X) < self.n_clusters:
            self.n_clusters = max(1, len(X) // 3)
        
        np.random.seed(self.random_state)
        
        # K-means++ initialization
        centroids = [X[np.random.randint(len(X))]]
        for _ in range(1, self.n_clusters):
            distances = np.min([np.linalg.norm(X - c, axis=1) for c in centroids], axis=0)
            probabilities = distances ** 2
            probabilities /= probabilities.sum()
            next_centroid = X[np.random.choice(len(X), p=probabilities)]
            centroids.append(next_centroid)
        
        centroids = np.array(centroids)
        
        for _ in range(self.max_iter):
            distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            
            new_centroids = np.array([
                X[labels == i].mean(axis=0) if np.sum(labels == i) > 0 
                else centroids[i] 
                for i in range(self.n_clusters)
            ])
            
            if np.allclose(centroids, new_centroids, rtol=1e-4):
                break
            
            centroids = new_centroids
        
        self.labels = labels
        self.centroids = centroids
        return self


# ============================================================================
# ALGORITMO PRINCIPAL
# ============================================================================
class KMeansRingCounter(BaseVisionAlgorithm):
    """
    Contador de anillos mediante clustering radial K-Means.
    
    Hereda de BaseVisionAlgorithm para compatibilidad con Apache Spark.
    
    Uso standalone (desarrollo):
        algorithm = KMeansRingCounter(csv_path="pith_location.csv")
        resultado = algorithm.process(imagen, coordinates=(512, 384))
    
    Uso en Spark (producción):
        from vision_engine.algorithms.kmeans_ring_counter import KMeansRingCounter
        # Se inicializa una vez por partición
        # Se llama process() para cada imagen
    """
    
    def __init__(self, csv_path=None, max_pixels=8000, 
                 separacion_min_factor=0.006, peso_radio=6.0):
        """
        Args:
            csv_path: Ruta al CSV con coordenadas (opcional)
            max_pixels: Píxeles máximos a procesar
            separacion_min_factor: 0.6% para separación mínima
            peso_radio: 6.0 (radio 6x más importante que intensidad)
        """
        self.max_pixels = max_pixels
        self.separacion_min_factor = separacion_min_factor
        self.peso_radio = peso_radio
        self.coordenadas_dict = {}
        
        # Cargar CSV si existe
        if csv_path and Path(csv_path).exists():
            self.cargar_coordenadas_csv(csv_path)
    
    def cargar_coordenadas_csv(self, csv_path):
        """
        Carga coordenadas del centro desde CSV.
        
        Formato esperado:
        codigo_imagen,x,y
        F02a,512,384
        F10b,500,400
        """
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                codigo = row['codigo_imagen']
                x = int(row['x'])
                y = int(row['y'])
                self.coordenadas_dict[codigo] = (x, y)
            print(f"✅ Coordenadas cargadas desde CSV: {len(self.coordenadas_dict)} imágenes")
        except Exception as e:
            print(f"⚠️  Error cargando CSV: {e}")
    
    def obtener_centro(self, mask, coordinates=None, imagen_nombre=None):
        """
        Obtiene el centro del tronco con prioridad:
        1. Parámetro 'coordinates' (x, y)
        2. CSV cargado usando 'imagen_nombre'
        3. Detección automática con momentos
        
        Args:
            mask: Máscara binaria
            coordinates: Tupla (x, y) opcional
            imagen_nombre: Nombre de la imagen (sin extensión)
            
        Returns:
            Tupla (cx, cy)
        """
        # Prioridad 1: Coordenadas pasadas como parámetro
        if coordinates and len(coordinates) == 2:
            return (int(coordinates[0]), int(coordinates[1]))
        
        # Prioridad 2: CSV con nombre de imagen
        if imagen_nombre and imagen_nombre in self.coordenadas_dict:
            cx, cy = self.coordenadas_dict[imagen_nombre]
            return (int(cx), int(cy))
        
        # Prioridad 3: Detección automática
        return self.detectar_centro_automatico(mask)
    
    def detectar_centro_automatico(self, mask):
        """Detecta centro usando momentos de imagen (fallback)."""
        y, x = np.where(mask > 127)
        
        if len(x) == 0:
            h, w = mask.shape
            return (w // 2, h // 2)
        
        M = cv2.moments(mask)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
        else:
            return (int(np.mean(x)), int(np.mean(y)))
    
    def extraer_features(self, img, mask, center):
        """Extrae características radiales con ponderación del radio."""
        y, x = np.where(mask > 127)
        
        if len(x) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Submuestreo si es necesario
        if len(x) > self.max_pixels:
            step = len(x) // self.max_pixels
            indices = np.arange(0, len(x), step)
            x = x[indices]
            y = y[indices]
        
        cx, cy = center
        radios = np.sqrt((x - cx)**2 + (y - cy)**2)
        intensidades = img[y, x].astype(np.float64)
        
        # Normalización
        radios_norm = radios / (np.max(radios) + 1e-6)
        intensidades_norm = intensidades / 255.0
        
        # Ponderación: radio 6x más importante
        features = np.column_stack([
            radios_norm * self.peso_radio,
            intensidades_norm
        ])
        coords = np.column_stack([y, x])
        
        return features, coords, radios
    
    def estimar_k(self, n_pixels, gt_anillos=None):
        """Estima rango de K a probar."""
        if gt_anillos and gt_anillos > 0:
            k_min = max(18, gt_anillos + 2)
            k_max = min(gt_anillos + 15, 55)
        else:
            k_min = 25
            k_max = 55
        
        return k_min, k_max
    
    def contar_anillos_clusters(self, centers, radios_raw, radio_max_valido):
        """Cuenta anillos desde centroides, filtrando por radio máximo."""
        if len(centers) == 0:
            return 0, []
        
        radio_max = np.max(radios_raw)
        radios_clusters = (centers[:, 0] / self.peso_radio) * radio_max
        
        # Filtrar círculos fuera del tronco
        radios_validos = radios_clusters[radios_clusters <= radio_max_valido]
        
        if len(radios_validos) == 0:
            return 0, []
        
        radios_ordenados = np.sort(radios_validos)
        
        # Separación mínima 0.6%
        separacion_min = radio_max * self.separacion_min_factor
        
        anillos = [radios_ordenados[0]]
        for r in radios_ordenados[1:]:
            if r - anillos[-1] >= separacion_min:
                anillos.append(r)
        
        return len(anillos), anillos
    
    def calcular_radio_maximo_tronco(self, mask, centro):
        """Calcula radio máximo real del tronco (percentil 87)."""
        y, x = np.where(mask > 127)
        if len(x) == 0:
            return 0
        
        cx, cy = centro
        distancias = np.sqrt((x - cx)**2 + (y - cy)**2)
        radio_max = np.percentile(distancias, 87)
        return radio_max
    
    def encontrar_mejor_k(self, features, radios_raw, k_min, k_max, radio_max_valido):
        """Busca el K óptimo probando múltiples valores."""
        if len(features) < k_min:
            return {
                'k': 0, 
                'n_anillos': 0, 
                'centers': np.array([]),
                'labels': np.array([]),
                'radios': []
            }
        
        mejor_score = float('inf')
        mejor_resultado = None
        
        # Probar 10 valores de K
        k_candidatos = np.linspace(k_min, k_max, 10, dtype=int)
        
        for k in k_candidatos:
            kmeans = KMeansFinal(n_clusters=k, max_iter=200)
            kmeans.fit(features)
            
            n_anillos, radios_anillos = self.contar_anillos_clusters(
                kmeans.centroids, radios_raw, radio_max_valido
            )
            
            # Score favorece 22 anillos
            score = abs(n_anillos - 22) * 3 + (k * 0.005)
            
            if score < mejor_score:
                mejor_score = score
                mejor_resultado = {
                    'k': k,
                    'n_anillos': n_anillos,
                    'centers': kmeans.centroids,
                    'labels': kmeans.labels,
                    'radios': radios_anillos
                }
        
        return mejor_resultado
    
    def mejorar_mascara(self, img_gray):
        """Crea máscara conservadora del tronco."""
        img_eq = cv2.equalizeHist(img_gray)
        img_blur = cv2.GaussianBlur(img_eq, (7, 7), 0)
        
        # Umbral manual conservador
        max_val = np.max(img_blur)
        umbral = int(max_val * 0.40)
        _, mask = cv2.threshold(img_blur, umbral, 255, cv2.THRESH_BINARY)
        
        # Contorno más grande
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            mask_clean = np.zeros_like(mask)
            cv2.drawContours(mask_clean, [largest_contour], -1, 255, -1)
            
            # Erosión agresiva
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            mask_clean = cv2.erode(mask_clean, kernel, iterations=2)
            
            return mask_clean
        
        return mask
    
    def crear_visualizacion(self, img, mask, centro, radios_anillos):
        """Dibuja anillos detectados en la imagen."""
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        cx, cy = centro
        
        # Dibujar anillos
        for i, radio in enumerate(radios_anillos):
            color_norm = i / max(len(radios_anillos), 1)
            color_bgr = (
                int(255 * (1 - color_norm)),
                int(255 * color_norm * 0.7),
                int(255 * color_norm)
            )
            cv2.circle(img_color, (int(cx), int(cy)), int(radio), color_bgr, 2)
        
        # Dibujar centro
        cv2.drawMarker(
            img_color, 
            (int(cx), int(cy)), 
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS, 
            markerSize=20, 
            thickness=3
        )
        
        # Texto
        cv2.putText(img_color, f"Anillos: {len(radios_anillos)}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return img_color
    
    def process(self, image_numpy: np.ndarray, coordinates: tuple = None, 
                imagen_nombre: str = None) -> dict:
        """
        Procesa imagen y cuenta anillos.
        
        FIRMA COMPATIBLE CON BaseVisionAlgorithm:
        - image_numpy: Imagen BGR/Gray de OpenCV (numpy array)
        - coordinates: Tupla (x, y) con centro del tronco (opcional)
        
        PARÁMETRO ADICIONAL PARA DESARROLLO:
        - imagen_nombre: Nombre para buscar en CSV (ej: "F10b")
        
        Returns:
            Dict con resultados:
                - num_anillos: Cantidad detectada
                - k_optimo: Número de clusters usado
                - centro: Coordenadas [cx, cy]
                - radios_anillos: Lista de radios
                - _visual_output: Imagen procesada (numpy array)
                - radio_max_usado: Radio máximo del tronco
        """
        try:
            # Convertir a escala de grises
            if len(image_numpy.shape) == 3:
                img_gray = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = image_numpy.copy()
            
            # Crear máscara
            mask = self.mejorar_mascara(img_gray)
            
            # Obtener centro (prioridad: coordinates > CSV > automático)
            centro = self.obtener_centro(mask, coordinates, imagen_nombre)
            
            # Calcular radio máximo del tronco
            radio_max_valido = self.calcular_radio_maximo_tronco(mask, centro)
            
            # Extraer características
            features, coords, radios_raw = self.extraer_features(img_gray, mask, centro)
            
            if len(features) < 100:
                return {
                    'num_anillos': 0,
                    'k_optimo': 0,
                    'centro': list(centro),
                    'radios_anillos': [],
                    'error': 'Muy pocos píxeles en ROI'
                }
            
            # Estimar rango de K
            k_min, k_max = self.estimar_k(len(features))
            
            # Buscar mejor K
            mejor = self.encontrar_mejor_k(features, radios_raw, k_min, k_max, radio_max_valido)
            
            if mejor is None or mejor['n_anillos'] == 0:
                return {
                    'num_anillos': 0,
                    'k_optimo': 0,
                    'centro': list(centro),
                    'radios_anillos': []
                }
            
            # Crear visualización
            img_visual = self.crear_visualizacion(
                img_gray, mask, centro, mejor['radios']
            )
            
            return {
                'num_anillos': mejor['n_anillos'],
                'k_optimo': mejor['k'],
                'centro': list(centro),
                'radios_anillos': [float(r) for r in mejor['radios']],
                '_visual_output': img_visual,  # CLAVE ESPECIAL para imagen procesada
                'radio_max_usado': float(radio_max_valido)
            }
            
        except Exception as e:
            import traceback
            return {
                'num_anillos': 0,
                'k_optimo': 0,
                'centro': [0, 0],
                'error': f"Error: {str(e)}",
                'traceback': traceback.format_exc()
            }


# ============================================================================
# EJEMPLO DE USO (para testing standalone)
# ============================================================================
if __name__ == "__main__":
    """
    Ejemplo de uso standalone para desarrollo.
    En producción, este código no se ejecuta.
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python kmeans_ring_counter.py <imagen.png> [csv_path]")
        sys.exit(1)
    
    img_path = sys.argv[1]
    csv_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Crear algoritmo
    algorithm = KMeansRingCounter(csv_path=csv_path)
    
    # Cargar imagen
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Error cargando imagen: {img_path}")
        sys.exit(1)
    
    # Procesar
    nombre = Path(img_path).stem
    resultado = algorithm.process(img, imagen_nombre=nombre)
    
    # Mostrar resultados
    print(f"\n{'='*60}")
    print(f"RESULTADO: {nombre}")
    print(f"{'='*60}")
    print(f"Anillos detectados: {resultado.get('num_anillos', 0)}")
    print(f"K óptimo: {resultado.get('k_optimo', 0)}")
    print(f"Centro: {resultado.get('centro', [0, 0])}")
    print(f"Radio máximo: {resultado.get('radio_max_usado', 0):.1f}")
    
    # Guardar visualización
    if '_visual_output' in resultado:
        output_path = f"{nombre}_result.png"
        cv2.imwrite(output_path, resultado['_visual_output'])
        print(f"✅ Visualización guardada: {output_path}")
    
    print(f"{'='*60}\n")