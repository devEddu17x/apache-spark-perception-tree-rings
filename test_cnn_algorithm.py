#!/usr/bin/env python3
"""
Test script for CNN Segmentation Algorithm (Standalone)

This script tests the CNNSegmentation algorithm WITHOUT Spark, allowing
for quick validation on a local machine with limited resources.

Usage:
    pip install tensorflow==2.19.0 opencv-python numpy requests
    python test_cnn_algorithm.py

Expected RAM usage: ~1.5GB (model + image)
"""

import cv2
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, 'src')

from vision_engine.algorithms.cnn_segmentation import CNNSegmentation


def test_cnn_algorithm():
    """Test CNN segmentation algorithm standalone."""
    print("=" * 70)
    print("TEST: CNN Segmentation Algorithm")
    print("=" * 70)
    
    # 1. Initialize algorithm
    print("\n1️⃣ Inicializando algoritmo...")
    print("   Esto cargará el modelo CNN (300MB)")
    
    try:
        algorithm = CNNSegmentation(
            model_path="data/models/cnn.keras",
            remote_url="https://apache-spark-perception-tree-rings.edducode.me/cnn.keras"
        )
        print("   ✅ Algoritmo inicializado correctamente")
    except Exception as e:
        print(f"   ❌ Error inicializando algoritmo: {e}")
        return False
    
    # 2. Load test image
    print("\n2️⃣ Cargando imagen de prueba...")
    test_image_url = "https://apache-spark-perception-tree-rings.edducode.me/raw/F02a.png"
    
    try:
        import requests
        response = requests.get(test_image_url, timeout=30)
        response.raise_for_status()
        
        img_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("No se pudo decodificar la imagen")
        
        print(f"   ✅ Imagen cargada: {img.shape[0]}x{img.shape[1]} píxeles")
    except Exception as e:
        print(f"   ❌ Error cargando imagen: {e}")
        return False
    
    # 3. Process image
    print("\n3️⃣ Procesando imagen con CNN...")
    print("   Esto puede tardar varios segundos...")
    
    try:
        # Coordinates from F02a (from metadata)
        coordinates = (780, 884)
        
        result = algorithm.process(img, coordinates)
        
        print("   ✅ Procesamiento completado")
    except Exception as e:
        print(f"   ❌ Error procesando imagen: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. Display results
    print("\n4️⃣ Resultados:")
    print(f"   - Parches procesados: {result['num_patches']}")
    print(f"   - Tamaño original: {result['image_size'][0]}x{result['image_size'][1]}")
    print(f"   - Tamaño procesado: {result['processed_size'][0]}x{result['processed_size'][1]}")
    print(f"   - Cobertura de máscara: {result['mask_coverage']:.2%}")
    print(f"   - Tiempo de procesamiento: {result['processing_time_seconds']:.2f}s")
    
    # 5. Save visualization
    if '_visual_output' in result:
        output_path = "test_cnn_output.png"
        try:
            cv2.imwrite(output_path, result['_visual_output'])
            print(f"\n5️⃣ Visualización guardada:")
            print(f"   ✅ {output_path}")
            print(f"   Abre este archivo para ver la máscara de segmentación")
        except Exception as e:
            print(f"   ⚠️ No se pudo guardar visualización: {e}")
    
    # Success
    print("\n" + "=" * 70)
    print("✅ TEST COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    print("\nEl algoritmo CNN está funcionando correctamente.")
    print("Puedes proceder a integrarlo con Spark.")
    
    return True


if __name__ == "__main__":
    success = test_cnn_algorithm()
    sys.exit(0 if success else 1)
