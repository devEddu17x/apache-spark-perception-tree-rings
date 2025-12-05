"""
Vision Engine Algorithms Package

This package contains concrete implementations of vision processing algorithms.
All algorithms inherit from BaseVisionAlgorithm and implement the process() method.

Current Algorithms:
- RingDetection: Simulates tree ring detection with CNN (mock)
- UnsharpMasking: Real OpenCV filter for image sharpening
- PolarRingDetection: Real polar coordinate transformation for ring detection
- SobelRingDetection: Manual Sobel edge detection for ring detection
- SecondDerivativeRingDetection: Laplacian-based ring detection using zero-crossings

To add new algorithms:
1. Create a new file in this directory (e.g., defect_detection.py)
2. Create a class that inherits from BaseVisionAlgorithm
3. Load heavy models in __init__ (runs once per partition)
4. Implement process(image_numpy, coordinates) -> dict
5. Optionally return '_visual_output' key with processed image
6. Import and export the class in this __init__.py file

Example:
    from .defect_detection import DefectDetection
"""

from .ring_detection import RingDetection
from .unsharp_masking import UnsharpMasking
from .polar_coordinates import PolarRingDetection
from .sobel import SobelRingDetection
from .autocorrelation_periodicity import AutocorrelationPeriodicity
from .second_derivative import SecondDerivativeRingDetection
from .kmeans_ring_counter import KMeansRingCounter
from .cnn_segmentation import CNNSegmentation

__all__ = [
    'RingDetection',
    'UnsharpMasking',
    'PolarRingDetection',
    'SobelRingDetection',
    'AutocorrelationPeriodicity',
    'SecondDerivativeRingDetection',
    'KMeansRingCounter',
    'CNNSegmentation'
]

