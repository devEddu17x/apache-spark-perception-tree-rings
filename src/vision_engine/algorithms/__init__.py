"""
Vision Engine Algorithms Package

This package contains concrete implementations of vision processing algorithms.
All algorithms inherit from BaseVisionAlgorithm and implement the process() method.

Current Algorithms:
- RingDetection: Simulates tree ring detection with CNN (mock)
- UnsharpMasking: Real OpenCV filter for image sharpening

To add new algorithms:
1. Create a new file in this directory (e.g., defect_detection.py)
2. Create a class that inherits from BaseVisionAlgorithm
3. Load heavy models in __init__ (runs once per partition)
4. Implement process(image_numpy, metadata) -> dict
5. Optionally return '_visual_output' key with processed image
6. Import and export the class in this __init__.py file

Example:
    from .defect_detection import DefectDetection
"""

from .ring_detection import RingDetection
from .unsharp_masking import UnsharpMasking

__all__ = ['RingDetection', 'UnsharpMasking']
