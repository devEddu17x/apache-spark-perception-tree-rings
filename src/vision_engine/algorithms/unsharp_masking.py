"""
Unsharp Masking Algorithm

Real OpenCV implementation for image sharpening.
This algorithm enhances image details by applying Gaussian blur and weighted addition.

Useful for preprocessing images before analysis or improving visual quality of results.
"""

import cv2
import numpy as np

from vision_engine.core import BaseVisionAlgorithm


class UnsharpMasking(BaseVisionAlgorithm):
    """
    Unsharp masking algorithm for image sharpening.
    
    This is a real OpenCV implementation that enhances image details
    by applying Gaussian blur and weighted addition.
    
    Useful for preprocessing images before analysis or improving
    visual quality of results.
    """
    
    def __init__(self):
        """
        Initialize the unsharp masking algorithm.
        
        No heavy resources needed for this algorithm, but __init__ is still
        called once per partition.
        """
        print("  🔧 Initializing Unsharp Masking filter...")
        self.kernel_size = (5, 5)
        self.sigma = 1.0
        self.amount = 1.5  # Sharpening strength
        self.threshold = 0  # Minimum contrast threshold
        print("  ✅ Unsharp Masking initialized")
    
    def process(self, image_numpy: np.ndarray, metadata: dict) -> dict:
        """
        Apply unsharp masking to enhance image sharpness.
        
        Args:
            image_numpy: Input image as numpy array
            metadata: Original metadata (not used in this algorithm)
        
        Returns:
            dict: Processing status and sharpened image
        """
        try:
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(image_numpy, self.kernel_size, self.sigma)
            
            # Calculate the sharpened image
            # sharpened = original + amount * (original - blurred)
            sharpened = cv2.addWeighted(
                image_numpy, 
                1.0 + self.amount, 
                blurred, 
                -self.amount, 
                0
            )
            
            # Add text overlay to show it's been processed
            output_image = sharpened.copy()
            cv2.putText(
                output_image,
                "Unsharp Mask Applied",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),  # Cyan
                2
            )
            
            return {
                "status": "success",
                "parameters": {
                    "kernel_size": self.kernel_size,
                    "sigma": self.sigma,
                    "amount": self.amount
                },
                "_visual_output": output_image  # Will be uploaded to R2
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
