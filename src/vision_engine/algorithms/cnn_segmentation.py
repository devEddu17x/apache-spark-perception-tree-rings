"""
CNN Segmentation Algorithm for Tree Ring Detection

This algorithm performs semantic segmentation on tree ring images using a 
TensorFlow CNN model. It uses a singleton pattern to share the model across
multiple Spark partitions within the same worker, optimizing memory usage.

Key Features:
- Singleton pattern: One model per worker (not per partition)
- Cache-first strategy: Load from disk, fallback to R2 download
- Patch-based processing: 800x800 patches for large images
- CPU-optimized: Configured for c5.xlarge instances
"""

import cv2
import numpy as np
import tensorflow as tf
import os
import requests
import threading
from pathlib import Path
from vision_engine.core import BaseVisionAlgorithm


# ============================================================================
# SINGLETON MODEL CACHE
# ============================================================================

# Global cache shared across all partitions in the same worker
_MODEL_CACHE = {}
_CACHE_LOCK = None


def get_cached_model(model_path, remote_url=None):
    """
    Get CNN model from cache, loading it if necessary.
    
    This implements a singleton pattern: the model is loaded ONCE per worker,
    not once per partition. Multiple partitions in the same worker will share
    the same model instance.
    
    Args:
        model_path: Local path to cache the model
        remote_url: URL to download model if not in cache
    
    Returns:
        Loaded TensorFlow model
    """
    global _MODEL_CACHE, _CACHE_LOCK
    
    # Initialize lock on first call
    if _CACHE_LOCK is None:
        _CACHE_LOCK = threading.Lock()
    
    # Thread-safe model loading
    with _CACHE_LOCK:
        if model_path not in _MODEL_CACHE:
            print(f"🧠 Cargando modelo CNN (compartido para todo el worker)...")
            print(f"   Ruta: {model_path}")
            
            # Ensure cache directory exists
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Download from R2 if not in local cache
            if not os.path.exists(model_path):
                if remote_url:
                    print(f"📥 Modelo no encontrado localmente, descargando desde R2...")
                    print(f"   URL: {remote_url}")
                    _download_model(remote_url, model_path)
                else:
                    raise FileNotFoundError(
                        f"Modelo no encontrado en {model_path} y no se proporcionó URL de descarga"
                    )
            else:
                print(f"✅ Modelo encontrado en cache local")
            
            # Configure TensorFlow for CPU
            _configure_tensorflow_cpu()
            
            # Load model with custom objects
            model = _load_model_with_custom_objects(model_path)
            
            # Cache the model
            _MODEL_CACHE[model_path] = model
            print(f"✅ Modelo cargado y cacheado en memoria")
        else:
            print(f"✅ Usando modelo desde cache compartido (singleton)")
    
    return _MODEL_CACHE[model_path]


def _download_model(url, destination):
    """Download model from R2 with progress tracking."""
    print(f"   Descargando modelo...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            
            # Progress every 50MB
            if downloaded % (50 * 1024 * 1024) == 0:
                progress = (downloaded / total_size) * 100 if total_size > 0 else 0
                print(f"   Progreso: {downloaded / (1024**2):.1f}MB / {total_size / (1024**2):.1f}MB ({progress:.1f}%)")
    
    print(f"   ✅ Descarga completada: {downloaded / (1024**2):.1f}MB")


def _configure_tensorflow_cpu():
    """Configure TensorFlow for CPU-only execution (c5.xlarge)."""
    # Force CPU (no GPU)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce logs
    
    # Optimize threads for c5.xlarge (4 cores)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)


def _load_model_with_custom_objects(model_path):
    """Load TensorFlow model with custom loss and metrics."""
    # Define custom objects used during training
    LOSS_FUNCTION = tf.keras.losses.BinaryFocalCrossentropy(
        apply_class_balancing=True,
        from_logits=False
    )
    
    IOU_METRIC = tf.keras.metrics.BinaryIoU(
        target_class_ids=[1],
        threshold=0.5,
        name="iou_anillos"
    )
    
    # Load model
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'BinaryFocalCrossentropy': LOSS_FUNCTION,
            'BinaryIoU': IOU_METRIC
        }
    )
    
    return model


# ============================================================================
# CNN SEGMENTATION ALGORITHM
# ============================================================================

class CNNSegmentation(BaseVisionAlgorithm):
    """
    Tree ring segmentation using CNN.
    
    This algorithm segments tree ring images using a U-Net style CNN trained
    on 800x800 patches. For larger images, it processes them in patches and
    assembles the final mask.
    
    The model is shared across partitions using a singleton pattern to
    minimize memory usage (300MB per worker instead of per partition).
    """
    
    def __init__(
        self,
        model_path="/opt/spark/data/models/cnn.keras",
        remote_url="https://apache-spark-perception-tree-rings.edducode.me/cnn.keras",
        patch_size=800,
        max_dimension=8000
    ):
        """
        Initialize CNN segmentation algorithm.
        
        IMPORTANT: This __init__ is called ONCE per partition, but the model
        is loaded ONCE per worker thanks to the singleton pattern.
        
        Args:
            model_path: Local path to cache the model
            remote_url: URL to download model if not cached
            patch_size: Size of patches for segmentation (800x800)
            max_dimension: Maximum image dimension (resize if larger)
        """
        self.model_path = model_path
        self.remote_url = remote_url
        self.patch_size = patch_size
        self.max_dimension = max_dimension
        
        # Get shared model (singleton pattern)
        self.model = get_cached_model(model_path, remote_url)
    
    def process(self, image_numpy: np.ndarray, coordinates: tuple) -> dict:
        """
        Segment tree rings in image.
        
        Args:
            image_numpy: Input image (BGR format)
            coordinates: Tuple (x, y) - not used for segmentation but kept for interface
        
        Returns:
            dict: Segmentation results with visual output
        """
        import time
        start_time = time.time()
        
        # Convert to RGB if needed
        if len(image_numpy.shape) == 3 and image_numpy.shape[2] == 3:
            img_rgb = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image_numpy
        
        h, w = img_rgb.shape[:2]
        original_size = (h, w)
        
        # Protect against very large images (OOM prevention)
        if h > self.max_dimension or w > self.max_dimension:
            scale = self.max_dimension / max(h, w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            img_rgb = cv2.resize(img_rgb, (new_w, new_h))
            h, w = new_h, new_w
            print(f"   ⚠️ Imagen redimensionada: {original_size[0]}x{original_size[1]} → {h}x{w}")
        
        # Calculate number of patches
        num_patches_h = int(np.ceil(h / self.patch_size))
        num_patches_w = int(np.ceil(w / self.patch_size))
        total_patches = num_patches_h * num_patches_w
        
        print(f"   🔄 Segmentando imagen {h}x{w} → {total_patches} parches de {self.patch_size}x{self.patch_size}")
        
        # Create output mask
        output_mask = np.zeros((h, w), dtype=np.float32)
        count_mask = np.zeros((h, w), dtype=np.float32)
        
        # Process each patch
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                # Calculate patch coordinates
                y_start = i * self.patch_size
                x_start = j * self.patch_size
                y_end = min(y_start + self.patch_size, h)
                x_end = min(x_start + self.patch_size, w)
                
                # Extract patch
                patch = img_rgb[y_start:y_end, x_start:x_end]
                patch_h, patch_w = patch.shape[:2]
                
                # Pad if necessary (for edge patches)
                if patch_h < self.patch_size or patch_w < self.patch_size:
                    padded = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)
                    padded[:patch_h, :patch_w] = patch
                    patch = padded
                
                # Normalize and predict
                patch_norm = patch.astype(np.float32) / 255.0
                patch_batch = np.expand_dims(patch_norm, axis=0)
                
                # CNN inference
                pred_mask = self.model.predict(patch_batch, verbose=0)[0, :, :, 0]
                
                # Remove padding
                pred_mask = pred_mask[:patch_h, :patch_w]
                
                # Accumulate in output mask
                output_mask[y_start:y_end, x_start:x_end] += pred_mask
                count_mask[y_start:y_end, x_start:x_end] += 1.0
        
        # Average predictions (handles overlaps if any)
        output_mask = output_mask / np.maximum(count_mask, 1.0)
        
        # Binarize (threshold at 0.5)
        mask_binary = (output_mask > 0.5).astype(np.uint8) * 255
        
        # Resize back to original size if needed
        if (h, w) != original_size:
            mask_binary = cv2.resize(mask_binary, (original_size[1], original_size[0]))
        
        # Save PURE mask (without dilation) for _mask_output
        mask_pure = cv2.cvtColor(mask_binary, cv2.COLOR_GRAY2BGR)
        
        # THICKEN LINES: Apply morphological dilation to make rings more visible
        # This increases line thickness from 1-2 pixels to ~3-4 pixels
        kernel_size = 3  # Adjust this to control thickness (3=thin, 5=medium, 7=thick)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_binary_dilated = cv2.dilate(mask_binary, kernel, iterations=2)
        
        # Create visualization: Draw green rings on original image
        # Convert original to BGR if needed
        if len(image_numpy.shape) == 2:
            # Grayscale to BGR
            img_visual = cv2.cvtColor(image_numpy, cv2.COLOR_GRAY2BGR)
        else:
            # Already BGR
            img_visual = image_numpy.copy()
        
        # Overlay detected rings in bright green (0, 255, 0)
        # Create green overlay where mask is white (using DILATED mask)
        green_overlay = np.zeros_like(img_visual)
        green_overlay[mask_binary_dilated > 127] = [0, 255, 0]  # BGR: Bright Green (Lime)
        
        # Blend: 70% original + 30% green overlay for visibility
        alpha = 0.7
        img_visual = cv2.addWeighted(img_visual, alpha, green_overlay, 1 - alpha, 0)
        
        # Calculate metrics
        elapsed = time.time() - start_time
        mask_coverage = float(np.sum(mask_binary > 0) / (mask_binary.shape[0] * mask_binary.shape[1]))
        
        print(f"   ✅ Segmentación completada en {elapsed:.2f}s")
        print(f"   📊 Cobertura de máscara: {mask_coverage:.2%}")
        
        return {
            "num_patches": int(total_patches),
            "image_size": [int(original_size[0]), int(original_size[1])],
            "processed_size": [int(h), int(w)],
            "mask_coverage": mask_coverage,
            "processing_time_seconds": float(elapsed),
            "_visual_output": img_visual,  # Green overlay on original image
            "_mask_output": mask_pure       # Pure binary mask (no dilation)
        }
