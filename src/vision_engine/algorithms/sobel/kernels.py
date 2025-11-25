import numpy as np


SOBEL_X = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

SOBEL_Y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float32)


GAUSSIAN_3x3 = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=np.float32) / 16.0


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    # Padding con replicación de bordes
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    
    # Convolución vectorizada
    output = np.zeros((h, w), dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            output += kernel[i, j] * padded[i:i+h, j:j+w]
    
    return output


def apply_sobel(image: np.ndarray) -> np.ndarray:
    smoothed = convolve2d(image, GAUSSIAN_3x3)
    gx = convolve2d(smoothed, SOBEL_X)
    gy = convolve2d(smoothed, SOBEL_Y)
    return np.sqrt(gx**2 + gy**2)
