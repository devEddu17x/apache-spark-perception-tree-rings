import cv2
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Tuple

from vision_engine.core import BaseVisionAlgorithm


class PolarRingDetection(BaseVisionAlgorithm):
    """
    Wood ring detection algorithm using polar coordinate transformation.
    
    This algorithm detects tree rings by:
    1. Converting the image to polar coordinates centered on the pith
    2. Computing radial intensity profiles
    3. Detecting valleys (rings) in the smoothed profile
    4. Validating rings across angular consistency
    """
    
    def __init__(
        self,
        num_angles: int = 360,
        prominence: float = 1.0,
        distance: int = 20,
        smooth_sigma: float = 2.5,
        num_sectors: int = 8,
        voting_threshold: int = 4,
        consolidation_tolerance: int = 10,
        min_radius: int = 50
    ):
        """
        Initialize the polar ring detection algorithm with sector-based detection.
        
        Args:
            num_angles: Number of angular samples for polar transformation
            prominence: Minimum prominence for peak detection (0.6 = strong rings only)
            distance: Minimum distance between peaks in pixels (15 = ring thickness filter)
            smooth_sigma: Sigma for Gaussian smoothing of radial profile
            num_sectors: Number of angular sectors to divide the polar image (default: 8)
            voting_threshold: Minimum number of sectors that must detect a ring (4/8 = 50% consensus)
            consolidation_tolerance: Tolerance in pixels for matching peaks across sectors (default: 10)
            min_radius: Minimum radius to consider (ignores pith/medulla noise, default: 50)
        """
        self.num_angles = num_angles
        self.prominence = prominence
        self.distance = distance
        self.smooth_sigma = smooth_sigma
        self.num_sectors = num_sectors
        self.voting_threshold = voting_threshold
        self.consolidation_tolerance = consolidation_tolerance
        self.min_radius = min_radius
    
    def process(self, image_numpy: np.ndarray, coordinates: tuple) -> dict:
        """
        Process an image to detect wood rings using sector-based polar transformation.
        
        Args:
            image_numpy: OpenCV image as numpy array (BGR format)
            coordinates: Tuple (x, y) representing the pith center
            
        Returns:
            dict: Results containing:
                - ring_count: Number of confirmed rings
                - radii: List of detected ring radii in pixels
                - metrics: Dictionary with intermediate metrics
                - _visual_output: Image with rings drawn for visualization
        """
        # Convert coordinates to int (OpenCV requires int, not float)
        center_x = int(coordinates[0])
        center_y = int(coordinates[1])
        
        # Step 1: Extract blue channel and convert to float32
        if len(image_numpy.shape) == 3:
            blue_channel = image_numpy[:, :, 0].astype(np.float32)
        else:
            blue_channel = image_numpy.astype(np.float32)
        
        # Normalize to uint8 range for CLAHE
        blue_channel = cv2.normalize(blue_channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Step 2: Apply CLAHE for local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blue_channel)
        
        # Convert back to float32 for processing
        enhanced = enhanced.astype(np.float32)
        
        # Step 3: Transform to polar coordinates
        polar_image = self._cartesian_to_polar(enhanced, center_x, center_y)
        
        # Step 4: Apply angular smoothing to handle cracks and knots
        polar_smoothed = self._angular_smoothing(polar_image)
        
        # Step 5: SECTOR-BASED DETECTION
        # Divide polar image into sectors and detect rings in each
        sector_detections = self._detect_rings_per_sector(polar_smoothed)
        
        # Step 6: VOTING/CONSOLIDATION
        # Consolidate detections across sectors using voting
        confirmed_radii, sector_votes = self._consolidate_detections(sector_detections)
        
        # Calculate metrics
        total_detections = sum(len(detections) for detections in sector_detections)
        
        # Create visual output
        visual_output = self._draw_rings(
            image_numpy.copy(),
            confirmed_radii, 
            center_x, 
            center_y
        )
        
        return {
            "ring_count": len(confirmed_radii),
            "radii": confirmed_radii,
            "metrics": {
                "total_sector_detections": total_detections,
                "confirmed_rings": len(confirmed_radii),
                "average_votes_per_ring": float(np.mean(sector_votes)) if len(sector_votes) > 0 else 0.0
            },
            "_visual_output": visual_output
        }
    
    def _cartesian_to_polar(
        self, 
        image: np.ndarray, 
        center_x: int, 
        center_y: int
    ) -> np.ndarray:
        """
        Convert image from Cartesian to Polar coordinates.
        
        Args:
            image: Grayscale image
            center_x: X coordinate of center (pith)
            center_y: Y coordinate of center (pith)
            
        Returns:
            Polar transformed image where rows are angles and columns are radii
        """
        height, width = image.shape
        
        # Calculate maximum radius (distance to farthest corner)
        max_radius = int(np.sqrt(
            max(center_x, width - center_x)**2 + 
            max(center_y, height - center_y)**2
        ))
        
        # Use cv2.warpPolar for efficient transformation
        # Note: cv2.WARP_POLAR_LINEAR gives linear polar transform
        polar = cv2.warpPolar(
            image,
            dsize=(max_radius, self.num_angles),
            center=(center_x, center_y),
            maxRadius=max_radius,
            flags=cv2.WARP_POLAR_LINEAR
        )
        
        return polar
    
    def _angular_smoothing(self, polar_image: np.ndarray) -> np.ndarray:
        """
        Apply smoothing along the angular axis to reduce noise from cracks and knots.
        
        This is critical for handling real-world wood samples with irregularities.
        A median filter is applied along each radius to smooth out localized defects.
        
        Args:
            polar_image: Polar transformed image (angles x radii)
            
        Returns:
            Smoothed polar image
        """
        from scipy.ndimage import median_filter
        
        # Apply median filter with a kernel that smooths along angular axis
        # kernel_size: (angular_window, radial_window)
        # We want to smooth along angles (axis 0) but preserve radial detail
        smoothed = median_filter(polar_image, size=(5, 1))
        
        return smoothed
    
    def _detect_rings_per_sector(self, polar_image: np.ndarray) -> List[List[int]]:
        """
        Detect rings independently in each angular sector of the polar image.
        
        This method divides the polar image into horizontal sectors (angular bands)
        and computes a radial profile for each sector independently. This prevents
        the destructive interference that occurs when averaging oval/irregular rings.
        
        Args:
            polar_image: Polar transformed and smoothed image (angles x radii)
            
        Returns:
            List of lists, where each inner list contains detected radii for that sector
        """
        num_angles = polar_image.shape[0]
        sector_size = num_angles // self.num_sectors
        
        sector_detections = []
        
        for sector_idx in range(self.num_sectors):
            # Define sector boundaries
            start_angle = sector_idx * sector_size
            end_angle = start_angle + sector_size if sector_idx < self.num_sectors - 1 else num_angles
            
            # Extract sector
            sector = polar_image[start_angle:end_angle, :]
            
            # Compute radial profile for this sector (average across angles in this sector)
            radial_profile = np.mean(sector, axis=0)
            
            # Smooth the profile
            smoothed = gaussian_filter1d(radial_profile, sigma=self.smooth_sigma)
            
            # Invert to find valleys as peaks
            inverted = -smoothed
            
            # Find peaks in the inverted signal
            peaks, _ = find_peaks(
                inverted,
                prominence=self.prominence,
                distance=self.distance
            )
            
            # CRITICAL FILTER: Exclude pith/medulla region (min_radius)
            # Only keep peaks that are beyond the minimum radius threshold
            filtered_peaks = [p for p in peaks if p >= self.min_radius]
            
            sector_detections.append(filtered_peaks)
        
        return sector_detections

    
    def _consolidate_detections(
        self, 
        sector_detections: List[List[int]]
    ) -> Tuple[List[int], List[int]]:
        """
        Consolidate ring detections across sectors using voting.
        
        A ring is confirmed if it appears in at least voting_threshold sectors
        at similar radii (within consolidation_tolerance pixels).
        
        Args:
            sector_detections: List of detected radii for each sector
            
        Returns:
            Tuple of (confirmed_radii, votes_per_ring)
        """
        # Flatten all detections with sector index
        all_detections = []
        for sector_idx, detections in enumerate(sector_detections):
            for radius in detections:
                all_detections.append((radius, sector_idx))
        
        if not all_detections:
            return [], []
        
        # Sort by radius
        all_detections.sort(key=lambda x: x[0])
        
        # Group detections within tolerance
        confirmed_radii = []
        votes_per_ring = []
        
        i = 0
        while i < len(all_detections):
            current_radius = all_detections[i][0]
            group = [all_detections[i]]
            
            # Find all detections within tolerance
            j = i + 1
            while j < len(all_detections):
                if all_detections[j][0] - current_radius <= self.consolidation_tolerance:
                    group.append(all_detections[j])
                    j += 1
                else:
                    break
            
            # Count unique sectors in this group
            unique_sectors = len(set(detection[1] for detection in group))
            
            # Confirm if enough sectors voted for this ring
            if unique_sectors >= self.voting_threshold:
                # Use median radius from the group
                group_radii = [detection[0] for detection in group]
                median_radius = int(np.median(group_radii))
                confirmed_radii.append(median_radius)
                votes_per_ring.append(unique_sectors)
            
            i = j if j > i else i + 1
        
        return confirmed_radii, votes_per_ring
    
    def _draw_rings(
        self, 
        image: np.ndarray, 
        radii: List[int], 
        center_x: int, 
        center_y: int
    ) -> np.ndarray:
        """
        Draw detected rings on the image for visualization.
        
        Args:
            image: Original image (BGR)
            radii: List of ring radii to draw
            center_x: X coordinate of center
            center_y: Y coordinate of center
            
        Returns:
            Image with rings drawn
        """
        output = image.copy()
        
        # Draw center point
        cv2.circle(output, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # Draw each ring
        for radius in radii:
            cv2.circle(output, (center_x, center_y), radius, (0, 255, 0), 2)
        
        return output