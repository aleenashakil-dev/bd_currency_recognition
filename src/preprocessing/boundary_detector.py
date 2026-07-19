"""Detect the currency note's four-corner boundary in an input image."""
import cv2
import numpy as np

from .image_utils import to_gray, adaptive_canny_thresholds


class BoundaryDetector:
    """Find the largest quadrilateral in an image (assumed to be the note)."""

    def __init__(self, config: dict):
        pp = config["preprocessing"]
        self.gaussian_kernel = tuple(pp["gaussian_kernel"])
        self.gaussian_sigma = pp["gaussian_sigma"]
        self.canny_adaptive = pp["canny"]["adaptive"]
        self.canny_lower = pp["canny"]["lower"]
        self.canny_upper = pp["canny"]["upper"]
        self.min_area_ratio = pp["contour"]["min_area_ratio"]
        self.approx_epsilon = pp["contour"]["approx_epsilon"]

    def detect(self, image: np.ndarray) -> np.ndarray | None:
        """Return the 4 corner points of the note, or None if not found.

        Returns:
            (4, 2) int array of corner points in the input image's coordinate space,
            OR None if no suitable quadrilateral is found.
        """
        gray = to_gray(image)
        blurred = cv2.GaussianBlur(gray, self.gaussian_kernel, self.gaussian_sigma)

        if self.canny_adaptive:
            lower, upper = adaptive_canny_thresholds(blurred)
        else:
            lower, upper = self.canny_lower, self.canny_upper

        edges = cv2.Canny(blurred, lower, upper)
        # Close small gaps in the boundary
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        image_area = image.shape[0] * image.shape[1]
        min_area = image_area * self.min_area_ratio

        # Search largest → smallest for the first valid quadrilateral
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(cnt)
            if area < min_area:
                break
            peri = cv2.arcLength(cnt, closed=True)
            approx = cv2.approxPolyDP(cnt, self.approx_epsilon * peri, closed=True)
            if len(approx) == 4:
                return approx.reshape(4, 2)

        # Fallback: use minAreaRect on the largest contour
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) >= min_area:
            rect = cv2.minAreaRect(largest)
            box = cv2.boxPoints(rect)
            return box.astype(int)

        return None
