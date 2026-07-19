"""Prepare a raw patch for Tesseract: upscale, contrast, binarize, denoise."""
import cv2
import numpy as np

from ..preprocessing.image_utils import to_gray


class PatchPreprocessor:
    """Turn a small color patch into a clean binary image ready for Tesseract."""

    def __init__(self, config: dict):
        pp = config["ocr"]["patch_preprocessing"]
        self.upscale = int(pp["upscale_factor"])
        self.clahe = cv2.createCLAHE(
            clipLimit=float(pp["clahe_clip_limit"]),
            tileGridSize=tuple(pp["clahe_tile_grid"]),
        )
        self.threshold_method = pp["threshold_method"]
        self.invert_if_dark_bg = bool(pp["invert_if_dark_bg"])
        self.morph_kernel = tuple(pp["morph_kernel"])

    def prepare(self, patch: np.ndarray) -> np.ndarray:
        """Return a binarized grayscale image (black text on white)."""
        gray = to_gray(patch)

        # Upscale first — OCR is much more reliable at ~300+ dpi effective resolution
        if self.upscale > 1:
            gray = cv2.resize(
                gray, None, fx=self.upscale, fy=self.upscale,
                interpolation=cv2.INTER_CUBIC,
            )

        # Local contrast enhancement
        gray = self.clahe.apply(gray)

        # Mild denoise
        gray = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

        # Binarize
        if self.threshold_method == "otsu":
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
        elif self.threshold_method == "adaptive":
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, blockSize=31, C=10,
            )
        else:  # "binary"
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Ensure dark text on light background (Tesseract prefers this)
        if self.invert_if_dark_bg and self._is_dark_background(binary):
            binary = cv2.bitwise_not(binary)

        # Small morphological open to remove speckle
        if self.morph_kernel and max(self.morph_kernel) > 1:
            kernel = np.ones(self.morph_kernel, np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        return binary

    @staticmethod
    def _is_dark_background(binary: np.ndarray) -> bool:
        """Heuristic: if more than half the pixels are black, background is dark."""
        return (binary == 0).mean() > 0.5
