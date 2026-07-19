"""High-level entry point for CamScanner-style note detection & alignment."""
from pathlib import Path
import cv2
import numpy as np

from ..utils import get_logger, save_debug_image, draw_contour
from .image_utils import load_image, resize_max_dim
from .boundary_detector import BoundaryDetector
from .perspective_correction import PerspectiveCorrector


class NoteDetector:
    """Full CamScanner-style pipeline: detect note → warp to aligned rectangle."""

    def __init__(self, config: dict):
        self.config = config
        self.log = get_logger(__name__, config.get("log_level", "INFO"))
        self.resize_max = config["preprocessing"]["resize_max_dim"]
        self.boundary = BoundaryDetector(config)
        self.perspective = PerspectiveCorrector(config)
        self.debug = config.get("debug", False)
        self.debug_dir = config.get("debug_output_dir", "data/processed")

    def process(self, image_or_path) -> np.ndarray | None:
        """Detect the note and return an aligned top-down view.

        Args:
            image_or_path: Either a file path (str/Path) or an already-loaded
                           BGR ndarray.

        Returns:
            Aligned note image, or None if no note could be detected.
        """
        if isinstance(image_or_path, (str, Path)):
            image = load_image(str(image_or_path))
        else:
            image = image_or_path

        resized, scale = resize_max_dim(image, self.resize_max)
        self.log.debug(f"Resized image to {resized.shape} (scale={scale:.3f})")

        if self.debug:
            save_debug_image(resized, "01_resized", self.debug_dir)

        corners = self.boundary.detect(resized)
        if corners is None:
            self.log.warning("No note boundary detected.")
            return None

        if self.debug:
            vis = draw_contour(resized, corners.reshape(-1, 1, 2))
            save_debug_image(vis, "02_boundary", self.debug_dir)

        aligned = self.perspective.correct_auto_orient(resized, corners)

        if self.debug:
            save_debug_image(aligned, "03_aligned", self.debug_dir)

        return aligned
