"""Four-point perspective warp to produce a top-down view of the note."""
import cv2
import numpy as np

from .image_utils import order_points


class PerspectiveCorrector:
    """Warp a quadrilateral region to a fixed-size rectangle."""

    def __init__(self, config: dict):
        out = config["preprocessing"]["aligned_output"]
        self.out_w = int(out["width"])
        self.out_h = int(out["height"])

    def correct(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Return a top-down warped view of the region defined by `corners`.

        Args:
            image: Original BGR image.
            corners: (4, 2) array of corner points (any order).

        Returns:
            (out_h, out_w, 3) warped image.
        """
        rect = order_points(corners)

        # Optionally: compute the destination size from the actual corner distances
        # so the output aspect ratio matches the source. We use a fixed size for
        # consistent downstream grid indexing.
        dst = np.array([
            [0, 0],
            [self.out_w - 1, 0],
            [self.out_w - 1, self.out_h - 1],
            [0, self.out_h - 1],
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (self.out_w, self.out_h))
        return warped

    def correct_auto_orient(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Warp, then rotate 90° if the source region is portrait (taller than wide).

        BDT notes are landscape; if the warped result comes out portrait we rotate.
        """
        warped = self.correct(image, corners)
        # If output is fixed-aspect, orientation is baked in — but if source was
        # portrait, the numerals may be sideways. Detect using aspect of the ORIGINAL
        # quad: measure sides after ordering.
        rect = order_points(corners)
        top_w = np.linalg.norm(rect[1] - rect[0])
        left_h = np.linalg.norm(rect[3] - rect[0])
        if left_h > top_w:  # source was portrait — rotate the warped result
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
            # Also re-warp to target size if rotation changed aspect
            warped = cv2.resize(warped, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)
        return warped
