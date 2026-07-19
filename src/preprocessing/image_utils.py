"""Low-level image helpers used across preprocessing stages."""
import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """Load an image from disk as BGR. Raises FileNotFoundError if unreadable."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def resize_max_dim(image: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    """Resize image so its longest side equals `max_dim`, preserving aspect ratio.

    Returns:
        (resized_image, scale) where scale = new_size / original_size.
    """
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image, 1.0
    scale = max_dim / longest
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA), scale


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert BGR to grayscale if not already."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def order_points(pts: np.ndarray) -> np.ndarray:
    """Return the 4 points ordered as: top-left, top-right, bottom-right, bottom-left.

    Standard trick: sum(x+y) is smallest at TL and largest at BR;
                    diff(y-x) is smallest at TR and largest at BL.

    Args:
        pts: (4, 2) array of points.
    Returns:
        (4, 2) float32 array in TL, TR, BR, BL order.
    """
    pts = pts.reshape(4, 2).astype("float32")
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]     # top-left
    rect[2] = pts[np.argmax(s)]     # bottom-right

    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]     # top-right
    rect[3] = pts[np.argmax(d)]     # bottom-left
    return rect


def adaptive_canny_thresholds(gray: np.ndarray, sigma: float = 0.33) -> tuple[int, int]:
    """Compute Canny lower/upper thresholds from the image's median intensity."""
    v = float(np.median(gray))
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return lower, upper
