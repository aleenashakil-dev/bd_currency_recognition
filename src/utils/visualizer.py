"""Visualization helpers for debugging the pipeline stages."""
from pathlib import Path
import cv2
import numpy as np


def save_debug_image(image: np.ndarray, stage: str, output_dir: str | Path) -> Path:
    """Save an intermediate image to disk for debugging.

    Args:
        image: Image (BGR or grayscale) to save.
        stage: Short label for this stage, e.g. "01_edges".
        output_dir: Directory to save to (created if missing).

    Returns:
        Path where the image was written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{stage}.png"
    cv2.imwrite(str(out), image)
    return out


def draw_contour(image: np.ndarray, contour: np.ndarray, color=(0, 255, 0), thickness: int = 3) -> np.ndarray:
    """Return a copy of `image` with the contour drawn on it."""
    vis = image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis, [contour], -1, color, thickness)
    return vis


def draw_grid(image: np.ndarray, rows: int = 3, cols: int = 3,
              color=(0, 255, 255), thickness: int = 2) -> np.ndarray:
    """Overlay an N×M grid on the image and label each cell (1-indexed)."""
    vis = image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    h, w = vis.shape[:2]
    cell_h, cell_w = h // rows, w // cols

    for r in range(1, rows):
        cv2.line(vis, (0, r * cell_h), (w, r * cell_h), color, thickness)
    for c in range(1, cols):
        cv2.line(vis, (c * cell_w, 0), (c * cell_w, h), color, thickness)

    n = 1
    for r in range(rows):
        for c in range(cols):
            cx, cy = c * cell_w + 20, r * cell_h + 40
            cv2.putText(vis, str(n), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, color, 2, cv2.LINE_AA)
            n += 1
    return vis
