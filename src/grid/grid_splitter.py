"""Split an aligned note into an N×M grid of patches."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Patch:
    """A single grid cell."""
    index: int           # 1-indexed, row-major (1..N*M)
    row: int             # 0-indexed row
    col: int             # 0-indexed col
    image: np.ndarray    # BGR image of the patch
    bbox: tuple          # (x0, y0, x1, y1) in the source image


class GridSplitter:
    """Split an image into rows × cols patches with optional padding."""

    def __init__(self, config: dict):
        g = config["grid"]
        self.rows = int(g["rows"])
        self.cols = int(g["cols"])
        self.padding_ratio = float(g.get("patch_padding_ratio", 0.0))

    def split(self, image: np.ndarray) -> list[Patch]:
        """Return a list of Patch objects, indexed 1..rows*cols (row-major)."""
        h, w = image.shape[:2]
        cell_h = h // self.rows
        cell_w = w // self.cols
        pad_x = int(cell_w * self.padding_ratio)
        pad_y = int(cell_h * self.padding_ratio)

        patches: list[Patch] = []
        idx = 1
        for r in range(self.rows):
            for c in range(self.cols):
                x0 = max(0, c * cell_w - pad_x)
                y0 = max(0, r * cell_h - pad_y)
                x1 = min(w, (c + 1) * cell_w + pad_x)
                y1 = min(h, (r + 1) * cell_h + pad_y)
                crop = image[y0:y1, x0:x1].copy()
                patches.append(Patch(
                    index=idx, row=r, col=c, image=crop, bbox=(x0, y0, x1, y1)
                ))
                idx += 1
        return patches

    def get_patches_by_index(self, image: np.ndarray, indices: list[int]) -> list[Patch]:
        """Return only the patches whose 1-indexed positions are in `indices`."""
        all_patches = self.split(image)
        wanted = set(indices)
        return [p for p in all_patches if p.index in wanted]
