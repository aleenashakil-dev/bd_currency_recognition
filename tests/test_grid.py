"""Tests for the grid splitter."""
import numpy as np

from src.grid import GridSplitter


class TestGridSplitter:
    def test_produces_nine_patches(self, default_config, dummy_aligned_note):
        splitter = GridSplitter(default_config)
        patches = splitter.split(dummy_aligned_note)
        assert len(patches) == 9

    def test_patch_indices_are_one_based_row_major(self, default_config, dummy_aligned_note):
        splitter = GridSplitter(default_config)
        patches = splitter.split(dummy_aligned_note)
        assert patches[0].index == 1
        assert patches[0].row == 0 and patches[0].col == 0
        assert patches[2].index == 3
        assert patches[2].row == 0 and patches[2].col == 2
        assert patches[-1].index == 9
        assert patches[-1].row == 2 and patches[-1].col == 2

    def test_only_selected_patches_returned(self, default_config, dummy_aligned_note):
        splitter = GridSplitter(default_config)
        wanted = [1, 3, 7, 9]
        subset = splitter.get_patches_by_index(dummy_aligned_note, wanted)
        assert [p.index for p in subset] == wanted


    def test_patch_shapes_sum_to_image_area_approx(self, default_config):
        """Patches should tile the image (approx — integer division may leave a remainder)."""
        splitter = GridSplitter(default_config)
        # Turn off padding for exact tiling
        splitter.padding_ratio = 0.0
        img = np.zeros((300, 600, 3), dtype=np.uint8)
        patches = splitter.split(img)
        # 3x3 grid on 300x600 → each patch 100x200
        for p in patches:
            assert p.image.shape[0] == 100
            assert p.image.shape[1] == 200
