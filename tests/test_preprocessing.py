"""Tests for the preprocessing stage."""
import numpy as np
import pytest

from src.preprocessing.image_utils import (
    resize_max_dim, to_gray, order_points,
)
from src.preprocessing.boundary_detector import BoundaryDetector
from src.preprocessing.perspective_correction import PerspectiveCorrector


class TestImageUtils:
    def test_resize_max_dim_downscales(self):
        img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        out, scale = resize_max_dim(img, 1000)
        assert max(out.shape[:2]) == 1000
        assert 0.49 < scale < 0.51

    def test_resize_max_dim_noop(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        out, scale = resize_max_dim(img, 1000)
        assert scale == 1.0
        assert out.shape == img.shape

    def test_to_gray_from_bgr(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        gray = to_gray(img)
        assert gray.ndim == 2

    def test_to_gray_already_gray(self):
        img = np.zeros((10, 10), dtype=np.uint8)
        assert to_gray(img).shape == img.shape

    def test_order_points_returns_TL_TR_BR_BL(self):
        # Unordered input
        pts = np.array([[100, 100], [200, 100], [200, 200], [100, 200]])
        # Shuffle
        pts = pts[[2, 0, 3, 1]]
        ordered = order_points(pts)
        # TL should have smallest sum
        assert tuple(ordered[0]) == (100.0, 100.0)
        assert tuple(ordered[2]) == (200.0, 200.0)


class TestBoundaryDetector:
    def test_detects_bright_rectangle(self, default_config, dummy_note_image):
        detector = BoundaryDetector(default_config)
        corners = detector.detect(dummy_note_image)
        assert corners is not None
        assert corners.shape == (4, 2)

    def test_returns_none_for_blank(self, default_config):
        detector = BoundaryDetector(default_config)
        blank = np.zeros((500, 500, 3), dtype=np.uint8)
        corners = detector.detect(blank)
        assert corners is None


class TestPerspectiveCorrector:
    def test_warp_output_shape(self, default_config, dummy_note_image):
        corrector = PerspectiveCorrector(default_config)
        corners = np.array([[150, 100], [1050, 100], [1050, 700], [150, 700]])
        warped = corrector.correct(dummy_note_image, corners)
        expected_h = default_config["preprocessing"]["aligned_output"]["height"]
        expected_w = default_config["preprocessing"]["aligned_output"]["width"]
        assert warped.shape == (expected_h, expected_w, 3)
