"""Tests for OCR components.

The Tesseract-dependent tests are skipped if the binary isn't installed.
"""
import shutil
import cv2
import numpy as np
import pytest

from src.ocr import PatchPreprocessor, TesseractOCR


TESSERACT_AVAILABLE = shutil.which("tesseract") is not None


class TestPatchPreprocessor:
    def test_output_is_binary_grayscale(self, default_config):
        prep = PatchPreprocessor(default_config)
        patch = np.random.randint(0, 255, (50, 100, 3), dtype=np.uint8)
        out = prep.prepare(patch)
        assert out.ndim == 2
        unique_vals = np.unique(out)
        # After binarization we should have (mostly) two levels
        assert len(unique_vals) <= 5   # some morphological artifacts allowed

    def test_upscaling_grows_image(self, default_config):
        prep = PatchPreprocessor(default_config)
        patch = np.full((30, 60, 3), 200, dtype=np.uint8)
        out = prep.prepare(patch)
        upscale = default_config["ocr"]["patch_preprocessing"]["upscale_factor"]
        assert out.shape[0] == 30 * upscale
        assert out.shape[1] == 60 * upscale


@pytest.mark.skipif(not TESSERACT_AVAILABLE, reason="Tesseract not installed")
class TestTesseractOCR:
    def test_recognizes_synthetic_digits(self, default_config):
        # Draw "500" on a white canvas
        img = np.full((80, 200), 255, dtype=np.uint8)
        cv2.putText(img, "500", (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    2.0, 0, 4, cv2.LINE_AA)
        ocr = TesseractOCR(default_config)
        result = ocr.recognize(img)
        # Tesseract may add stray chars; the whitelist should give us digits only
        assert "500" in result.text
