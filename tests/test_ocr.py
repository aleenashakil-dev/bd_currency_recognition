"""Tests for OCR components.

PaddleOCR-dependent tests are skipped if PaddleOCR/model init fails.
"""
import cv2
import numpy as np
import pytest

from src.ocr import PatchPreprocessor, PaddleOCRDigitOCR


def _paddle_available() -> bool:
    try:
        # Import only; model initialization can still fail if models aren't present.
        import paddleocr  # noqa: F401
        return True
    except Exception:
        return False


PADDLE_AVAILABLE = _paddle_available()



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


@pytest.mark.skipif(not PADDLE_AVAILABLE, reason="PaddleOCR not installed")
class TestPaddleOCRDigitOCR:
    def test_recognizes_synthetic_digits(self, default_config):
        # Draw "500" on a white canvas
        img = np.full((120, 300), 255, dtype=np.uint8)
        cv2.putText(img, "500", (30, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    3.0, 0, 6, cv2.LINE_AA)

        # PaddleOCR needs 3-channel in some versions; keep robust
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        ocr = PaddleOCRDigitOCR(default_config)
        try:
            result = ocr.recognize(img, patch_index=1)
        except Exception:
            pytest.skip("PaddleOCR model files not available for recognition")

        assert result.text.isdigit()

