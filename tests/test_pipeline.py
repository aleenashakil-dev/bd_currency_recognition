"""Integration test for the full pipeline (with mocked OCR)."""
from unittest.mock import patch as mock_patch
import numpy as np

from src.pipeline import CurrencyRecognitionPipeline
from src.ocr import OCRResult


class TestPipelineIntegration:
    def test_pipeline_returns_result_object(self, default_config, dummy_note_image):
        """Full end-to-end call with OCR mocked to return "100" for every patch."""
        pipe = CurrencyRecognitionPipeline(default_config)

        with mock_patch.object(pipe.ocr, "recognize",
                               return_value=OCRResult(text="100", confidence=85.0, raw={})):
            result = pipe.run(dummy_note_image)

        assert result.denomination == 100
        assert result.confidence > 0
        assert result.total_patches == len(default_config["grid"]["ocr_patches"])

        assert result.elapsed_ms > 0

    def test_pipeline_handles_missing_note(self, default_config):
        pipe = CurrencyRecognitionPipeline(default_config)
        blank = np.zeros((500, 500, 3), dtype=np.uint8)
        result = pipe.run(blank)
        assert result.denomination is None
        assert result.error is not None
