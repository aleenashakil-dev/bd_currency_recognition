"""PaddleOCR-based digit OCR for BDT denominations.

Requirements enforced by the project:
- Offline inference only (no cloud calls).
- Initialize PaddleOCR only once per instance and reuse.
- Extract digits only.
- Provide a confidence score.

The implementation uses PaddleOCR's built-in configuration and then
post-filters OCR text to digits.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np

from ..utils import get_logger


@dataclass
class OCRResult:
    """OCR result for a single patch."""

    text: str  # digits-only string
    confidence: float  # 0–1 (Paddle OCR) or -1 if unavailable
    raw: dict[str, Any]  # full Paddle output for debugging
    patch_index: int | None = None


class PaddleOCRDigitOCR:
    """Digit-only OCR tuned for currency denomination numerals."""

    _DIGITS_RE = re.compile(r"\d+")

    def __init__(self, config: dict):
        self.config = config
        ocr_cfg = config.get("ocr", {})
        self.log = get_logger(__name__, config.get("log_level", "INFO"))

        # Confidence threshold used later in pipeline/validation.
        self.conf_threshold = float(ocr_cfg.get("confidence_threshold", 0.0))

        # PaddleOCR config
        # - offline mode: PaddleOCR does not require network for inference if models are present.
        # - use_angle_cls is optional; default off for speed.
        # - use_gpu controls device selection.
        try:
            from paddleocr import PaddleOCR
        except ModuleNotFoundError as e:  # pragma: no cover
            raise ModuleNotFoundError(
                "PaddleOCR is not installed. Install dependencies and ensure paddleocr is available."
            ) from e

        paddle_args = {
            "lang": ocr_cfg.get("language", "en"),
            "use_angle_cls": bool(ocr_cfg.get("use_angle_cls", False)),
            "use_gpu": bool(ocr_cfg.get("use_gpu", False)),
        }

        self.ocr = PaddleOCR(**paddle_args)


    def recognize(self, image: np.ndarray, patch_index: int | None = None) -> OCRResult:
        """Run OCR on a preprocessed patch.

        Args:
            image: preprocessed patch (typically binarized/gray) as ndarray.
            patch_index: optional patch number for logging/debug.
        """
        try:
            # PaddleOCR expects either file paths or images as ndarray (BGR/RGB).
            result = self.ocr.ocr(image, cls=False)
        except Exception as e:  # pragma: no cover
            self.log.error(f"PaddleOCR failed for patch {patch_index}: {e}")
            return OCRResult(text="", confidence=-1.0, raw={"error": str(e)}, patch_index=patch_index)

        # result format: [ [ (box, (text, conf)), ... ] ]
        # For cls=False, confidence comes from recognition.
        texts: list[str] = []
        confs: list[float] = []

        # Normalize structure
        if not result:
            return OCRResult(text="", confidence=-1.0, raw={"paddle": result}, patch_index=patch_index)

        # Some Paddle versions wrap output; handle both.
        blocks = result
        if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
            blocks = result[0]

        for item in blocks:
            # item: (box, (text, conf))
            try:
                text = item[1][0]
                conf = float(item[1][1])
            except Exception:
                continue

            digits = "".join(self._DIGITS_RE.findall(text or ""))
            if digits:
                texts.append(digits)
                confs.append(conf)

        text_out = "".join(texts)
        mean_conf = float(np.mean(confs)) if confs else -1.0

        return OCRResult(text=text_out, confidence=mean_conf, raw={"paddle": result}, patch_index=patch_index)

