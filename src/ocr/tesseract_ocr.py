"""Thin wrapper around pytesseract with digit-only whitelist."""
from dataclasses import dataclass
import numpy as np
import pytesseract

from ..utils import get_logger


@dataclass
class OCRResult:
    """OCR result for a single patch."""
    text: str            # Raw recognized string (digits only after whitelist)
    confidence: float    # Mean confidence in [0, 100], or -1 if unavailable
    raw: dict            # Full pytesseract image_to_data output (for debugging)


class TesseractOCR:
    """Digit-only OCR tuned for currency denomination numerals."""

    def __init__(self, config: dict):
        cfg = config["ocr"]
        self.log = get_logger(__name__, config.get("log_level", "INFO"))

        if cfg.get("tesseract_cmd"):
            pytesseract.pytesseract.tesseract_cmd = cfg["tesseract_cmd"]

        self.lang = cfg.get("language", "eng")
        self.psm = int(cfg.get("psm", 7))
        self.oem = int(cfg.get("oem", 3))
        self.whitelist = cfg.get("char_whitelist", "0123456789")

    def _tesseract_config(self) -> str:
        return (
            f"--oem {self.oem} --psm {self.psm} "
            f"-c tessedit_char_whitelist={self.whitelist}"
        )

    def recognize(self, image: np.ndarray) -> OCRResult:
        """Run OCR on a preprocessed (binary) patch and return digits + confidence."""
        try:
            data = pytesseract.image_to_data(
                image,
                lang=self.lang,
                config=self._tesseract_config(),
                output_type=pytesseract.Output.DICT,
            )
        except pytesseract.TesseractNotFoundError:
            self.log.error(
                "Tesseract binary not found. Install it and ensure it's on PATH, "
                "or set ocr.tesseract_cmd in config.yaml."
            )
            return OCRResult(text="", confidence=-1.0, raw={})

        # Aggregate all non-empty tokens
        tokens: list[str] = []
        confidences: list[float] = []
        for tok, conf in zip(data.get("text", []), data.get("conf", [])):
            tok = tok.strip()
            if not tok:
                continue
            try:
                conf_val = float(conf)
            except (TypeError, ValueError):
                conf_val = -1.0
            tokens.append(tok)
            if conf_val >= 0:
                confidences.append(conf_val)

        text = "".join(tokens)
        mean_conf = float(np.mean(confidences)) if confidences else -1.0
        return OCRResult(text=text, confidence=mean_conf, raw=data)
