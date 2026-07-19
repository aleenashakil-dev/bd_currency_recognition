"""Extract and validate a denomination integer from raw OCR text."""
import re
from dataclasses import dataclass

from .constants import BD_DENOMINATIONS_SET


@dataclass
class ValidatedResult:
    """OCR result after validation."""
    raw_text: str                       # Original OCR text
    extracted_number: int | None        # First plausible integer, if any
    is_valid: bool                      # True if extracted_number is a BDT denom
    confidence: float                   # Passed through from OCR


class DenominationValidator:
    """Turn raw OCR strings into validated denomination integers."""

    _DIGIT_RUN = re.compile(r"\d+")

    def __init__(self, config: dict):
        v = config["validation"]
        self.valid = set(v.get("valid_denominations", list(BD_DENOMINATIONS_SET)))
        self.min_digits = int(v.get("min_digits", 1))
        self.max_digits = int(v.get("max_digits", 4))

    def validate(self, raw_text: str, confidence: float = -1.0) -> ValidatedResult:
        """Extract the best candidate integer and check if it's a valid denom.

        Strategy:
        1. Find all digit runs.
        2. Discard runs shorter than min_digits or longer than max_digits.
        3. Try each candidate (longest first) — keep the first that matches a
           valid denomination. If none match exactly, return the longest
           candidate but mark is_valid=False.
        """
        candidates = self._DIGIT_RUN.findall(raw_text or "")
        candidates = [
            c for c in candidates if self.min_digits <= len(c) <= self.max_digits
        ]

        if not candidates:
            return ValidatedResult(
                raw_text=raw_text, extracted_number=None,
                is_valid=False, confidence=confidence,
            )

        # Try longer candidates first (more digits = more discriminative)
        candidates.sort(key=len, reverse=True)

        for c in candidates:
            n = int(c)
            if n in self.valid:
                return ValidatedResult(
                    raw_text=raw_text, extracted_number=n,
                    is_valid=True, confidence=confidence,
                )

        # No exact match — return the best candidate but flag as invalid
        return ValidatedResult(
            raw_text=raw_text, extracted_number=int(candidates[0]),
            is_valid=False, confidence=confidence,
        )
