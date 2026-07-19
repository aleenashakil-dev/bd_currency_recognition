"""Tests for OCR validation logic."""
from src.validation import DenominationValidator, BD_DENOMINATIONS


class TestDenominationValidator:
    def test_recognizes_all_valid_denominations(self, default_config):
        v = DenominationValidator(default_config)
        for d in BD_DENOMINATIONS:
            result = v.validate(str(d))
            assert result.extracted_number == d
            assert result.is_valid

    def test_rejects_invalid_number(self, default_config):
        v = DenominationValidator(default_config)
        result = v.validate("123")   # Not a real BDT denom
        assert result.extracted_number == 123
        assert not result.is_valid

    def test_returns_none_for_empty(self, default_config):
        v = DenominationValidator(default_config)
        result = v.validate("")
        assert result.extracted_number is None
        assert not result.is_valid

    def test_extracts_first_valid_from_noise(self, default_config):
        v = DenominationValidator(default_config)
        result = v.validate("abc500xyz")
        assert result.extracted_number == 500
        assert result.is_valid

    def test_prefers_longer_valid_candidate(self, default_config):
        # OCR sometimes reads "1000" as "100 0" — validator should still catch it
        v = DenominationValidator(default_config)
        result = v.validate("500extra")
        assert result.extracted_number == 500

    def test_confidence_passthrough(self, default_config):
        v = DenominationValidator(default_config)
        result = v.validate("100", confidence=87.5)
        assert result.confidence == 87.5
