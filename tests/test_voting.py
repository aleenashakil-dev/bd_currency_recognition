"""Tests for the majority voter."""
from src.validation import ValidatedResult
from src.voting import MajorityVoter


def _mk(num: int | None, conf: float = 80.0, valid: bool = True) -> ValidatedResult:
    return ValidatedResult(
        raw_text=str(num) if num is not None else "",
        extracted_number=num, is_valid=valid, confidence=conf,
    )


class TestMajorityVoter:
    def test_all_agree(self, default_config):
        voter = MajorityVoter(default_config)
        results = {1: _mk(500), 3: _mk(500), 9: _mk(500)}
        vote = voter.vote(results)
        assert vote.denomination == 500
        assert vote.agreement == 3
        assert vote.total_patches == 3

    def test_two_agree_one_disagrees(self, default_config):
        voter = MajorityVoter(default_config)
        results = {1: _mk(100), 3: _mk(100), 9: _mk(500)}
        vote = voter.vote(results)
        assert vote.denomination == 100
        assert vote.agreement == 2

    def test_all_disagree_uses_tie_breaker(self, default_config):
        voter = MajorityVoter(default_config)
        results = {
            1: _mk(100, conf=50.0),
            3: _mk(500, conf=90.0),
            9: _mk(1000, conf=70.0),
        }
        vote = voter.vote(results)
        # Weighted tie-breaker: highest average confidence should win
        assert vote.denomination == 500


    def test_no_valid_patches(self, default_config):
        voter = MajorityVoter(default_config)
        results = {
            1: _mk(None, valid=False),
            3: _mk(123, valid=False),
            9: _mk(None, valid=False),
        }
        vote = voter.vote(results)
        assert vote.denomination is None
        assert vote.agreement == 0
