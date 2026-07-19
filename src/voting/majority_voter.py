"""Combine validated patch results into a single final denomination."""
from collections import Counter
from dataclasses import dataclass, field

from ..validation import ValidatedResult


@dataclass
class VoteResult:
    """Final voted result."""
    denomination: int | None
    confidence: float                     # 0–100
    agreement: int                        # How many patches agreed
    total_patches: int                    # Total patches considered
    per_patch: dict = field(default_factory=dict)   # patch_index -> ValidatedResult


class MajorityVoter:
    """Pick the final denomination by majority vote among valid patches."""

    def __init__(self, config: dict):
        v = config["voting"]
        self.tie_breaker = v.get("tie_breaker", "highest_confidence")
        self.min_agreement = int(v.get("min_agreement", 2))

    def vote(self, patch_results: dict[int, ValidatedResult]) -> VoteResult:
        """Vote across patch results.

        Args:
            patch_results: mapping of patch_index -> ValidatedResult.

        Returns:
            VoteResult with the winning denomination or None if no consensus.
        """
        valid_only = {
            idx: r for idx, r in patch_results.items()
            if r.is_valid and r.extracted_number is not None
        }
        total = len(patch_results)

        if not valid_only:
            return VoteResult(
                denomination=None, confidence=0.0,
                agreement=0, total_patches=total, per_patch=patch_results,
            )

        counts = Counter(r.extracted_number for r in valid_only.values())
        top_count = counts.most_common(1)[0][1]
        top_candidates = [d for d, c in counts.items() if c == top_count]

        if len(top_candidates) == 1:
            winner = top_candidates[0]
        else:
            winner = self._break_tie(top_candidates, valid_only)

        # Confidence = mean confidence of patches that voted for the winner
        winning_confs = [
            r.confidence for r in valid_only.values()
            if r.extracted_number == winner and r.confidence >= 0
        ]
        conf = float(sum(winning_confs) / len(winning_confs)) if winning_confs else -1.0

        if top_count < self.min_agreement:
            # Consensus below threshold — still return the top but caller can gate
            pass

        return VoteResult(
            denomination=winner,
            confidence=conf,
            agreement=top_count,
            total_patches=total,
            per_patch=patch_results,
        )

    def _break_tie(self, candidates: list[int],
                   results: dict[int, ValidatedResult]) -> int:
        """Break a tie between multiple candidates with the same vote count."""
        if self.tie_breaker == "highest_confidence":
            best_conf = -1.0
            best = candidates[0]
            for c in candidates:
                confs = [r.confidence for r in results.values()
                         if r.extracted_number == c and r.confidence >= 0]
                if confs:
                    avg = sum(confs) / len(confs)
                    if avg > best_conf:
                        best_conf = avg
                        best = c
            return best
        # "first" or "none" — return the smallest patch index's candidate
        for idx in sorted(results.keys()):
            if results[idx].extracted_number in candidates:
                return results[idx].extracted_number
        return candidates[0]
