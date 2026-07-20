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
    """Weighted majority vote over validated patch results.

    Weight of each vote equals the OCR confidence score.
    """

    def __init__(self, config: dict):
        v = config.get("voting", {})
        self.tie_breaker = v.get("tie_breaker", "highest_confidence")
        self.min_agreement = int(v.get("min_agreement", 2))

    def vote(self, patch_results: dict[int, ValidatedResult]) -> VoteResult:
        """Vote across patch results.

        Only patches with `is_valid=True` participate.
        Each vote weight equals `ValidatedResult.confidence`.
        """
        total = len(patch_results)

        valid_only = {
            idx: r for idx, r in patch_results.items()
            if r.is_valid and r.extracted_number is not None and r.confidence is not None
        }

        if not valid_only:
            return VoteResult(
                denomination=None, confidence=0.0,
                agreement=0, total_patches=total, per_patch=patch_results,
            )

        weights_by_denom: dict[int, float] = {}
        count_by_denom: dict[int, int] = {}
        for r in valid_only.values():
            denom = int(r.extracted_number)
            w = float(r.confidence) if r.confidence is not None else 0.0
            weights_by_denom[denom] = weights_by_denom.get(denom, 0.0) + w
            count_by_denom[denom] = count_by_denom.get(denom, 0) + 1

        max_weight = max(weights_by_denom.values())
        top_candidates = [d for d, w in weights_by_denom.items() if w == max_weight]

        if len(top_candidates) == 1:
            winner = top_candidates[0]
        else:
            winner = self._break_tie(top_candidates, valid_only)

        winning_patches = [
            r for r in valid_only.values()
            if int(r.extracted_number) == winner
        ]
        winning_confs = [float(r.confidence) for r in winning_patches if r.confidence is not None]
        conf = float(sum(winning_confs) / len(winning_confs)) if winning_confs else -1.0

        agreement = int(count_by_denom.get(winner, 0))
        if agreement < self.min_agreement:
            # keep winner anyway; caller can gate on confidence/agreement
            pass

        return VoteResult(
            denomination=winner,
            confidence=conf,
            agreement=agreement,
            total_patches=total,
            per_patch=patch_results,
        )

    def _break_tie(self, candidates: list[int],
                   results: dict[int, ValidatedResult]) -> int:
        """Tie breaker among equal total weight candidates."""
        if self.tie_breaker == "highest_confidence":
            best = candidates[0]
            best_avg = -1.0
            for c in candidates:
                confs = [float(r.confidence) for r in results.values()
                         if r.is_valid and r.extracted_number == c and r.confidence is not None]
                avg = sum(confs) / len(confs) if confs else -1.0
                if avg > best_avg:
                    best_avg = avg
                    best = c
            return best

        # "first" — smallest patch index wins
        for idx in sorted(results.keys()):
            if results[idx].is_valid and results[idx].extracted_number in candidates:
                return int(results[idx].extracted_number)
        return int(candidates[0])

