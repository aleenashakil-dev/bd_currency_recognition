"""End-to-end pipeline: image → aligned note → grid → OCR → validate → vote."""
from dataclasses import dataclass, field
from pathlib import Path
import time

from config import load_config
from .preprocessing import NoteDetector
from .grid import GridSplitter
from .ocr import PaddleOCRDigitOCR, PatchPreprocessor

from .validation import DenominationValidator, ValidatedResult
from .voting import MajorityVoter, VoteResult
from .utils import get_logger, save_debug_image



@dataclass
class PipelineResult:
    """Final result returned by the pipeline."""
    denomination: int | None
    confidence: float
    agreement: int
    total_patches: int
    per_patch: dict[int, ValidatedResult] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    error: str | None = None

    def __str__(self) -> str:
        if self.error:
            return f"ERROR: {self.error}"
        if self.denomination is None:
            return "No denomination detected."
        return (
            f"Denomination: {self.denomination} BDT | "
            f"Confidence: {self.confidence:.1f} | "
            f"Agreement: {self.agreement}/{self.total_patches} | "
            f"Elapsed: {self.elapsed_ms:.0f} ms"
        )


class CurrencyRecognitionPipeline:
    """Full offline BDT recognition pipeline."""

    def __init__(self, config: dict | str | Path | None = None):
        if isinstance(config, dict):
            self.config = config
        else:
            self.config = load_config(config)

        self.log = get_logger(__name__, self.config.get("log_level", "INFO"))
        self.note_detector = NoteDetector(self.config)
        self.grid_splitter = GridSplitter(self.config)
        self.patch_prep = PatchPreprocessor(self.config)
        # OCR initialization can be heavy and may fail if paddleocr or models are missing.
        # Keep it centralized; tests may mock `pipe.ocr.recognize`, so we must still
        # allow pipeline construction.
        try:
            self.ocr = PaddleOCRDigitOCR(self.config)
        except ModuleNotFoundError:
            # Defer actual OCR failures until recognize-time.
            self.ocr = None


        self.validator = DenominationValidator(self.config)
        self.voter = MajorityVoter(self.config)
        self.ocr_patches = self.config["grid"]["ocr_patches"]

        self.debug = self.config.get("debug", False)
        self.debug_dir = self.config.get("debug_output_dir", "data/processed")

    def run(self, image_or_path) -> PipelineResult:
        """Run the full pipeline on an image path or ndarray."""
        t0 = time.perf_counter()
        try:
            # Step 1: detect and align the note
            aligned = self.note_detector.process(image_or_path)
            if aligned is None:
                return PipelineResult(
                    denomination=None, confidence=0.0,
                    agreement=0, total_patches=0,
                    elapsed_ms=(time.perf_counter() - t0) * 1000,
                    error="Note not detected",
                )

            # Step 2: split into 3x3 grid and select target patches
            patches = self.grid_splitter.get_patches_by_index(aligned, self.ocr_patches)
            self.log.debug(f"Extracted {len(patches)} patches: "
                           f"indices={[p.index for p in patches]}")

            # Step 3–4: OCR + validate each patch
            per_patch: dict[int, ValidatedResult] = {}
            for p in patches:
                prepped = self.patch_prep.prepare(p.image)
                if self.debug:
                    save_debug_image(prepped, f"04_patch_{p.index}_binary", self.debug_dir)

                ocr_res = self.ocr.recognize(prepped, patch_index=p.index)

                # Validation + confidence threshold rejection
                # (DenominationValidator checks valid denominations; Pipeline gates confidence.)
                validated = self.validator.validate(ocr_res.text, ocr_res.confidence)

                conf_ok = ocr_res.confidence >= float(self.config["ocr"].get("confidence_threshold", 0.0))
                if not conf_ok:
                    validated.is_valid = False
                    validated.extracted_number = None

                per_patch[p.index] = validated
                self.log.debug(
                    f"Patch {p.index}: raw='{ocr_res.text}' → "
                    f"num={validated.extracted_number} valid={validated.is_valid} "
                    f"conf={ocr_res.confidence:.3f}"
                )


            # Step 5: majority vote
            vote: VoteResult = self.voter.vote(per_patch)

            return PipelineResult(
                denomination=vote.denomination,
                confidence=vote.confidence,
                agreement=vote.agreement,
                total_patches=vote.total_patches,
                per_patch=vote.per_patch,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as e:  # pragma: no cover
            self.log.exception("Pipeline error")
            return PipelineResult(
                denomination=None, confidence=0.0,
                agreement=0, total_patches=0,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                error=str(e),
            )
