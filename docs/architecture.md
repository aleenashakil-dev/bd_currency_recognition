# Architecture

## Design goals

- **Fully offline.** No network calls at inference time. No model downloads at runtime.
- **Lightweight.** Runs on modest hardware — laptop CPU, Raspberry Pi 4, mid-range Android.
- **Fast.** Target < 500 ms per image on a modern laptop CPU.
- **Auditable.** Every stage produces an inspectable intermediate output.
- **Classical + OCR.** OpenCV for image ops, PaddleOCR for OCR. No cloud APIs.


## Stage-by-stage

### 1. Preprocessing (CamScanner-style)
`src/preprocessing/`

- `image_utils.py` — resize, grayscale, `order_points`, adaptive Canny thresholds.
- `boundary_detector.py` — Gaussian blur → Canny → dilation → contour search → largest quadrilateral via `approxPolyDP`. Falls back to `minAreaRect` if no 4-point polygon is found.
- `perspective_correction.py` — Four-point `getPerspectiveTransform` + `warpPerspective` to a fixed 1200×500 target. Auto-orients if source region was portrait.
- `note_detector.py` — Ties the above together and exposes `.process(image_or_path)`.

### 2. Grid split
`src/grid/grid_splitter.py`

Splits the aligned note into a 3×3 grid (9 `Patch` objects, 1-indexed row-major). Optional padding so numerals near cell edges aren't clipped. Only patches **1, 3, 9** (top-left, top-right, bottom-right) are passed to OCR by default — these correspond to the corners where BDT numerals are printed.

### 3. OCR
`src/ocr/`

- `ocr_preprocessor.py` — Per-patch: upscale ×3 → CLAHE → bilateral filter → Otsu binarize → auto-invert if dark background → morphological open. Getting the numerals into a clean high-contrast state is where most of the accuracy comes from.
- `tesseract_ocr.py` — Wraps `pytesseract.image_to_data` with digit whitelist (`0123456789`) and PSM 7 (single line of text). Returns tokens + mean confidence.

### 4. Validation
`src/validation/`

- `constants.py` — `BD_DENOMINATIONS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]`.
- `denomination_validator.py` — Regex out all digit runs from the OCR string, filter by min/max length, prefer the longest that matches a valid denomination. Returns a `ValidatedResult` with `is_valid` flag.

### 5. Voting
`src/voting/majority_voter.py`

`Counter`-based majority over the validated patch results. Configurable tie-breaker (`highest_confidence` by default). Returns final denomination + agreement count + mean confidence.

## Orchestration

`src/pipeline.py::CurrencyRecognitionPipeline` chains everything and returns a `PipelineResult` dataclass with:

- `denomination` (int | None)
- `confidence` (0–100)
- `agreement` and `total_patches`
- `per_patch` (dict of `ValidatedResult` by patch index) — for debugging
- `elapsed_ms`
- `error` (str | None)

Debug mode (`debug: true` in config) dumps intermediate images to `data/processed/`:

- `01_resized.png` — after downsizing
- `02_boundary.png` — detected quadrilateral overlaid
- `03_aligned.png` — perspective-corrected note
- `04_patch_{i}_binary.png` — each patch after OCR preprocessing

## Key tradeoffs & where to tune

| Symptom | Where to look | Typical fix |
| --- | --- | --- |
| Note not detected | `preprocessing.canny`, `preprocessing.contour.min_area_ratio` | Lower `min_area_ratio`; try `canny.adaptive: false` with tuned thresholds |
| Boundary is bad but detected | `preprocessing.contour.approx_epsilon` | Reduce epsilon (0.01–0.02) for tighter polygon fit |
| Aligned note is upside-down | Add a second pass with rotated patches | Handled at voter level — outside current scope |
| OCR returns garbage | `ocr.patch_preprocessing` | Increase `upscale_factor`; switch `threshold_method` to `adaptive` |
| OCR reads "500" as "5OO" or "S00" | `ocr.char_whitelist` | Whitelist already restricts to digits — check binarization instead |
| Slow on Raspberry Pi | `preprocessing.resize_max_dim` | Lower to 1000 or 800; disable debug mode |
