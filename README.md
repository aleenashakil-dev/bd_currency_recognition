# Offline Bangladeshi Currency Denomination Recognition

A completely offline, lightweight system that recognizes the denomination of
Bangladeshi currency notes (Taka) using OpenCV + PaddleOCR.

**No AI models/cloud APIs. No internet required after install (models must be available locally).**


## Pipeline

```
Input Image
     │
     ▼
CamScanner-style Preprocessing (OpenCV)
     ├─ Detect currency note
     ├─ Remove background
     ├─ Detect boundaries
     ├─ Perspective correction
     └─ Return aligned note
     │
     ▼
Split into 3×3 Grid  →  9 patches
     │
     ▼
OCR on Patches 1, 3, 9  (corners where denomination appears)
     │
     ▼
Validate OCR Results  →  keep only {1, 2, 5, 10, 20, 50, 100, 200, 500, 1000}
     │
     ▼
Majority Voting
     │
     ▼
Final Denomination
```

## Project Structure

```
bd_currency_recognition/
├── config/                 # YAML configs (thresholds, paths)
├── src/
│   ├── preprocessing/      # Note detection, boundary, perspective warp
│   ├── grid/               # 3x3 grid splitter
│   ├── ocr/                # Tesseract wrapper + patch preprocessing
│   ├── validation/         # Filter to valid BDT denominations
│   ├── voting/             # Majority vote
│   ├── utils/              # Logging, visualization helpers
│   ├── pipeline.py         # Orchestrates full workflow
│   └── main.py             # CLI entry point
├── tests/                  # Unit tests per stage
├── data/
│   ├── raw/                # Original test images
│   ├── processed/          # Debug intermediate outputs
│   └── samples/            # Committed sample images
├── notebooks/              # Jupyter notebooks for tuning/exploration
├── scripts/                # run_single, run_batch, benchmark
└── docs/                   # Architecture + usage docs
```

## Setup

```bash
# 1. Install Tesseract OCR (system dependency)
#    Ubuntu:  sudo apt install tesseract-ocr
#    macOS:   brew install tesseract
#    Windows: https://github.com/UB-Mannheim/tesseract/wiki

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Verify Tesseract is on PATH
tesseract --version
```

## Usage

```bash
# Run on a single image
python scripts/run_single.py --image data/samples/note_500.jpg

# Run on a folder (batch)
python scripts/run_batch.py --input data/raw --output results.csv

# Benchmark speed on your device
python scripts/benchmark.py --input data/samples
```

Or programmatically:

```python
from src.pipeline import CurrencyRecognitionPipeline

pipe = CurrencyRecognitionPipeline()
result = pipe.run("path/to/note.jpg")
print(result.denomination, result.confidence)
```

## Development

```bash
# Run tests
pytest tests/ -v

# Explore in notebooks
jupyter lab notebooks/
```

## Supported Denominations

BDT ৳1, ৳2, ৳5, ৳10, ৳20, ৳50, ৳100, ৳200, ৳500, ৳1000

## Notes

- Patches 1, 3, 9 correspond to top-left, top-right, and bottom-right corners
  of the 3×3 grid (1-indexed), which is where BDT denomination numerals are
  most prominently printed.
- All thresholds are configurable via `config/config.yaml`.
- Set `debug: true` in config to dump intermediate images to `data/processed/`.
