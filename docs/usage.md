# Usage

## Quick start

```bash
python scripts/run_single.py --image data/samples/note_500.jpg
```

Output:
```
============================================================
Denomination: 500 BDT | Confidence: 87.3 | Agreement: 3/3 | Elapsed: 210 ms
============================================================

Per-patch details:
  patch 1: raw='500' → num=500  valid=True  conf=90.1
  patch 3: raw='500' → num=500  valid=True  conf=86.4
  patch 9: raw='500' → num=500  valid=True  conf=85.5
```

## Batch mode

```bash
python scripts/run_batch.py --input data/raw --output results.csv
```

Produces a CSV like:
```
file,denomination,confidence,agreement,total_patches,elapsed_ms,error
note_100.jpg,100,86.20,3,3,198,
note_500.jpg,500,91.40,3,3,203,
blurry_note.jpg,,0.00,0,0,145,Note not detected
```

## Debug mode

```bash
python scripts/run_single.py --image data/samples/note.jpg --debug
```

Writes intermediate images to `data/processed/`:
- `01_resized.png` — after downsizing
- `02_boundary.png` — detected note boundary overlaid on input
- `03_aligned.png` — perspective-corrected top-down view
- `04_patch_{1,3,9}_binary.png` — each corner patch after OCR preprocessing

Inspect these when accuracy is low — the failure is almost always visible.

## Programmatic use

```python
from src.pipeline import CurrencyRecognitionPipeline

# Load default config from config/config.yaml
pipe = CurrencyRecognitionPipeline()

# Or with a custom config path
pipe = CurrencyRecognitionPipeline("path/to/my_config.yaml")

# Or with an in-memory dict
from config import load_config
cfg = load_config()
cfg["debug"] = True
cfg["preprocessing"]["resize_max_dim"] = 1000
pipe = CurrencyRecognitionPipeline(cfg)

result = pipe.run("note.jpg")
print(result.denomination)   # e.g. 500
print(result.confidence)     # e.g. 87.3
print(result.per_patch)      # dict of per-patch ValidatedResult
```

## Tuning

Edit `config/config.yaml`. See `docs/architecture.md` § "Key tradeoffs" for
what to change when.
