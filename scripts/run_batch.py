"""Run recognition on every image in a directory and dump results to CSV.

Usage:
    python scripts/run_batch.py --input data/raw --output results.csv
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import CurrencyRecognitionPipeline  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def main():
    parser = argparse.ArgumentParser(description="Batch BDT recognition.")
    parser.add_argument("--input", "-i", required=True, help="Input directory.")
    parser.add_argument("--output", "-o", default="results.csv", help="Output CSV.")
    parser.add_argument("--config", "-c", default=None, help="Optional config path.")
    args = parser.parse_args()

    pipe = CurrencyRecognitionPipeline(args.config)
    files = sorted(p for p in Path(args.input).iterdir()
                   if p.suffix.lower() in IMG_EXTS)

    print(f"Processing {len(files)} images...")
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "denomination", "confidence",
                         "agreement", "total_patches", "elapsed_ms", "error"])
        for path in files:
            res = pipe.run(str(path))
            writer.writerow([
                path.name, res.denomination, f"{res.confidence:.2f}",
                res.agreement, res.total_patches,
                f"{res.elapsed_ms:.0f}", res.error or "",
            ])
            print(f"  {path.name}: {res}")

    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
