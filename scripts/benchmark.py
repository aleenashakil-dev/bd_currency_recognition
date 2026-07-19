"""Simple speed benchmark across a folder of images.

Usage:
    python scripts/benchmark.py --input data/samples
"""
import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import CurrencyRecognitionPipeline  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def main():
    parser = argparse.ArgumentParser(description="Benchmark BDT pipeline speed.")
    parser.add_argument("--input", "-i", required=True, help="Input directory.")
    parser.add_argument("--config", "-c", default=None)
    parser.add_argument("--warmup", type=int, default=1,
                        help="Warmup iterations (excluded from stats).")
    args = parser.parse_args()

    pipe = CurrencyRecognitionPipeline(args.config)
    files = sorted(p for p in Path(args.input).iterdir()
                   if p.suffix.lower() in IMG_EXTS)
    if not files:
        print("No images found.")
        return

    # Warmup
    for _ in range(args.warmup):
        pipe.run(str(files[0]))

    times = []
    for path in files:
        res = pipe.run(str(path))
        times.append(res.elapsed_ms)
        status = res.denomination if res.denomination is not None else "N/A"
        print(f"  {path.name}: {res.elapsed_ms:6.1f} ms → {status}")

    print("\n=== Benchmark ===")
    print(f"Images:     {len(times)}")
    print(f"Mean:       {statistics.mean(times):.1f} ms")
    print(f"Median:     {statistics.median(times):.1f} ms")
    if len(times) > 1:
        print(f"Stdev:      {statistics.stdev(times):.1f} ms")
    print(f"Min / Max:  {min(times):.1f} / {max(times):.1f} ms")
    print(f"Throughput: {1000 / statistics.mean(times):.1f} fps")


if __name__ == "__main__":
    main()
