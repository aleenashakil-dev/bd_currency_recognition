"""Quick single-image test script.

Usage:
    python scripts/run_single.py --image data/samples/note.jpg
    python scripts/run_single.py --image data/samples/note.jpg --debug
"""
import argparse
import sys
from pathlib import Path

# Make src importable when running from repo root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import CurrencyRecognitionPipeline  # noqa: E402
from config import load_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Recognize a single BDT note.")
    parser.add_argument("--image", "-i", required=True, help="Path to input image.")
    parser.add_argument("--config", "-c", default=None, help="Optional config path.")
    parser.add_argument("--debug", action="store_true",
                        help="Save intermediate images to data/processed/.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.debug:
        cfg["debug"] = True
        cfg["log_level"] = "DEBUG"

    pipe = CurrencyRecognitionPipeline(cfg)
    result = pipe.run(args.image)

    print("=" * 60)
    print(result)
    print("=" * 60)
    if result.per_patch:
        print("\nPer-patch details:")
        for idx, r in sorted(result.per_patch.items()):
            print(f"  patch {idx}: raw='{r.raw_text}' → "
                  f"num={r.extracted_number}  valid={r.is_valid}  "
                  f"conf={r.confidence:.1f}")


if __name__ == "__main__":
    main()
