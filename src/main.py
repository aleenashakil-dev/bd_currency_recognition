"""CLI entry point for the currency recognition pipeline."""
from pathlib import Path
import click

from .pipeline import CurrencyRecognitionPipeline


@click.group()
def cli():
    """Offline Bangladeshi Currency Denomination Recognition."""


@cli.command()
@click.option("--image", "-i", required=True, type=click.Path(exists=True, dir_okay=False),
              help="Path to input image.")
@click.option("--config", "-c", default=None, type=click.Path(exists=True, dir_okay=False),
              help="Optional path to a config.yaml file.")
@click.option("--debug/--no-debug", default=False,
              help="Save intermediate images to data/processed/.")
def recognize(image: str, config: str | None, debug: bool):
    """Recognize the denomination of a single note image."""
    from config import load_config
    cfg = load_config(config)
    if debug:
        cfg["debug"] = True

    pipe = CurrencyRecognitionPipeline(cfg)
    result = pipe.run(image)
    click.echo(str(result))

    if result.per_patch:
        click.echo("\nPer-patch details:")
        for idx, r in sorted(result.per_patch.items()):
            click.echo(
                f"  patch {idx}: raw='{r.raw_text}' → num={r.extracted_number} "
                f"valid={r.is_valid} conf={r.confidence:.1f}"
            )


@cli.command()
@click.option("--input", "-i", "input_dir", required=True,
              type=click.Path(exists=True, file_okay=False))
@click.option("--output", "-o", default="results.csv",
              type=click.Path(dir_okay=False))
@click.option("--config", "-c", default=None,
              type=click.Path(exists=True, dir_okay=False))
def batch(input_dir: str, output: str, config: str | None):
    """Run recognition on a directory of images."""
    import csv

    pipe = CurrencyRecognitionPipeline(config)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    files = [p for p in Path(input_dir).iterdir() if p.suffix.lower() in exts]

    with open(output, "w", newline="", encoding="utf-8") as f:
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
            click.echo(f"{path.name}: {res}")

    click.echo(f"\nSaved {len(files)} results → {output}")


if __name__ == "__main__":
    cli()
