# Setup

## System requirements

- Python 3.9+
- Tesseract OCR 4.x or 5.x
- ~200 MB disk for dependencies

## Install Tesseract

### Ubuntu / Debian
```bash
sudo apt update
sudo apt install tesseract-ocr libtesseract-dev
```

### macOS (Homebrew)
```bash
brew install tesseract
```

### Windows
1. Download the installer: https://github.com/UB-Mannheim/tesseract/wiki
2. Add the install path (e.g. `C:\Program Files\Tesseract-OCR`) to your `PATH`.
3. Verify: open a new terminal and run `tesseract --version`.

If Tesseract is installed but not on `PATH`, set its full path in `config/config.yaml`:

```yaml
ocr:
  tesseract_cmd: "/full/path/to/tesseract"
```

## Python environment

```bash
git clone <your-repo-url>
cd bd_currency_recognition

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Verify install

```bash
# Should print version info without errors
tesseract --version

# Should print help
python scripts/run_single.py --help

# Run the unit tests
pytest tests/ -v
```

## Optional: install the package in editable mode

```bash
pip install -e .
# Then you can use:
bd-currency recognize --image path/to/note.jpg
```

## Preparing sample data

Drop test images into `data/samples/` (small, committed) or `data/raw/`
(larger, gitignored). Filenames like `note_500_front.jpg` make batch
evaluation easier.
