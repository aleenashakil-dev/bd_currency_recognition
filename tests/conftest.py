"""Shared pytest fixtures and path setup."""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def default_config():
    """Load the default project config for tests."""
    from config import load_config
    return load_config()


@pytest.fixture
def dummy_note_image():
    """Create a synthetic 'note' image: white rectangle on black background."""
    img = np.zeros((800, 1200, 3), dtype=np.uint8)
    # Draw a bright rectangle simulating a note
    img[100:700, 150:1050] = (240, 240, 240)
    return img


@pytest.fixture
def dummy_aligned_note():
    """A fake aligned note image sized like real output."""
    return np.full((500, 1200, 3), 220, dtype=np.uint8)
