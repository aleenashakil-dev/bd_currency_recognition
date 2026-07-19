from setuptools import setup, find_packages

setup(
    name="bd_currency_recognition",
    version="0.1.0",
    description="Offline Bangladeshi currency denomination recognition using classical CV",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "pytesseract>=0.3.10",
        "PyYAML>=6.0",
        "click>=8.1.0",
    ],
    entry_points={
        "console_scripts": [
            "bd-currency=src.main:cli",
        ],
    },
)
