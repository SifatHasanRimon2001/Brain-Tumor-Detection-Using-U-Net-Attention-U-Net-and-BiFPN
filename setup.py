"""Setup script for the BRISC2025 Brain Tumor project."""

from setuptools import find_packages, setup

setup(
    name="brisc-tumor",
    version="1.0.0",
    description="Brain Tumor Segmentation & Classification using U-Net variants",
    author="CSE428 Project",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.8.0",
        "albumentations>=1.3.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
)
