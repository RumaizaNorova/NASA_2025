#!/usr/bin/env python3
"""
Setup script for Sharks from Space - NASA Space Apps Challenge 2025
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements from environment.yml
def read_requirements():
    requirements = []
    with open("environment.yml", "r") as f:
        in_deps = False
        for line in f:
            line = line.strip()
            if line.startswith("dependencies:"):
                in_deps = True
                continue
            if in_deps and line and not line.startswith("-") and not line.startswith("python="):
                if "=" in line:
                    requirements.append(line)
                else:
                    requirements.append(line)
            elif in_deps and line.startswith("-"):
                continue
            elif in_deps and not line:
                break
    return requirements

setup(
    name="sharks-from-space",
    version="1.0.0",
    author="NASA Space Apps Challenge 2025 Team",
    author_email="contact@sharksfromspace.nasa",
    description="Advanced shark habitat prediction using NASA satellite data",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/nasa-space-apps/sharks-from-space",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "scikit-image>=0.18.0",
        "xarray>=0.20.0",
        "rioxarray>=0.9.0",
        "rasterio>=1.2.0",
        "xesmf>=0.6.0",
        "netCDF4>=1.5.0",
        "shapely>=1.8.0",
        "geopy>=2.2.0",
        "pyproj>=3.2.0",
        "xgboost>=1.5.0",
        "lightgbm>=3.3.0",
        "shap>=0.41.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "folium>=0.12.0",
        "tqdm>=4.62.0",
        "requests>=2.26.0",
        "aiohttp>=3.8.0",
        "python-dotenv>=0.19.0",
        "click>=8.0.0",
        "pillow>=8.3.0",
        "notebook>=6.4.0",
        "jupyterlab>=3.2.0",
        "dask>=2021.10.0",
        "zarr>=2.10.0",
        "lz4>=3.1.0",
        "pyyaml>=6.0",
        "pytest>=6.2.0",
        "plotly>=5.3.0",
        "ipywidgets>=7.6.0",
        "mapboxgl>=0.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.9.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.2.0",
            "sphinx-rtd-theme>=1.0.0",
            "nbsphinx>=0.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sharks-data-scout=src.data_scout:main",
            "sharks-fetch-data=src.fetch_data:main",
            "sharks-compute-features=src.compute_features:main",
            "sharks-label-join=src.label_join:main",
            "sharks-train-model=src.train_model:main",
            "sharks-predict=src.predict_grid:main",
            "sharks-make-maps=src.make_maps:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.md", "*.txt", "*.json"],
    },
    zip_safe=False,
)


