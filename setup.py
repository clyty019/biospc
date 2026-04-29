#!/usr/bin/env python
# coding: utf-8

from setuptools import setup, find_packages

setup(
    name="biospc",
    version="0.1.0",
    description="Bio-SPC: A Python library for identifying cliff points in time/pseudotime trajectories based on single-cell data",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Bio-SPC Team",
    author_email="",
    url="https://github.com/your-username/biospc",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scanpy>=1.9.0",
        "torch>=1.10.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "anndata>=0.8.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Intended Audience :: Science/Research",
    ],
    keywords="single-cell, pseudotime, tipping-point, trajectory, bioinformatics",
)
