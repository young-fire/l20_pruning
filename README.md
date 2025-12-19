# Code for Structured Pruning via Cross-Layer Metric and $\ell_{2,0} \text {-norm}$ Sparse Reconstruction
Neural Network Pruning Pipeline

A complete pipeline for neural network compression using Activation Probability of Zero (APoZ) metrics, feature extraction, and fine-tuning.
Overview

This repository implements a three-stage neural network pruning pipeline:

1. APoZ Generation with Standardization: Generate APoZ metrics and apply standardization normalization
2. Feature Extraction: Extract relevant features for pruning decisions
3. Fine-tuning: Implement cross-layer ranking and sparse reconstruction
Usage
Stage 1: Generate APoZ Files with Standardization
bash
cd generation_and_std && sh apoz_gen.sh && python gyh.py
Stage 2: Extract Features
bash
sh feature_gen.sh
Stage 3: Fine-tuning with Cross-Layer Ranking and Sparse Reconstruction
bash
cd l20_pruning && sh run.sh
Implementation Details
Cross-rank code: Located in std_layer.py
L20 sparse reconstruction code: Located in l20.py
File Structure

project-root/
├── generation_and_std/
│ ├── apoz_gen.sh # APoZ generation script
│ └── gyh.py # Standardization normalization
├── feature_gen.sh # Feature extraction pipeline
└── l20_pruning/
├── run.sh # Fine-tuning execution script
├── std_layer.py # Cross-layer ranking implementation
└── l20.py # L20 sparse reconstruction
Requirements
Python 3.7+
PyTorch
NumPy
Bash environment
License

MIT License
