# Cicindela Classification using Deep Learning

This repository contains code for taxonomic classification of tiger beetles in the genus *Cicindela* using deep learning multilabel classification with fastai and PyTorch.

## Citation

This code was created by Bruno A. S. de Medeiros and is used in the following paper:

Postema, E. G., Briscoe, L., Harder, C., Hancock, G. R. A., Guarnieri, L. D., Eisel, T., Welch, K., de Souza, D., Phillip, D., Baquiran, R., Sepulveda, T., Ree, R., & de Medeiros, B. A. S. (in preparation). DrawerDissect: Whole-drawer insect imaging, segmentation, and transcription using AI.

## Overview

The project implements a multilabel classification system to identify *Cicindela* species and subspecies from specimen images. The approach uses transfer learning with modern vision architectures and addresses class imbalance through weighted sampling and asymmetric loss.

## Environment Setup

Create the conda environment:
```bash
conda env create -f environment.yml
conda activate fastai
```

## Usage

### Training Models
Open and run the main Jupyter notebook:
```bash
jupyter lab Run_multilabel_fastai.ipynb
```

### Generating Analysis
Render the precision-recall analysis:
```bash
Rscript -e "rmarkdown::render('plot_precision_recall.Rmd')"
```

## Key Features

- **Hierarchical classification**: Predicts both species and subspecies levels
- **Weighted sampling**: Addresses class imbalance using inverse frequency weighting
- **Multiple architectures**: Supports timm library models
- **Mixed precision training**: Optimized for GPU memory efficiency

## Model Files

Trained models are saved in `exported_fastai_models/`:
- `*_sl.pkl`: Single-label models
- `*_ml.pkl`: Multi-label models

These are not part of this repository due to their large size. See https://huggingface.co/brunoasm/eva02_large_patch14_448.Cicindela_ID_FMNH for the final multilabel model.

## Hardware Requirements

- Multiple GPU support (configured for 2 GPUs)
- Substantial GPU memory for large batch processing
- Mixed precision training support

## File Contents

### Core Files
- `Run_multilabel_fastai.ipynb` - Main training pipeline and model development notebook
- `environment.yml` - Conda environment specification with all dependencies
- `push_to_hf.py` - Script to upload trained models to Hugging Face Hub
- `CLAUDE.md` - Development guidelines and project documentation for AI assistants

### Data and Images
- `images/` - Dataset directory containing train/valid/test splits organized by taxonomic labels. The actual files are not available in this repository due to storage size.
- `unknowns/` - Test images for model evaluation and inference

### Models and Results
- `exported_fastai_models/` - Directory containing trained model files (.pkl format)
- `models/` - Additional model storage and configurations
- `test_results_*.csv` - Model performance metrics by taxonomic classification
- `taxonomy_metrics.pdf` and `taxonomy_metrics.png` - Performance visualization charts

### Analysis and Visualization
- `plot_precision_recall.Rmd` - R Markdown script for generating performance analysis plots
- `plot_precision_recall.nb.html` - Rendered HTML output of the precision-recall analysis

### Project Configuration
- `Cicindela_id.Rproj` - R Studio project configuration file
- `LICENSE` - Project license information
- `README.md` - This documentation file

## License

See LICENSE file for details.
