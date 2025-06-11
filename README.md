# Cicindela Classification using Deep Learning

This repository contains code for taxonomic classification of tiger beetles in the genus *Cicindela* using deep learning multilabel classification with fastai and PyTorch.

## Citation

This code was created by Bruno A. S. de Medeiros and is used in the following paper:

Postema, E. G., Briscoe, L., Harder, C., Hancock, G. R. A., Guarnieri, L. D., Eisel, T., Welch, K., de Souza, D., Phillip, D., Baquiran, R., Sepulveda, T., Ree, R., & de Medeiros, B. A. S. (in preparation). DrawerDissect: Whole-drawer insect imaging, segmentation, and transcription using AI.

## Overview

The project implements a hierarchical multilabel classification system to identify *Cicindela* species and subspecies from specimen images. The approach uses transfer learning with modern vision architectures and addresses class imbalance through weighted sampling.

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
- **Multiple architectures**: Supports EVA-02 and ResNeXt models
- **Mixed precision training**: Optimized for GPU memory efficiency

## Model Files

Trained models are saved in `exported_fastai_models/`:
- `*_sl.pkl`: Single-label models
- `*_ml.pkl`: Multi-label models

## Hardware Requirements

- Multiple GPU support (configured for 2 GPUs)
- Substantial GPU memory for large batch processing
- Mixed precision training support

## License

See LICENSE file for details.