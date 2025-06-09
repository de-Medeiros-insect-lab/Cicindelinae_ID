# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a scientific research project for taxonomic classification of tiger beetles in the genus *Cicindela* using deep learning. The project implements multilabel classification using fastai and PyTorch to identify species and subspecies from insect specimen images.

## Environment Setup

Create the conda environment:
```bash
conda env create -f environment.yml
conda activate fastai
```

## Key Components

### Data Structure
- Images are organized in `images/` directory with subdirectories for train/valid/test splits
- Each subdirectory name represents a taxonomic label (species or subspecies)
- Label format: `species` or `species_subspecies_variety`

### Model Training Pipeline
1. **Single-label model**: First trains on the most specific taxonomic level available
2. **Multi-label model**: Transfers weights and fine-tunes for hierarchical classification
3. **Weighted sampling**: Addresses class imbalance using inverse frequency weighting

### Exported Models
Models are saved in `exported_fastai_models/`:
- `*_sl.pkl`: Single-label models
- `*_ml.pkl`: Multi-label models
- Available architectures: `eva02_large_patch14_448.mim_m38m_ft_in22k_in1k`, `resnext101_32x16d`

## Running the Project

### Train Models
Open and run the Jupyter notebook:
```bash
jupyter lab Run_multilabel_fastai.ipynb
```

### Generate Analysis Plots
Render the R Markdown analysis:
```bash
Rscript -e "rmarkdown::render('plot_precision_recall.Rmd')"
```

## Data Processing Notes

- The `remove_redundant_subspecies()` function eliminates redundant taxonomic labels where a species always has the same subspecies
- Multi-label setup allows prediction at both species and subspecies levels
- Weighted dataloaders use square root of inverse frequency to balance rare classes

## Hardware Requirements

- Training uses multiple GPUs if available (configured for 2 GPUs with `device_ids=[0,1]`)
- Mixed precision training with `to_fp16()` for memory efficiency
- Large batch processing requires substantial GPU memory

## Key Files

- `Run_multilabel_fastai.ipynb`: Main training pipeline
- `test_results.csv`: Model performance metrics by taxonomic term
- `plot_precision_recall.Rmd`: Analysis and visualization scripts
- `environment.yml`: Complete dependency specification