# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Scientific research project for taxonomic classification of beetles across multiple genera using deep learning. A single multilabel classifier is trained jointly on *Cicindela* (tiger beetles) and *Platydracus* (rove beetles); performance is then reported per-genus. Built on fastai + PyTorch.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate fastai
```

## Key Components

### Data Structure

- `images/` is a symlink to `/data/bdemedeiros/DrawerDissect_Taxon_ID_data/`.
- Layout: `images/<Genus>/{train,valid,test}/<species_[subsp]>/*.{png,jpg}`.
- Genera are discovered at runtime (`GENERA = sorted(images_root.iterdir()…)`) — adding a third genus is a matter of dropping a directory.

### Label scheme

Each specimen carries a multilabel set of 2 or 3 entries:

- **Genus** (capitalized): `Cicindela`, `Platydracus`.
- **Species** (genus-prefixed, lowercase, exactly one `_`): `cicindela_repanda`, `platydracus_angusticeps`.
- **Subspecies** (genus-prefixed, two or more `_`): `cicindela_repanda_repanda`. Hybrid folder names (≥2 `_` after the prefix) also land here.

`taxonomic_level(label)` implements the rule: capitalized → genus; one `_` → species; ≥2 `_` → subspecies.

### Model Training Pipeline

1. **Single-label pretraining**: uses the most specific label per specimen (concatenated genus_species[_subsp]) as a single category.
2. **Multi-label transfer**: loads single-label weights into a `MultiCategoryBlock` model and fine-tunes with `AsymmetricLossMultiLabel`.
3. **Weighted sampling**: square-root inverse-frequency weights balance rare classes.

### Evaluation

Per-term metrics are computed on the test set and written to `test_results_<arch>.csv`. Each row carries `Term`, `genus`, `taxonomic_level`, `N_train`, `N`, precision/recall, and confusion counts. Per-genus and per-(genus, level) micro-averages are also printed by the notebook.

### Exported Models

Saved in `exported_fastai_models/`:
- `*_sl.pkl`: single-label models
- `*_ml.pkl`: multi-label models

Available architectures: `eva02_large_patch14_448.mim_m38m_ft_in22k_in1k`, `resnext101_32x16d`.

## Running the Project

```bash
jupyter lab Run_multilabel_fastai.ipynb
Rscript -e "rmarkdown::render('plot_precision_recall.Rmd')"
```

## Data Processing Notes

- `remove_redundant_subspecies()` drops subspecies labels where a species always has exactly one subspecies. Operates on the genus-prefixed labels.
- Hybrid specimen folders were renamed on disk so the same folder names appear in train, valid, and test (previously the test split used a simplified name — now all three splits match).

## Hardware Requirements

- Multi-GPU training (configured for 2 GPUs via `device_ids=[0,1]`).
- Mixed-precision (`to_fp16()`) and large batches.

## Key Files

- `Run_multilabel_fastai.ipynb`: training + evaluation pipeline.
- `plot_precision_recall.Rmd`: per-genus faceted performance figures.
- `push_to_hf.py`: HuggingFace upload script (updated for the multi-genus model).
- `test_results_<arch>.csv`: per-term metrics with genus column.
- `environment.yml`: dependency specification.
