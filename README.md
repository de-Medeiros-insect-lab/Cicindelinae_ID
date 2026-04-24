# Beetle Taxonomic Classification (Cicindela & Platydracus) using Deep Learning

This repository contains code for taxonomic classification of tiger beetles (*Cicindela*) and rove beetles (*Platydracus*) using a single deep-learning multilabel classifier built with fastai and PyTorch.

## Citation

This code was created by Bruno A. S. de Medeiros and is used in the following paper:

Postema, E. G., Briscoe, L., Harder, C., Hancock, G. R. A., Guarnieri, L. D., Eisel, T., Welch, K., de Souza, D., Phillip, D., Baquiran, R., Sepulveda, T., Ree, R., & de Medeiros, B. A. S. (in preparation). DrawerDissect: Whole-drawer insect imaging, segmentation, and transcription using AI.

## Overview

One multi-label classifier is trained on the union of *Cicindela* and *Platydracus* specimen images. The label vocabulary is a three-level hierarchy:

- **Genus**: capitalized — `Cicindela`, `Platydracus`.
- **Species**: genus-prefixed, lowercase — e.g. `cicindela_repanda`, `platydracus_angusticeps`.
- **Subspecies** (when present): genus-prefixed — e.g. `cicindela_repanda_repanda`.

Genus-prefixing guarantees uniqueness across taxa and makes the taxonomic level of every label decidable by the underscore-count rule. At evaluation time the test set is split by genus so per-genus precision/recall can be reported alongside global metrics.

## Data layout

Images are expected under `images/`, which is a symlink to the drive-mounted data root:

```
images -> /data/bdemedeiros/DrawerDissect_Taxon_ID_data/
  Cicindela/{train,valid,test}/<species_[subsp]>/*.png
  Platydracus/{train,valid,test}/<species_[subsp]>/*.png
```

Any additional genus can be added later by dropping another directory under the data root with the same `{train,valid,test}/<species_[subsp]>/` structure — the notebook discovers genera at runtime via `GENERA = sorted(…images_root.iterdir()…)`.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate fastai
```

## Usage

### Training

```bash
jupyter lab Run_multilabel_fastai.ipynb
```

The notebook performs single-label pretraining, transfers weights into a multi-label head, fine-tunes on the combined dataset, evaluates on the held-out test set, and writes `test_results_<arch>.csv` with `Term`, `genus`, `taxonomic_level`, and per-class metrics. A per-genus and per-(genus, level) micro-averaged summary is also printed.

### Performance analysis

```bash
Rscript -e "rmarkdown::render('plot_precision_recall.Rmd')"
```

Produces per-genus faceted precision/recall, training-size, and F1 figures (`taxonomy_metrics.pdf` / `.png`).

### HuggingFace upload

```bash
python push_to_hf.py --dry-run    # inspect config without uploading
python push_to_hf.py              # upload to the default repo
```

## Key Features

- **Three-level hierarchy** (genus / species / subspecies), predicted jointly.
- **Weighted sampling**: square-root inverse-frequency weighting addresses class imbalance.
- **Mixed precision, multi-GPU** (configured for 2 GPUs).
- **Modern vision backbone**: EVA-02 Large / ResNeXt options.

## Model Files

Trained models are saved in `exported_fastai_models/`:
- `*_sl.pkl`: single-label pretrained models
- `*_ml.pkl`: final multi-label models

## Notes

- The project directory is still named `Cicindela_id/` for git-history continuity. Rename the parent folder (and `Beetle_taxon_id.Rproj`) locally if you prefer the updated name — all paths inside the repo are relative.
- Hybrid-specimen folders (`*_x_*`) were canonicalized on disk so the same folder name appears in train, valid, and test splits.

## License

See LICENSE file for details.
