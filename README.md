# CosPlace_XS

A geo-localization (visual place recognition) framework based on the [CosPlace](https://github.com/gmberton/CosPlace) method, adapted for use with the **San Francisco Extra-Small (SF-XS)** dataset. Given a query street-view image, the model retrieves the closest matching database image by geographic proximity.

> **Based on:** *"Rethinking Visual Geo-localization for Large-Scale Applications"* — Berton et al., CVPR 2022 [[paper](https://arxiv.org/pdf/2204.02287)] [[original repo](https://github.com/gmberton/CosPlace)]

---

## How It Works

CosPlace frames geo-localization as a classification problem over geographic cells. The training area is divided into a grid using UTM coordinates and heading angles. Images in the same cell form a class. To prevent data leakage between train and validation, classes are organized into **groups** with a guaranteed spatial gap between them — only one group is used per epoch.

Each group has its own **CosFace classifier** (large-margin cosine loss), which pushes descriptors from the same place closer together while separating different places with an angular margin. At inference, the model produces a compact L2-normalized descriptor per image, and retrieval is done via nearest-neighbor search over the database descriptors.

```
Query Image
    ↓
Backbone (ResNet / VGG16)
    ↓
L2Norm → GeM Pooling → Flatten → FC(→ 512-d) → L2Norm
    ↓
512-d descriptor
    ↓
FAISS k-NN search over database
    ↓
Top-K predictions (ranked by descriptor distance)
```

---

## Installation

```bash
git clone https://github.com/sam-khanjani/CosPlace_XS.git
cd CosPlace_XS
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.8, PyTorch ≥ 1.8.2, CUDA-capable GPU recommended.

---

## Dataset

This project uses the **SF-XL processed** dataset (a compact San Francisco street-view dataset). The dataset folder must contain three subfolders:

```
dataset_folder/
├── train/
├── val/
└── test/
    └── queries/
```

Set the dataset path via environment variable or pass it directly:

```bash
export SF_XL_PROCESSED_FOLDER=/path/to/sf_xl/processed
# or pass --dataset_folder /path/to/sf_xl/processed
```

---

## Training

```bash
python train.py \
  --dataset_folder /path/to/sf_xl/processed \
  --backbone ResNet50 \
  --fc_output_dim 512 \
  --batch_size 32 \
  --epochs_num 50
```

Logs and checkpoints are saved under `logs/<save_dir>/<timestamp>/`.

### Resume training from a checkpoint

```bash
python train.py \
  --dataset_folder /path/to/sf_xl/processed \
  --resume_train logs/default/<timestamp>/last_checkpoint.pth
```

### Load pretrained weights and fine-tune

```bash
python train.py \
  --dataset_folder /path/to/sf_xl/processed \
  --resume_model logs/default/<timestamp>/best_model.pth
```

---

## Evaluation

Run evaluation only (no training) on the test set:

```bash
python eval.py \
  --dataset_folder /path/to/sf_xl/processed \
  --resume_model logs/default/<timestamp>/best_model.pth
```

Metrics reported: **Recall@1, Recall@5, Recall@10, Recall@20** — a prediction is counted correct if the retrieved image is within `--positive_dist_threshold` meters (default 25 m) of the query.

### Save prediction visualizations

```bash
python train.py \
  --dataset_folder /path/to/sf_xl/processed \
  --num_preds_to_save 5
```

Add `--save_only_wrong_preds` to save only queries where the top-1 prediction was wrong. Outputs are saved as image grids in `<output_folder>/preds/` (green border = correct, red = incorrect).

---

## Changes from the Original CosPlace

This repo is a focused adaptation of [gmberton/CosPlace](https://github.com/gmberton/CosPlace). The changes below are the meaningful differences — everything else (loss, retrieval, checkpointing, logging, visualization) is unchanged.

### Added

**Two new GPU augmentation classes** (`augmentations.py`):

- `DeviceAgnosticRandomRotation` — applies a random rotation of ±`rotation_degrees` to each image in the batch on the GPU.
- `DeviceAgnosticCutout` — randomly masks a square patch of `cutout_size × cutout_size` pixels (applied with probability 0.5). Helps prevent the model from over-relying on small discriminative regions.

Both are inserted into the GPU augmentation pipeline in `train.py` alongside the original ColorJitter and RandomResizedCrop.

**Standalone `eval.py`**: the original repo has no dedicated evaluation script — testing is only triggered at the end of `train.py`. The new `eval.py` lets you run test-set evaluation on any saved checkpoint without touching training.

**New arguments**:

| Argument | Default | Description |
|---|---|---|
| `--rotation_degrees` | `10` | Max rotation angle (degrees) for random rotation augmentation |
| `--cutout_size` | `16` | Side length (pixels) of the cutout mask |
| `--classifiers_wd` | `0.001` | Weight decay for classifier optimizers |
| `--lambd` | `0.001` | Lambda for model ASGD optimizer (alternative, commented out) |
| `--classifiers_lambd` | `0.001` | Lambda for classifier ASGD optimizer (alternative, commented out) |

**Environment variable fallback**: `--dataset_folder` falls back to `$SF_XL_PROCESSED_FOLDER` automatically. The original requires the argument to be passed explicitly.

### Removed / Not Ported

- **EfficientNet backbones** — the original supports B0–B7. Only ResNet18/50/101/152 and VGG16 are kept here.
- **`--train_all_layers`** — upstream flag to unfreeze the full backbone; not present here.
- **`--image_size` / `--resize_test_imgs`** — upstream options to resize test images; not present here.

### Changed Defaults

| Parameter | CosPlace_XS | Original |
|---|---|---|
| `--brightness` | 0.8 | 0.7 |
| `--contrast` | 0.3 | 0.7 |
| `--hue` | 0.0 | 0.5 |
| `--classifiers_lr` | 0.001 | 0.01 |

---

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--backbone` | `ResNet50` | CNN backbone: `VGG16`, `ResNet18/50/101/152` |
| `--fc_output_dim` | `512` | Descriptor dimensionality |
| `--M` | `10` | Geographic cell size in meters |
| `--alpha` | `30` | Heading class size in degrees |
| `--N` | `5` | Spatial gap between groups (in M-units) |
| `--L` | `2` | Heading gap between groups (in alpha-units) |
| `--groups_num` | `8` | Number of geographic training groups |
| `--min_images_per_class` | `10` | Minimum images for a class to be included |
| `--batch_size` | `32` | Training batch size |
| `--epochs_num` | `50` | Number of training epochs |
| `--lr` | `1e-5` | Model learning rate |
| `--classifiers_lr` | `1e-3` | Classifier learning rate |
| `--positive_dist_threshold` | `25` | Distance (meters) for a correct retrieval |
| `--use_amp16` | off | Enable automatic mixed precision (AMP) |
| `--augmentation_device` | `cuda` | Run augmentation on `cuda` or `cpu` |

---

## Project Structure

```
CosPlace_XS/
├── train.py                    # Main training loop
├── eval.py                     # Standalone evaluation script
├── test.py                     # Recall computation (FAISS k-NN retrieval)
├── parser.py                   # Argument parsing
├── commons.py                  # Logging, reproducibility utilities
├── util.py                     # Checkpoint save / resume
├── cosface_loss.py             # CosFace (large-margin cosine) loss
├── augmentations.py            # GPU-accelerated data augmentation
├── visualizations.py           # Prediction visualization
├── hubconf.py                  # PyTorch Hub interface
├── requirements.txt
├── cosplace_model/
│   ├── cosplace_network.py     # GeoLocalizationNet (backbone + GeM aggregation)
│   └── layers.py               # GeM, L2Norm, Flatten layers
├── datasets/
│   ├── train_dataset.py        # Geographic grouping and class assignment
│   └── test_dataset.py         # Database/query loading and positive matching
└── cache/                      # Cached dataset metadata
```

---

## Model Architecture

**Backbone** (ResNet50 default): For ResNets, layers before `layer3` are frozen; `layer3` and `layer4` are trained. For VGG16, only the last 5 layers are trained.

**Aggregation head:**
1. L2 normalization of feature maps
2. GeM pooling (learnable exponent `p`, initialized to 3)
3. Flatten
4. Linear projection to `fc_output_dim` dimensions
5. L2 normalization → final descriptor

**Loss:** CosFace (`MarginCosineProduct`) with scale `s=30` and margin `m=0.40`.

---

## Citation

If you use this code, please also cite the original CosPlace paper:

```bibtex
@inProceedings{CosPlace2022,
  author    = {Berton, Gabriele and Mereu, Riccardo and Trivigno, Gabriele and Masone, Carlo
               and Csurka, Gabriela and Sattler, Torsten and Caputo, Barbara},
  title     = {Rethinking Visual Geo-localization for Large-Scale Applications},
  booktitle = {CVPR},
  year      = {2022},
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
