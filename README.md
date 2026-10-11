# SVD-Based Far-Field Diffraction Metrology for TSV OCD

Open-source code and demo data for:
> "Singular Value Decomposition of Far-Field Electromagnetic Data for Compressing and Accelerating Deep Neural Networks in the Prediction of Geometric Parameters for Through Silicon Via Array"
> IEEE Transactions on Instrumentation and Measurement (MEAS_D_2613346)

## Repository Structure

```
open_source/
├── README.md
├── requirements.txt
├── data/
│   └── demo/                    # 50-geometry demo subset (89 wavelengths)
│       ├── training/            # ~40 training HDF5 files
│       └── validation/          # ~10 validation HDF5 files
├── models/
│   ├── ResNet.py                  # FeatureNet18 backbone (shared by all models)
│   ├── ── Polar (θ,φ) models ──
│   ├── Model_SVD_w30_p1_2.py    # Polar SVD, 30 wavelengths (input: 1×60×361)
│   ├── Model_SVD_w45_p1_1.py    # Polar SVD, 45 wavelengths (input: 1×90×361)
│   ├── Model_SVD_w62_p1_1.py    # Polar SVD, 62 wavelengths (input: 1×124×361)
│   ├── Model_SVD_w89_p1_2.py    # Polar SVD, 89 wavelengths (input: 1×178×361) [primary]
│   ├── Model_SVD_w267_p1_2.py   # Polar SVD, 267 wavelengths (input: 1×534×361)
│   ├── Model_raw3_3.py          # Polar Raw, 3 wavelengths (input: 3×181×361)
│   ├── Model_raw30_1.py         # Polar Raw, 30 wavelengths (input: 30×181×361)
│   ├── Model_raw89_1.py         # Polar Raw, 89 wavelengths (input: 89×181×361) [baseline]
│   ├── ── BFP (x,y) models ──
│   ├── Model_BFP_raw3_1.py      # BFP Raw, 3 wavelengths (input: 3×4000×3000)
│   ├── Model_BFP_SVD_w30_p1_3.py# BFP SVD, 30 wavelengths (input: 1×60×4000)
│   └── Model_BFP_raw3_3.py      # BFP Raw, 3 wavelengths + log1p (input: 3×4000×3000)
├── training/
│   ├── ── Polar (θ,φ) trainers ──
│   ├── trainer_SVD_w30_p1_2.1_2.py   # Polar SVD 30λ (lr=5e-6)
│   ├── trainer_SVD_w45_p1_1.1_1.py   # Polar SVD 45λ (lr=1e-5)
│   ├── trainer_SVD_w62_p1_1.1_1.py   # Polar SVD 62λ (lr=1e-5)
│   ├── trainer_SVD_w89_p1_2.1_2.py   # Polar SVD 89λ (lr=5e-6) [primary]
│   ├── trainer_SVD_w267_p1_2.1_2.py  # Polar SVD 267λ (lr=1e-5)
│   ├── trainer_raw3_3.2_3.py         # Polar Raw 3λ (lr=5e-6)
│   ├── trainer_raw30_1.1_1.py        # Polar Raw 30λ (lr=5e-6)
│   ├── trainer_raw89_1.1_1.py        # Polar Raw 89λ (lr=5e-6) [baseline]
│   ├── ── BFP (x,y) trainers ──
│   ├── trainer_BFP_raw3_1.1_1.py     # BFP Raw 3λ (lr=1e-6, bs=4)
│   ├── trainer_BFP_SVD_w30_p1_3.1.2_3.py  # BFP SVD 30λ (lr=1e-5, bs=32)
│   └── trainer_BFP_raw3_2.1_3.py     # BFP Raw 3λ + log1p (lr=1e-6, bs=4)
├── evaluation/
│   ├── noise_robustness_v1.py         # Noise robustness experiment (Table 7)
│   └── timing_benchmark_v1.py         # Training/inference time benchmark (Polar + BFP)
└── scripts/
    └── prepare_demo_data.py           # Extract demo subset from full dataset
```

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.10 (CUDA 11.x recommended)
- NumPy, h5py, pandas, torchinfo

See `requirements.txt` for pinned versions.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare demo data

The demo dataset (50 geometries × 89 wavelengths) is included in `data/demo/`.
Each HDF5 file contains:
- `norm_E2_angle`: uint16 far-field intensity, shape (801, 181, 361)
- `Us`, `Vt`: pre-computed SVD features (top-1 singular vectors)
- `retch`, `Rtop`, `Rbot`, `n`, `height`, `b`: OCD ground truth

To extract the demo subset from the full dataset:

```bash
python scripts/prepare_demo_data.py -f /path/to/full_dataset -n 50
```

### 3. Train the SVD model (Polar)

```bash
python training/trainer_SVD_w89_p1_2.1_2.py -f data/demo
```

### 4. Train a BFP model

```bash
python training/trainer_BFP_SVD_w30_p1_3.1.2_3.py -f /path/to/BFP_dataset
```

### 5. Run noise robustness experiment

```bash
python evaluation/noise_robustness_v1.py \
    -f data/demo \
    --svd-ckpt /path/to/svd_checkpoint.pt \
    --raw-ckpt /path/to/raw_checkpoint.pt
```

### 6. Run timing benchmark (all 11 models)

```bash
python evaluation/timing_benchmark_v1.py \
    --polar-data /path/to/polar_dataset \
    --bfp-data /path/to/bfp_dataset
```

No pre-trained checkpoints required — models are randomly initialized.
Runs 10 timed epochs (+ 1 warmup) for training time and 5 repeats for inference time.
Output: `evaluation/timing_benchmark_results.csv`

To test a subset:
```bash
python evaluation/timing_benchmark_v1.py \
    --polar-data /path/to/polar_dataset \
    --bfp-data /path/to/bfp_dataset \
    --models SVD_w89,Raw_w89,BFP_SVD_w30
```

## Model Architecture

All models share the same backbone: **FeatureNet18** (ResNet-18) + **Encoder_OCD** head.

### Polar (θ,φ) SVD-compressed models

Data: 801 wavelengths, 181×361 angular grid. Key: `norm_E2_angle`.
Input: `(B, 1, n_lambda×2, 361)` = `[pad(Us); Vt]` concatenated along height.

| Model | n_lambda | Input Shape | Learning Rate |
|-------|----------|-------------|---------------|
| Model_SVD_w30_p1_2 | 30 | (B, 1, 60, 361) | 5e-6 |
| Model_SVD_w45_p1_1 | 45 | (B, 1, 90, 361) | 1e-5 |
| Model_SVD_w62_p1_1 | 62 | (B, 1, 124, 361) | 1e-5 |
| **Model_SVD_w89_p1_2** | **89** | **(B, 1, 178, 361)** | **5e-6** |
| Model_SVD_w267_p1_2 | 267 | (B, 1, 534, 361) | 1e-5 |

### Polar (θ,φ) Raw far-field models (baseline)

Input: `(B, n_lambda, 181, 361)` = `norm_E2_angle[::9]` directly.

| Model | n_lambda | Input Shape | Learning Rate |
|-------|----------|-------------|---------------|
| Model_raw3_3 | 3 | (B, 3, 181, 361) | 5e-6 |
| Model_raw30_1 | 30 | (B, 30, 181, 361) | 5e-6 |
| **Model_raw89_1** | **89** | **(B, 89, 181, 361)** | **5e-6** |

### BFP (x,y) SVD-compressed models

Data: 30 wavelengths, 3000×4000 spatial grid. Key: `norm_E2`.
Input: `(B, 1, n_lambda×2, 4000)` = `[pad(Us); Vt]` where Us (30, 3000) → padded to (30, 4000).

| Model | n_lambda | Input Shape | Learning Rate | Batch Size |
|-------|----------|-------------|---------------|------------|
| **Model_BFP_SVD_w30_p1_3** | **30** | **(B, 1, 60, 4000)** | **1e-5** | **32** |

### BFP (x,y) Raw far-field models

Input: `(B, 3, 4000, 3000)` = 3 selected wavelength slices from 30 total.

| Model | Wavelengths | Input Shape | Learning Rate | Batch Size | log1p |
|-------|-------------|-------------|---------------|------------|-------|
| Model_BFP_raw3_1 | [0, 14, 29] | (B, 3, 4000, 3000) | 1e-6 | 4 | No |
| Model_BFP_raw3_3 | [0, 15, 29] | (B, 3, 4000, 3000) | 1e-6 | 4 | Yes |

**Bold** = primary model (SVD) and baseline (Raw) used in the paper.

## Data Format

### Polar (θ,φ) dataset

Directory: `TSV_diffraction_thetaphi_norm_svd_20260311/`

Each HDF5 file (`.hdf5_norm_E2_SVD`) contains:

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `norm_E2_angle` | (801, 181, 361) | uint16 | Far-field intensity [0, 4095] |
| `Us` | (801, 181, 1) | float32 | Top-1 left singular vectors |
| `Vt` | (801, 1, 361) | float32 | Top-1 right singular vectors |
| `retch` | scalar | float64 | Reticle CD (μm) |
| `Rtop` | scalar | float64 | Top radius (μm) |
| `Rbot` | scalar | float64 | Bottom radius (μm) |
| `n` | scalar | float64 | Scallop count |
| `height` | scalar | float64 | TSV height (μm) |
| `b` | scalar | float64 | Bottom ellipsoid depth (μm) |

Wavelength selection: `[::9]` from 801 → 89 wavelengths.

### BFP (x,y) dataset

Directory: `TSV_diffraction_norm_svd_20260306/`

Each HDF5 file (`.hdf5_norm_E2_SVD`) contains:

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `norm_E2` | (30, 3000, 4000) | uint16 | Far-field intensity [0, 4095] |
| `Us` | (30, 3000, 1) | float32 | Top-1 left singular vectors |
| `Vt` | (30, 1, 4000) | float32 | Top-1 right singular vectors |
| `retch` | scalar | float64 | Reticle CD (μm) |
| `Rtop` | scalar | float64 | Top radius (μm) |
| `Rbot` | scalar | float64 | Bottom radius (μm) |
| `n` | scalar | float64 | Scallop count |
| `height` | scalar | float64 | TSV height (μm) |
| `b` | scalar | float64 | Bottom ellipsoid depth (μm) |

All 30 wavelengths used (no subsampling).

## License

[To be determined]

## Citation

If you use this code, please cite:

```
@article{chen2026svd,
  title={Singular Value Decomposition of Far-Field Electromagnetic Data for Compressing and Accelerating Deep Neural Networks in the Prediction of Geometric Parameters for Through Silicon Via Array},
  author={Chen, S.-E. and ...},
  journal={IEEE Transactions on Instrumentation and Measurement},
  year={2026}
}
```