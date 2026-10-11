"""
R1-6: Training / Inference Time Benchmark (All 11 Models)
Measures per-epoch training time and per-sample inference time for all model
variants listed in the README model table.

Models tested:
    Polar (θ,φ):
        SVD:  w30, w45, w62, w89, w267  (input: 1×(n_lambda×2)×361)
        Raw:  w3, w30, w89              (input: n_lambda×181×361)
    BFP (x,y):
        SVD:  w30                       (input: 1×60×4000)
        Raw:  w3 (no log1p), w3 (log1p) (input: 3×4000×3000)

No pre-trained checkpoints required — models are initialized randomly.
Only average timing is needed, so 10 timed epochs (after 1 warmup) suffice.

Usage (on RTX 3090 machine):
    python timing_benchmark_v1.py \
        --polar-data D:\\FDTD_SVD\\TSV_diffraction_thetaphi_norm_svd_20260311 \
        --bfp-data D:\\FDTD_SVD\\TSV_diffraction_norm_svd_20260306

    # Test only Polar models:
    python timing_benchmark_v1.py --polar-data /path/to/polar

    # Test only BFP models:
    python timing_benchmark_v1.py --bfp-data /path/to/bfp

    # Test specific models:
    python timing_benchmark_v1.py \
        --polar-data /path/to/polar \
        --bfp-data /path/to/bfp \
        --models SVD_w89,Raw_w89,BFP_SVD_w30

Outputs:
    timing_benchmark_results.csv  (in same directory as this script)
    Console summary table

CSV columns:
    model, category, n_lambda, input_shape, params,
    train_time_s, train_std_s,
    inference_ms, inference_std_ms
"""
import os, sys, time, signal, argparse
from os.path import join, abspath
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'))

# Polar (θ,φ) models
from Model_SVD_w30_p1_2 import DUVModel as SVDModel_w30
from Model_SVD_w45_p1_1 import DUVModel as SVDModel_w45
from Model_SVD_w62_p1_1 import DUVModel as SVDModel_w62
from Model_SVD_w89_p1_2 import DUVModel as SVDModel_w89
from Model_SVD_w267_p1_2 import DUVModel as SVDModel_w267
from Model_raw3_3 import DUVModel as RawModel_w3
from Model_raw30_1 import DUVModel as RawModel_w30
from Model_raw89_1 import DUVModel as RawModel_w89

# BFP (x,y) models
from Model_BFP_raw3_1 import DUVModel as BFP_RawModel_w3
from Model_BFP_SVD_w30_p1_3 import DUVModel as BFP_SVDModel_w30
from Model_BFP_raw3_3 import DUVModel as BFP_RawModel_w3_log1p

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

OCD_MIN = np.array([0.02, 0.27, 0.2, 100.0, 2.7, 0.05])
OCD_MAX = np.array([0.1, 0.33, 0.27, 600.0, 3.3, 0.1])
HDF5_EXT = '.hdf5_norm_E2_SVD'

# ===================================================================================================================
# Model configuration
# ===================================================================================================================
# Polar (θ,φ): 801 wavelengths, 181×361 angular grid, key='norm_E2_angle'
# BFP (x,y):   30 wavelengths, 3000×4000 spatial grid, key='norm_E2'

WAVELENGTH_CONFIG = {
    # Polar SVD models
    'SVD_w30':  {'n_lambda': 30,  'step': 27,  'type': 'svd',   'category': 'polar'},
    'SVD_w45':  {'n_lambda': 45,  'step': 18,  'type': 'svd',   'category': 'polar'},
    'SVD_w62':  {'n_lambda': 62,  'step': 13,  'type': 'svd',   'category': 'polar'},
    'SVD_w89':  {'n_lambda': 89,  'step': 9,   'type': 'svd',   'category': 'polar'},
    'SVD_w267': {'n_lambda': 267, 'step': 3,   'type': 'svd',   'category': 'polar'},
    # Polar Raw models
    'Raw_w3':   {'n_lambda': 3,   'step': 9,   'type': 'raw',   'category': 'polar'},
    'Raw_w30':  {'n_lambda': 30,  'step': 27,  'type': 'raw',   'category': 'polar'},
    'Raw_w89':  {'n_lambda': 89,  'step': 9,   'type': 'raw',   'category': 'polar'},
    # BFP SVD models
    'BFP_SVD_w30': {'n_lambda': 30, 'step': 1,  'type': 'bfp_svd', 'category': 'bfp'},
    # BFP Raw models
    'BFP_Raw_w3':      {'n_lambda': 3, 'step': 1, 'type': 'bfp_raw',     'category': 'bfp'},
    'BFP_Raw_w3_log1p':{'n_lambda': 3, 'step': 1, 'type': 'bfp_raw_log1p','category': 'bfp'},
}

# Learning rates matching each trainer
LEARNING_RATES = {
    'SVD_w30': 5e-6, 'SVD_w45': 1e-5, 'SVD_w62': 1e-5,
    'SVD_w89': 5e-6, 'SVD_w267': 1e-5,
    'Raw_w3': 5e-6,  'Raw_w30': 5e-6, 'Raw_w89': 5e-6,
    'BFP_SVD_w30': 1e-5,
    'BFP_Raw_w3': 1e-6, 'BFP_Raw_w3_log1p': 1e-6,
}

# Batch sizes matching each trainer
BATCH_SIZES = {
    'SVD_w30': 32, 'SVD_w45': 32, 'SVD_w62': 32,
    'SVD_w89': 32, 'SVD_w267': 32,
    'Raw_w3': 32,  'Raw_w30': 32, 'Raw_w89': 32,
    'BFP_SVD_w30': 32,
    'BFP_Raw_w3': 4, 'BFP_Raw_w3_log1p': 4,
}

# Model constructors
MODEL_CLASSES = {
    'SVD_w30':  SVDModel_w30,
    'SVD_w45':  SVDModel_w45,
    'SVD_w62':  SVDModel_w62,
    'SVD_w89':  SVDModel_w89,
    'SVD_w267': SVDModel_w267,
    'Raw_w3':   RawModel_w3,
    'Raw_w30':  RawModel_w30,
    'Raw_w89':  RawModel_w89,
    'BFP_SVD_w30': BFP_SVDModel_w30,
    'BFP_Raw_w3': BFP_RawModel_w3,
    'BFP_Raw_w3_log1p': BFP_RawModel_w3_log1p,
}


def find_files(directory):
    """Recursively find all HDF5 data files in a directory."""
    match = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith(HDF5_EXT):
                match.append(os.path.join(root, f))
    return sorted(match)


# ─── Polar SVD Dataset ─────────────────────────────────────────────────────
class PolarSVDDataset(Dataset):
    """Loads Polar SVD features (Us, Vt) with configurable wavelength step.

    Data: 801 wavelengths, 181×361 angular grid.
    Input: (1, n_lambda*2, 361) = [pad(Us); Vt]
    """

    def __init__(self, data, step=9, topN=1):
        self.data = data
        self.step = step
        self.topN = topN
        self.ocd_norm_div = OCD_MAX - OCD_MIN

    def __len__(self):
        return len(self.data)

    def denormalize_ocd(self, ocd):
        return ocd * self.ocd_norm_div + OCD_MIN

    def __getitem__(self, index):
        with h5py.File(self.data[index], "r") as f:
            Us = f['Us'][::self.step, :, :self.topN]
            Vt = f['Vt'][::self.step, :self.topN]
            retch, Rtop, Rbot = f['retch'][()], f['Rtop'][()], f['Rbot'][()]
            n, height, b = f['n'][()], f['height'][()], f['b'][()]
        Us = np.squeeze(Us)
        Vt = np.squeeze(Vt)
        _, n_row = Us.shape
        _, n_col = Vt.shape
        pad_Us = np.pad(Us, ((0, 0), (0, n_col - n_row)), mode='constant', constant_values=0)
        images = np.expand_dims(np.concatenate((pad_Us, Vt), axis=0), axis=0)
        ocd = np.array([retch, Rtop, Rbot, n, height, b])
        ocd = (ocd - OCD_MIN) / self.ocd_norm_div
        return torch.tensor(images, dtype=torch.float32), torch.tensor(ocd, dtype=torch.float32)


# ─── Polar Raw Dataset ─────────────────────────────────────────────────────
class PolarRawDataset(Dataset):
    """Loads Polar raw far-field intensity with configurable wavelength step.

    Data: 801 wavelengths, 181×361 angular grid.
    Input: (n_lambda, 181, 361) = norm_E2_angle[::step]
    """

    def __init__(self, data, step=9):
        self.data = data
        self.step = step
        self.ocd_norm_div = OCD_MAX - OCD_MIN

    def __len__(self):
        return len(self.data)

    def denormalize_ocd(self, ocd):
        return ocd * self.ocd_norm_div + OCD_MIN

    def __getitem__(self, index):
        with h5py.File(self.data[index], "r") as f:
            images = f['norm_E2_angle'][::self.step]
            retch, Rtop, Rbot = f['retch'][()], f['Rtop'][()], f['Rbot'][()]
            n, height, b = f['n'][()], f['height'][()], f['b'][()]
        ocd = np.array([retch, Rtop, Rbot, n, height, b])
        ocd = (ocd - OCD_MIN) / self.ocd_norm_div
        return torch.tensor(images, dtype=torch.float32), torch.tensor(ocd, dtype=torch.float32)


# ─── Polar Raw_w3: 3 specific slices (first, mid, last) ───────────────────
class PolarRaw3Dataset(Dataset):
    """Loads 3 wavelength slices: first, middle, last of the 89 selected."""

    def __init__(self, data, step=9):
        self.data = data
        self.step = step
        self.ocd_norm_div = OCD_MAX - OCD_MIN

    def __len__(self):
        return len(self.data)

    def denormalize_ocd(self, ocd):
        return ocd * self.ocd_norm_div + OCD_MIN

    def __getitem__(self, index):
        with h5py.File(self.data[index], "r") as f:
            images = f['norm_E2_angle'][::self.step]  # (89, 181, 361)
            retch, Rtop, Rbot = f['retch'][()], f['Rtop'][()], f['Rbot'][()]
            n, height, b = f['n'][()], f['height'][()], f['b'][()]
        # Select first, middle, last
        images = np.stack([images[0], images[len(images) // 2], images[-1]])
        ocd = np.array([retch, Rtop, Rbot, n, height, b])
        ocd = (ocd - OCD_MIN) / self.ocd_norm_div
        return torch.tensor(images, dtype=torch.float32), torch.tensor(ocd, dtype=torch.float32)


# ─── BFP SVD Dataset ───────────────────────────────────────────────────────
class BFPSVDDataset(Dataset):
    """Loads BFP SVD features (Us, Vt) — all 30 wavelengths.

    Data: 30 wavelengths, 3000×4000 spatial grid.
    Us (30, 3000) → padded to (30, 4000), Vt (30, 4000).
    Input: (1, 60, 4000) = [pad(Us); Vt]
    """

    def __init__(self, data, topN=1):
        self.data = data
        self.topN = topN
        self.ocd_norm_div = OCD_MAX - OCD_MIN

    def __len__(self):
        return len(self.data)

    def denormalize_ocd(self, ocd):
        return ocd * self.ocd_norm_div + OCD_MIN

    def __getitem__(self, index):
        with h5py.File(self.data[index], "r") as f:
            Us = f['Us'][()][:, :, :self.topN]
            Vt = f['Vt'][()][:, :self.topN]
            retch, Rtop, Rbot = f['retch'][()], f['Rtop'][()], f['Rbot'][()]
            n, height, b = f['n'][()], f['height'][()], f['b'][()]
        Us = np.squeeze(Us)  # (30, 3000)
        Vt = np.squeeze(Vt)  # (30, 4000)
        # Pad Us from (30, 3000) to (30, 4000)
        pad_Us = np.pad(Us, ((0, 0), (0, 1000)), mode='constant', constant_values=0)
        images = np.expand_dims(np.concatenate((pad_Us, Vt), axis=0), axis=0)  # (1, 60, 4000)
        ocd = np.array([retch, Rtop, Rbot, n, height, b])
        ocd = (ocd - OCD_MIN) / self.ocd_norm_div
        return torch.tensor(images, dtype=torch.float32), torch.tensor(ocd, dtype=torch.float32)


# ─── BFP Raw Dataset (no log1p) ────────────────────────────────────────────
class BFPRawDataset(Dataset):
    """Loads BFP raw 3-wavelength slices (no log1p).

    Selects indices [0, 14, 29] from 30 wavelengths.
    Input: (3, 4000, 3000)
    """

    def __init__(self, data):
        self.data = data
        self.ocd_norm_div = OCD_MAX - OCD_MIN

    def __len__(self):
        return len(self.data)

    def denormalize_ocd(self, ocd):
        return ocd * self.ocd_norm_div + OCD_MIN

    def __getitem__(self, index):
        with h5py.File(self.data[index], "r") as f:
            images = f['norm_E2']
            # Select first, mid, last from 30 wavelengths
            image_front = images[0]
            image_mid = images[14]
            image_end = images[29]
            images = np.stack([image_front, image_mid, image_end])
            retch, Rtop, Rbot = f['retch'][()], f['Rtop'][()], f['Rbot'][()]
            n, height, b = f['n'][()], f['height'][()], f['b'][()]
        ocd = np.array([retch, Rtop, Rbot, n, height, b])
        ocd = (ocd - OCD_MIN) / self.ocd_norm_div
        return torch.tensor(images, dtype=torch.float32), torch.tensor(ocd, dtype=torch.float32)


# ─── BFP Raw Dataset (with log1p) ──────────────────────────────────────────
class BFPRawLog1pDataset(Dataset):
    """Loads BFP raw 3-wavelength slices with log1p transformation.

    Selects indices [0, 15, 29] from 30 wavelengths.
    Applies np.log1p() to input images.
    Input: (3, 4000, 3000)
    """

    def __init__(self, data):
        self.data = data
        self.ocd_norm_div = OCD_MAX - OCD_MIN

    def __len__(self):
        return len(self.data)

    def denormalize_ocd(self, ocd):
        return ocd * self.ocd_norm_div + OCD_MIN

    def __getitem__(self, index):
        with h5py.File(self.data[index], "r") as f:
            images = f['norm_E2']
            # Select first, mid, last from 30 wavelengths
            image_front = images[0]
            image_mid = images[len(images) // 2]
            image_end = images[-1]
            images = np.stack([image_front, image_mid, image_end])
            # Apply log1p transformation
            images = np.log1p(images)
            retch, Rtop, Rbot = f['retch'][()], f['Rtop'][()], f['Rbot'][()]
            n, height, b = f['n'][()], f['height'][()], f['b'][()]
        ocd = np.array([retch, Rtop, Rbot, n, height, b])
        ocd = (ocd - OCD_MIN) / self.ocd_norm_div
        return torch.tensor(images, dtype=torch.float32), torch.tensor(ocd, dtype=torch.float32)


def count_params(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_input_shape(model_name):
    """Return the input shape string for a model."""
    cfg = WAVELENGTH_CONFIG[model_name]
    n = cfg['n_lambda']
    if cfg['type'] == 'svd':
        return f"1x{n*2}x361"
    elif cfg['type'] == 'bfp_svd':
        return f"1x{n*2}x4000"
    elif cfg['type'] in ('raw', 'bfp_raw', 'bfp_raw_log1p'):
        if cfg['category'] == 'polar':
            return f"{n}x181x361"
        else:
            return f"{n}x4000x3000"
    return "unknown"


def benchmark_training(model, dataloader, optimizer, device, n_epochs=10, warmup=1):
    """Measure per-epoch training time. Returns list of epoch times (seconds)."""
    times = []
    for epoch in range(warmup + n_epochs):
        model.train()
        t0 = time.perf_counter()
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            pred = model(images)
            loss = F.l1_loss(pred, targets, reduction='mean')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        if epoch >= warmup:
            times.append(elapsed)
    return times


def benchmark_inference(model, dataloader, device, n_repeats=5, warmup=1):
    """Measure per-sample inference time. Returns list of per-sample times (seconds)."""
    model.eval()
    results = []
    for rep in range(warmup + n_repeats):
        total_time = 0.0
        n_samples = 0
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(device, non_blocking=True)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(images)
                torch.cuda.synchronize()
                total_time += time.perf_counter() - t0
                n_samples += images.shape[0]
        if rep >= warmup:
            results.append(total_time / n_samples)
    return results


def build_model(model_name, device):
    """Instantiate a model with the correct constructor arguments."""
    cfg = WAVELENGTH_CONFIG[model_name]
    n = cfg['n_lambda']
    cls = MODEL_CLASSES[model_name]

    if cfg['type'] == 'svd':
        # Polar SVD: DUVModel(n_lambda=n)
        model = cls(n_lambda=n)
    elif cfg['type'] == 'bfp_svd':
        # BFP SVD: DUVModel(n_lambda=n)
        model = cls(n_lambda=n)
    elif cfg['type'] == 'raw':
        # Polar Raw: DUVModel(input_shape=(n, 181, 361))
        model = cls(input_shape=(n, 181, 361))
    elif cfg['type'] in ('bfp_raw', 'bfp_raw_log1p'):
        # BFP Raw: DUVModel(input_shape=(3, 4000, 3000))
        model = cls(input_shape=(3, 4000, 3000))
    else:
        raise ValueError(f"Unknown model type: {cfg['type']}")

    return model.to(device)


def build_dataset(model_name, files):
    """Build the appropriate dataset for a model."""
    cfg = WAVELENGTH_CONFIG[model_name]
    step = cfg['step']

    if model_name == 'Raw_w3':
        return PolarRaw3Dataset(files, step=step)
    elif cfg['type'] == 'svd':
        return PolarSVDDataset(files, step=step)
    elif cfg['type'] == 'bfp_svd':
        return BFPSVDDataset(files)
    elif cfg['type'] == 'bfp_raw':
        return BFPRawDataset(files)
    elif cfg['type'] == 'bfp_raw_log1p':
        return BFPRawLog1pDataset(files)
    else:
        return PolarRawDataset(files, step=step)


def save_csv(results, out_file):
    """Write results to CSV (called after each model to survive SSH disconnect)."""
    with open(out_file, 'w') as f:
        f.write("model,category,n_lambda,input_shape,params,train_time_s,train_std_s,inference_ms,inference_std_ms\n")
        for r in results:
            f.write(f"{r['model']},{r['category']},{r['n_lambda']},{r['input_shape']},{r['params']},"
                    f"{r['train_time_s']:.4f},{r['train_std_s']:.4f},"
                    f"{r['inference_ms']:.4f},{r['inference_std_ms']:.4f}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Training/Inference time benchmark (all 11 models: Polar + BFP)')
    parser.add_argument('--polar-data', type=str, default=None,
                        help='Polar (θ,φ) data root directory (contains training/ and validation/)')
    parser.add_argument('--bfp-data', type=str, default=None,
                        help='BFP (x,y) data root directory (contains training/ and validation/)')
    parser.add_argument('--n-epochs', type=int, default=10,
                        help='Number of timed training epochs (after 1 warmup) [default: 10]')
    parser.add_argument('--n-repeats', type=int, default=5,
                        help='Number of inference timing repeats [default: 5]')
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated list of models to test (default: all available)')
    args = parser.parse_args()

    # Validate that at least one data directory is provided
    if args.polar_data is None and args.bfp_data is None:
        parser.error("At least one of --polar-data or --bfp-data must be specified.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load data files for each category
    polar_train_files = []
    polar_val_files = []
    bfp_train_files = []
    bfp_val_files = []

    if args.polar_data:
        polar_root = abspath(args.polar_data)
        polar_train_files = find_files(join(polar_root, 'training'))
        polar_val_files = find_files(join(polar_root, 'validation'))
        print(f"Polar data: {len(polar_train_files)} training, {len(polar_val_files)} validation files")

    if args.bfp_data:
        bfp_root = abspath(args.bfp_data)
        bfp_train_files = find_files(join(bfp_root, 'training'))
        bfp_val_files = find_files(join(bfp_root, 'validation'))
        print(f"BFP data: {len(bfp_train_files)} training, {len(bfp_val_files)} validation files")

    # Determine which models to test
    if args.models:
        model_names = [m.strip() for m in args.models.split(',')]
    else:
        # Default: all models whose data is available
        model_names = []
        for name, cfg in WAVELENGTH_CONFIG.items():
            if cfg['category'] == 'polar' and polar_train_files:
                model_names.append(name)
            elif cfg['category'] == 'bfp' and bfp_train_files:
                model_names.append(name)

    # Filter out models whose data is not available
    available_models = []
    for name in model_names:
        cfg = WAVELENGTH_CONFIG[name]
        if cfg['category'] == 'polar' and not polar_train_files:
            print(f"  [SKIP] {name}: Polar data not provided")
        elif cfg['category'] == 'bfp' and not bfp_train_files:
            print(f"  [SKIP] {name}: BFP data not provided")
        else:
            available_models.append(name)

    print(f"Models to test: {available_models}")
    print(f"Timed epochs: {args.n_epochs} (+ 1 warmup)")
    print(f"Inference repeats: {args.n_repeats} (+ 1 warmup)")

    # Results storage
    results = []
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_file = join(out_dir, 'timing_benchmark_results.csv')

    # Load existing results (skip already-completed models)
    existing = {}
    if os.path.exists(out_file):
        with open(out_file, 'r') as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 9:
                    existing[parts[0]] = {
                        'model': parts[0], 'category': parts[1],
                        'n_lambda': int(parts[2]), 'input_shape': parts[3],
                        'params': int(parts[4]), 'train_time_s': float(parts[5]),
                        'train_std_s': float(parts[6]),
                        'inference_ms': float(parts[7]), 'inference_std_ms': float(parts[8]),
                    }
        if existing:
            print(f"Found {len(existing)} existing result(s) in CSV — will skip those models.")

    # Signal handler: save partial results on SIGINT/SIGTERM (SSH disconnect)
    def _signal_handler(signum, frame):
        print(f"\n[INTERRUPTED] Signal {signum} received. Saving {len(results)} completed model(s) to CSV...")
        if results:
            save_csv(results, out_file)
            print(f"Partial results saved to: {out_file}")
        sys.exit(1)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    for model_name in available_models:
        cfg = WAVELENGTH_CONFIG[model_name]
        batch_size = BATCH_SIZES[model_name]

        # Skip if already in existing results
        if model_name in existing:
            results.append(existing[model_name])
            r = existing[model_name]
            print(f"\n  [SKIP] {model_name} — already in CSV "
                  f"(train: {r['train_time_s']:.2f}s, infer: {r['inference_ms']:.3f}ms)")
            continue

        print(f"\n{'='*70}")
        print(f"  {model_name}  [{cfg['category'].upper()}]  (input: {get_input_shape(model_name)})")
        print(f"{'='*70}")

        # Select data files based on category
        if cfg['category'] == 'polar':
            train_files = polar_train_files
            val_files = polar_val_files
        else:
            train_files = bfp_train_files
            val_files = bfp_val_files

        # Build dataset and dataloader
        train_ds = build_dataset(model_name, train_files)
        val_ds = build_dataset(model_name, val_files)

        # Reduce workers on Windows
        num_workers = 1 if os.name == 'nt' else 4
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # Build model (random init, no checkpoint needed)
        model = build_model(model_name, device)
        params = count_params(model)
        print(f"  Parameters: {params:,}")
        print(f"  Batch size: {batch_size}")

        # Training benchmark
        lr = LEARNING_RATES[model_name]
        optimizer = torch.optim.NAdam(model.parameters(), lr=lr,
                                      weight_decay=1e-4, decoupled_weight_decay=True)
        print(f"  Training ({args.n_epochs} epochs, lr={lr:.0e})...")
        train_times = benchmark_training(model, train_dl, optimizer, device,
                                         n_epochs=args.n_epochs, warmup=1)
        train_mean = np.mean(train_times)
        train_std = np.std(train_times)
        print(f"  Train: {train_mean:.2f} ± {train_std:.2f} s/epoch")

        # Inference benchmark
        print(f"  Inference ({args.n_repeats} repeats)...")
        inf_times = benchmark_inference(model, val_dl, device,
                                        n_repeats=args.n_repeats, warmup=1)
        inf_mean = np.mean(inf_times) * 1000  # ms
        inf_std = np.std(inf_times) * 1000
        print(f"  Inference: {inf_mean:.3f} ± {inf_std:.3f} ms/sample")

        results.append({
            'model': model_name,
            'category': cfg['category'],
            'n_lambda': cfg['n_lambda'],
            'input_shape': get_input_shape(model_name),
            'params': params,
            'train_time_s': train_mean,
            'train_std_s': train_std,
            'inference_ms': inf_mean,
            'inference_std_ms': inf_std,
        })

        # Free GPU memory
        del model, optimizer, train_dl, val_dl, train_ds, val_ds
        torch.cuda.empty_cache()

        # Save CSV after each model (survives SSH disconnect)
        save_csv(results, out_file)
        print(f"  [saved {len(results)}/{len(available_models)} models to CSV]")

    # ── Summary table ──
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    header = f"{'Model':<20} | {'Cat':<6} | {'n_λ':>4} | {'Input':<12} | {'Params':>10} | {'Train (s/ep)':>14} | {'Infer (ms/s)':>14}"
    print(header)
    print('-' * len(header))
    for r in results:
        print(f"{r['model']:<20} | {r['category']:<6} | {r['n_lambda']:>4} | {r['input_shape']:<12} | "
              f"{r['params']:>10,} | {r['train_time_s']:>7.2f}±{r['train_std_s']:.2f} | "
              f"{r['inference_ms']:>7.3f}±{r['inference_std_ms']:.3f}")

    # Speedup ratios (Polar: SVD_w89 vs Raw_w89)
    svd89 = next((r for r in results if r['model'] == 'SVD_w89'), None)
    raw89 = next((r for r in results if r['model'] == 'Raw_w89'), None)
    if svd89 and raw89:
        print(f"\nSpeedup Polar SVD_w89 vs Raw_w89:")
        print(f"  Training:  {raw89['train_time_s']/svd89['train_time_s']:.2f}x")
        print(f"  Inference: {raw89['inference_ms']/svd89['inference_ms']:.2f}x")

    # ── Final CSV save (already saved per-model, this is the final confirmation) ──
    save_csv(results, out_file)
    print(f"\nResults saved to: {out_file}")


if __name__ == '__main__':
    main()