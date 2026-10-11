"""
Phase 4: Noise Robustness Experiment
Injects Gaussian noise (σ=0.01, 0.05, 0.1) into uint16 far-field intensity [0, 4095],
re-computes SVD for the SVD model, and evaluates both models on validation data.

Data flow (per Plan_v2):
    norm_E2_angle (uint16 [0,4095])
        ↓ add noise: clip(E2 + randn×σ×4095, 0, 4095)
    Noisy E2_angle
        ↓ SVD (p=1): Us, Vt
    Noisy SVD features → SVD model inference
    Noisy E2_angle → Raw model inference

SVD model: Model_SVD_w89_p1_2, input (1, 178, 361) = [pad(Us); Vt]
Raw model: Model_raw89_1, input (89, 181, 361) = norm_E2_angle[::9]

Usage:
    python noise_robustness_v1.py -f D:\\FDTD_SVD\\TSV_diffraction_thetaphi_norm_svd_20260311 \
        --svd-ckpt CHKPT_SVD_w89_p1_2.1_2\\20260515-09-51-26\\model_ep-1567.pt \
        --raw-ckpt CHKPT_raw89_1.1_1\\20260524-20-52-22\\model_ep-1591.pt
"""
import os, sys, time, argparse
from os.path import join, abspath
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'))
from Model_SVD_w89_p1_2 import DUVModel as SVDModel
from Model_raw89_1 import DUVModel as RawModel

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

N_LAMBDA = 89
OCD_MIN = np.array([0.02, 0.27, 0.2, 100.0, 2.7, 0.05])
OCD_MAX = np.array([0.1, 0.33, 0.27, 600.0, 3.3, 0.1])
OCD_NAMES = ['retch', 'Rtop', 'Rbot', 'n', 'height', 'b']
NOISE_LEVELS = [0.0, 0.01, 0.05, 0.1]
BATCH_SIZE = 32
ADC_MAX = 4095.0


def find_val_files(root_dir):
    val_dir = join(root_dir, 'validation')
    files = []
    for f in sorted(os.listdir(val_dir)):
        if f.endswith('.hdf5_norm_E2_SVD'):
            files.append(join(val_dir, f))
    return files


def load_data(file_path):
    """Load norm_E2_angle (uint16) and OCD targets from SVD-processed HDF5."""
    with h5py.File(file_path, 'r') as f:
        e2 = f['norm_E2_angle'][::9].astype(np.float64)  # (89, 181, 361)
        retch = f['retch'][()]
        Rtop = f['Rtop'][()]
        Rbot = f['Rbot'][()]
        n = f['n'][()]
        height = f['height'][()]
        b = f['b'][()]
    ocd = np.array([retch, Rtop, Rbot, n, height, b])
    ocd_norm = (ocd - OCD_MIN) / (OCD_MAX - OCD_MIN)
    return e2, ocd_norm


def add_noise_uint16(e2, sigma, rng):
    """Add Gaussian noise to uint16-scale data [0, 4095].
    e2: (89, 181, 361) float64 in [0, 4095]
    Returns noisy data clipped to [0, 4095].
    """
    if sigma == 0.0:
        return e2.copy()
    noise = rng.normal(0, sigma * ADC_MAX, size=e2.shape)
    return np.clip(e2 + noise, 0, ADC_MAX)


def _svd_one(mat):
    """Compute top-1 singular vectors for a single (181, 361) matrix."""
    U, s, Vt = np.linalg.svd(mat, full_matrices=False)
    return U[:, 0], Vt[0, :]


def compute_svd_features(e2_noisy, n_workers=16):
    """Compute SVD features (Us, Vt) from noisy E2 data (parallelized).
    e2_noisy: (89, 181, 361)
    Returns:
        Us: (89, 181) — top-1 left singular vector per wavelength
        Vt: (89, 361) — top-1 right singular vector per wavelength
    """
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(_svd_one, e2_noisy))
    Us = np.array([r[0] for r in results])  # (89, 181)
    Vt = np.array([r[1] for r in results])  # (89, 361)
    return Us, Vt


def build_svd_input(Us, Vt):
    """Build SVD model input: pad Us to (89, 361), concatenate with Vt → (1, 178, 361)."""
    n_row = Us.shape[1]  # 181
    n_col = Vt.shape[1]  # 361
    pad_Us = np.pad(Us, ((0, 0), (0, n_col - n_row)), mode='constant', constant_values=0)
    images = np.concatenate((pad_Us, Vt), axis=0)  # (178, 361)
    return np.expand_dims(images, 0)  # (1, 178, 361)


def denormalize_ocd(ocd_norm):
    return ocd_norm * (OCD_MAX - OCD_MIN) + OCD_MIN


def evaluate_svd_model(model, e2_data, ocd_data, sigma, device, rng, batch_size=BATCH_SIZE):
    """SVD model: add noise → re-compute SVD → inference."""
    model.eval()
    pe_list = np.zeros((len(e2_data), 6))

    for i in range(0, len(e2_data), batch_size):
        batch_e2 = e2_data[i:i+batch_size]
        batch_ocd = ocd_data[i:i+batch_size]
        bs = len(batch_e2)

        images_list = []
        for j in range(bs):
            e2_noisy = add_noise_uint16(batch_e2[j], sigma, rng)
            Us, Vt = compute_svd_features(e2_noisy)
            images_list.append(build_svd_input(Us, Vt))

        images = torch.tensor(np.stack(images_list), dtype=torch.float32)  # (bs, 1, 178, 361)
        images = images.to(device)
        with torch.no_grad():
            pred = model(images)
            pred = pred.cpu().numpy()

        pred_ocd = denormalize_ocd(pred)
        true_ocd = denormalize_ocd(batch_ocd)
        pe = np.abs(pred_ocd - true_ocd) / np.abs(true_ocd) * 100.0
        pe_list[i:i+bs] = pe

    return pe_list.mean(axis=0), pe_list.std(axis=0)


def evaluate_raw_model(model, e2_data, ocd_data, sigma, device, rng, batch_size=BATCH_SIZE):
    """Raw model: add noise directly to norm_E2_angle → inference."""
    model.eval()
    pe_list = np.zeros((len(e2_data), 6))

    for i in range(0, len(e2_data), batch_size):
        batch_e2 = e2_data[i:i+batch_size]
        batch_ocd = ocd_data[i:i+batch_size]
        bs = len(batch_e2)

        noisy_batch = np.stack([add_noise_uint16(batch_e2[j], sigma, rng) for j in range(bs)])
        images = torch.tensor(noisy_batch, dtype=torch.float32)  # (bs, 89, 181, 361)
        images = images.to(device)
        with torch.no_grad():
            pred = model(images)
            pred = pred.cpu().numpy()

        pred_ocd = denormalize_ocd(pred)
        true_ocd = denormalize_ocd(batch_ocd)
        pe = np.abs(pred_ocd - true_ocd) / np.abs(true_ocd) * 100.0
        pe_list[i:i+bs] = pe

    return pe_list.mean(axis=0), pe_list.std(axis=0)


def main():
    parser = argparse.ArgumentParser(description='Noise robustness experiment')
    parser.add_argument('-f', '--file', required=True, help='Root data directory (contains validation/)')
    parser.add_argument('--svd-ckpt', required=True, help='SVD model checkpoint (Model_SVD_w89_p1_2)')
    parser.add_argument('--raw-ckpt', required=True, help='Raw model checkpoint (Model_raw89_1)')
    parser.add_argument('--n-trials', type=int, default=5, help='Number of noise trials per sigma')
    args = parser.parse_args()

    root_dir = abspath(args.file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load validation data
    val_files = find_val_files(root_dir)
    print(f"Found {len(val_files)} validation files")

    e2_data = []
    ocd_data = []
    for fp in val_files:
        e2, ocd = load_data(fp)
        e2_data.append(e2)
        ocd_data.append(ocd)
    e2_data = np.stack(e2_data)  # (139, 89, 181, 361)
    ocd_data = np.stack(ocd_data)  # (139, 6)
    print(f"Data loaded: E2 shape={e2_data.shape}, range=[{e2_data.min():.0f}, {e2_data.max():.0f}]")
    print(f"OCD shape={ocd_data.shape}")

    # Load models
    print("Loading SVD model (Model_SVD_w89_p1_2)...")
    svd_model = SVDModel(n_lambda=N_LAMBDA)
    svd_ckpt = torch.load(args.svd_ckpt, map_location=device)
    if isinstance(svd_ckpt, dict) and 'model' in svd_ckpt:
        svd_model.load_state_dict(svd_ckpt['model'], strict=False)
    else:
        svd_model.load_state_dict(svd_ckpt, strict=False)
    svd_model = svd_model.to(device)
    svd_model.eval()

    print("Loading Raw model (Model_raw89_1)...")
    raw_model = RawModel()
    raw_ckpt = torch.load(args.raw_ckpt, map_location=device)
    if isinstance(raw_ckpt, dict) and 'model' in raw_ckpt:
        raw_model.load_state_dict(raw_ckpt['model'], strict=False)
    else:
        raw_model.load_state_dict(raw_ckpt, strict=False)
    raw_model = raw_model.to(device)
    raw_model.eval()

    # Run experiments
    results = {}
    for sigma in NOISE_LEVELS:
        print(f"\n{'='*60}")
        print(f"σ = {sigma}")
        print(f"{'='*60}")

        svd_mape_trials = []
        raw_mape_trials = []

        for trial in range(args.n_trials):
            rng = np.random.default_rng(seed=42 + trial)

            # SVD model
            t0 = time.time()
            mape_s, std_s = evaluate_svd_model(svd_model, e2_data, ocd_data, sigma, device, rng)
            svd_mape_trials.append(mape_s)
            print(f"  SVD  trial {trial} ({time.time()-t0:.1f}s): " + " ".join(f"{m:.2f}" for m in mape_s))

            # Raw model
            t0 = time.time()
            mape_r, std_r = evaluate_raw_model(raw_model, e2_data, ocd_data, sigma, device, rng)
            raw_mape_trials.append(mape_r)
            print(f"  Raw  trial {trial} ({time.time()-t0:.1f}s): " + " ".join(f"{m:.2f}" for m in mape_r))

        results[sigma] = {
            'svd': np.array(svd_mape_trials),  # (n_trials, 6)
            'raw': np.array(raw_mape_trials),  # (n_trials, 6)
        }

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Mean MAPE (%) across trials")
    print(f"{'='*60}")
    header = f"{'σ':>6} | {'Model':>5} | " + " | ".join(f"{n:>8}" for n in OCD_NAMES)
    print(header)
    print("-" * len(header))
    for sigma in NOISE_LEVELS:
        for model_name in ['svd', 'raw']:
            mean_mape = results[sigma][model_name].mean(axis=0)
            std_mape = results[sigma][model_name].std(axis=0)
            row = f"{sigma:>6.2f} | {model_name:>5} | " + " | ".join(f"{m:>8.2f}" for m in mean_mape)
            print(row)
            row_std = f"{'':>6} | {'':>5} | " + " | ".join(f"±{s:<7.2f}" for s in std_mape)
            print(row_std)
        print()

    # Save results
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_file = join(out_dir, 'noise_robustness_results.csv')
    with open(out_file, 'w') as f:
        f.write("sigma,model,retch,Rtop,Rbot,n,height,b\n")
        for sigma in NOISE_LEVELS:
            for model_name in ['svd', 'raw']:
                mean_mape = results[sigma][model_name].mean(axis=0)
                f.write(f"{sigma},{model_name}," + ",".join(f"{m:.4f}" for m in mean_mape) + "\n")
    print(f"\nResults saved to: {out_file}")


if __name__ == '__main__':
    main()