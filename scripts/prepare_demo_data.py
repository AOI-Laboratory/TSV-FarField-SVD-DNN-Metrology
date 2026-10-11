"""
Extract a demo subset (N geometries) from the full FDTD dataset.
Creates data/demo/training/ and data/demo/validation/ directories.

Usage:
    python prepare_demo_data.py -f /path/to/full_dataset -n 50

The full dataset directory should contain:
    training/SN*.hdf5_norm_E2_SVD
    validation/SN*.hdf5_norm_E2_SVD
"""
import os, sys, shutil, argparse
from os.path import join, abspath, basename

HDF5_EXT = '.hdf5_norm_E2_SVD'


def find_files(directory):
    match = []
    for root, dirs, files in os.walk(directory):
        for f in sorted(files):
            if f.endswith(HDF5_EXT):
                match.append(os.path.join(root, f))
    return match


def main():
    parser = argparse.ArgumentParser(description='Extract demo data subset')
    parser.add_argument('-f', '--file', required=True, help='Full dataset root (contains training/ and validation/)')
    parser.add_argument('-n', '--num', type=int, default=50, help='Total number of geometries for demo')
    parser.add_argument('-o', '--output', default=None, help='Output directory (default: ./data/demo)')
    args = parser.parse_args()

    root = abspath(args.file)
    out_dir = abspath(args.output) if args.output else join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'demo')

    train_files = find_files(join(root, 'training'))
    val_files = find_files(join(root, 'validation'))
    print(f"Found {len(train_files)} training, {len(val_files)} validation files")

    # Split: 80% train, 20% val
    n_val = max(5, args.num // 5)
    n_train = args.num - n_val

    # Take first N from each (sorted by filename for reproducibility)
    selected_train = train_files[:n_train]
    selected_val = val_files[:n_val]

    os.makedirs(join(out_dir, 'training'), exist_ok=True)
    os.makedirs(join(out_dir, 'validation'), exist_ok=True)

    print(f"\nCopying {len(selected_train)} training files...")
    for fp in selected_train:
        dst = join(out_dir, 'training', basename(fp))
        shutil.copy2(fp, dst)

    print(f"Copying {len(selected_val)} validation files...")
    for fp in selected_val:
        dst = join(out_dir, 'validation', basename(fp))
        shutil.copy2(fp, dst)

    total_size = sum(os.path.getsize(join(out_dir, 'training', f)) for f in os.listdir(join(out_dir, 'training')))
    total_size += sum(os.path.getsize(join(out_dir, 'validation', f)) for f in os.listdir(join(out_dir, 'validation')))
    print(f"\nDone. Demo data: {len(selected_train)} train + {len(selected_val)} val = {args.num} geometries")
    print(f"Total size: {total_size / 1e6:.1f} MB")
    print(f"Output: {out_dir}")


if __name__ == '__main__':
    main()