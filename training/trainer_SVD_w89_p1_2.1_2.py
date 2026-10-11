"""
Training script for Polar (θ,φ) SVD-compressed model (89 wavelengths, rank p=1).

Trains DUVModel from Model_SVD_w89_p1_2 on HDF5 far-field data.
Uses NAdam optimizer with L1 loss. Saves best checkpoint based on
validation MAPE (Rtop + Rbot + height).

Usage:
    python trainer_SVD_w89_p1_2.1_2.py -f /path/to/dataset
    python trainer_SVD_w89_p1_2.1_2.py -f /path/to/dataset -c /path/to/checkpoint.pt

Paper: "SVD of Far-Field EM Data for Compressing and Accelerating DNNs
       in TSV Geometric Parameter Prediction" (IEEE TIM, MEAS_D_2613346)
"""
import sys, os, time, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models'))
import argparse
from os.path import join, dirname, basename, splitext, abspath, exists
import logging
import logging.handlers
import pandas as pd
import h5py
import datetime
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchinfo import summary
from Model_SVD_w89_p1_2 import DUVModel

# CUDA and HDF5 environment configuration
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# Model configuration
N_LAMBDA = 89
DATASET_TYPE = ['DGSM', 'Sobol', 'Random']
OCD_MAP = {'retch': 0, 'Rtop': 1, 'Rbot': 2, 'n': 3, 'height': 4, 'b': 5}
MODEL = '{}'.format(splitext(basename(__file__))[0].replace('trainer', 'CHKPT'))
HDF5_EXT_SVD = '.hdf5_norm_E2_SVD'

FLOAT32_MIN = 1.1754944e-38
FLOAT32_MAX = 3.4028235e+38
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")


# ===================================================================================================================
# Hyperparameters
# ===================================================================================================================
num_epochs = 100000
batch_size = 32
learning_rate = 5.0e-6
weight_decay = 1.0e-4


def main(root_dir, ckpt):
    """Main training loop.

    Args:
        root_dir: Path to dataset root (containing 'training/' and 'validation/').
        ckpt: Optional path to a pre-trained checkpoint for fine-tuning.
    """
    logger = logging.getLogger(__name__)
    logger.info('torch version: {}'.format(torch.version.__version__))
    logger.info('cudnn version: {}'.format(torch.backends.cudnn.version()))
    if torch.cuda.is_available():
        print('How many GPUs? {}'.format(torch.cuda.device_count()))
        device = torch.device('cuda')
    else:
        print('Only CPU')
        device = torch.device('cpu')

    # Discover HDF5 data files
    train_files = find_files(join(root_dir, 'training'))
    val_files = find_files(join(root_dir, 'validation'))

    logging.info('Found {} training files'.format(len(train_files)))
    logging.info('Found {} validation files'.format(len(val_files)))
    logger.info('batch size = {}'.format(batch_size))
    logger.info('learning rate = {}'.format(learning_rate))
    logger.info('weight decay = {}'.format(weight_decay))

    num_workers = 16

    # Create data loaders
    train_dataset = DUVDataset(train_files)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dataset = DUVDataset(val_files)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    freq = train_dataset.freq
    logger.info('wavelength = {}'.format(freq))
    logger.info('nFreq = {}'.format(len(freq)))

    # Initialize model (from checkpoint or ImageNet-pretrained ResNet)
    if ckpt:
        model = DUVModel(n_lambda=N_LAMBDA)
        model.load_state_dict(torch.load(ckpt), strict=False)
    else:
        model = DUVModel(n_lambda=N_LAMBDA)
        model.resnet.load_state_dict(torch.load(join('..', 'resnet18-f37072fd.pth')), strict=False)

    total_params = count_parameters(model)
    logger.info('Total trainable parameters = {}'.format(total_params))

    summary_str = str(summary(model, input_size=(batch_size, 1, N_LAMBDA * 2, 361), verbose=0))
    logger.info(summary_str)

    model = model.to(device)

    # NAdam optimizer with decoupled weight decay
    optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay, decoupled_weight_decay=True)

    # TensorBoard logging
    tensorboard_path = join(MODEL, timestamp, 'logs')
    os.makedirs(tensorboard_path, exist_ok=True)
    writer = SummaryWriter(tensorboard_path)

    time_consumption = list()
    best_regression_mape = FLOAT32_MAX

    for epoch in range(num_epochs):
        total_train_loss = 0.0
        total_val_loss = 0.0

        # Per-parameter PE (percentage error) accumulators for training set
        retch_pe_list = np.array(list())
        Rtop_pe_list = np.array(list())
        Rbot_pe_list = np.array(list())
        n_pe_list = np.array(list())
        height_pe_list = np.array(list())
        b_pe_list = np.array(list())

        train_spent_time = 0.0
        val_spent_time = 0.0

        # ==================== Training phase ====================
        for ti, (images, targets) in enumerate(train_dataloader):
            start_time = time.time()
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass and loss computation
            pred_ocd = model(images)
            loss = DUVLoss(pred_ocd, targets)

            # Backward pass and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss = total_train_loss + loss

            # Log weight/gradient norms from first batch (for monitoring)
            if ti == 0:
                ocd_weights = torch.norm(model.enc_ocd.feature_linear.weight, 1)
                ocd_grads = torch.norm(model.enc_ocd.feature_linear.weight.grad, 1)

            # Compute per-parameter percentage error (PE) on training batch
            with torch.no_grad():
                pred_ocd = pred_ocd.cpu().numpy()
                pred_ocd = train_dataset.denormalize_ocd(pred_ocd)
                ocd = targets.cpu().numpy()
                ocd = train_dataset.denormalize_ocd(ocd)
                diff = abs(pred_ocd - ocd)
                pe = (diff / ocd) * 100.0
                retch_pe_list = np.concatenate((retch_pe_list, np.array(pe[:, 0])))
                Rtop_pe_list = np.concatenate((Rtop_pe_list, np.array(pe[:, 1])))
                Rbot_pe_list = np.concatenate((Rbot_pe_list, np.array(pe[:, 2])))
                n_pe_list = np.concatenate((n_pe_list, np.array(pe[:, 3])))
                height_pe_list = np.concatenate((height_pe_list, np.array(pe[:, 4])))
                b_pe_list = np.concatenate((b_pe_list, np.array(pe[:, 5])))

            train_spent_time = train_spent_time + time.time() - start_time

        total_train_loss = total_train_loss / (ti + 1)

        # Compute training MAPE and STD per parameter
        retch_mape = np.mean(retch_pe_list)
        Rtop_mape = np.mean(Rtop_pe_list)
        Rbot_mape = np.mean(Rbot_pe_list)
        n_mape = np.mean(n_pe_list)
        height_mape = np.mean(height_pe_list)
        b_mape = np.mean(b_pe_list)

        retch_mape_std = np.std(retch_pe_list)
        Rtop_mape_std = np.std(Rtop_pe_list)
        Rbot_mape_std = np.std(Rbot_pe_list)
        n_mape_std = np.std(n_pe_list)
        height_mape_std = np.std(height_pe_list)
        b_mape_std = np.std(b_pe_list)

        logger.info('Epoch {}'.format(epoch + 1))
        logger.info('Train Loss =  {}'.format(total_train_loss))
        logger.info('Train MAPE (%) = <{:.4f}> <{:.4f}> <{:.4f}> <{:.4f}> <{:.4f}> <{:.4f}>'.format(
            retch_mape, Rtop_mape, Rbot_mape, n_mape, height_mape, b_mape))
        logger.info('Train STD (%) = <{:.4f}> <{:.4f}> <{:.4f}> <{:.4f}> <{:.4f}> <{:.4f}>'.format(
            retch_mape_std, Rtop_mape_std, Rbot_mape_std, n_mape_std, height_mape_std, b_mape_std))
        logger.info('weight: [{:.8f}]'.format(ocd_weights))
        logger.info('grad: [{:.8f}]'.format(ocd_grads))

        # Log training metrics to TensorBoard
        writer.add_scalar('Train Loss', loss.item(), epoch)
        writer.add_scalar('Train retch MAPE', retch_mape, epoch)
        writer.add_scalar('Train Rtop MAPE', Rtop_mape, epoch)
        writer.add_scalar('Train Rbot MAPE', Rbot_mape, epoch)
        writer.add_scalar('Train n MAPE', n_mape, epoch)
        writer.add_scalar('Train height MAPE', height_mape, epoch)
        writer.add_scalar('Train b MAPE', b_mape, epoch)
        writer.add_scalar('Train retch STD', retch_mape_std, epoch)
        writer.add_scalar('Train Rtop STD', Rtop_mape_std, epoch)
        writer.add_scalar('Train Rbot STD', Rbot_mape_std, epoch)
        writer.add_scalar('Train n STD', n_mape_std, epoch)
        writer.add_scalar('Train height STD', height_mape_std, epoch)
        writer.add_scalar('Train b STD', b_mape_std, epoch)

        # ==================== Validation phase ====================
        retch_pe_list = np.array(list())
        Rtop_pe_list = np.array(list())
        Rbot_pe_list = np.array(list())
        n_pe_list = np.array(list())
        height_pe_list = np.array(list())
        b_pe_list = np.array(list())

        model.eval()
        with torch.no_grad():
            for vi, (images, targets) in enumerate(val_dataloader):
                start_time = time.time()
                images = images.to(device)
                targets = targets.to(device)

                pred_ocd = model(images)
                loss = DUVLoss(pred_ocd, targets)
                total_val_loss = total_val_loss + loss

                # Compute per-parameter PE on validation batch
                pred_ocd = pred_ocd.cpu().numpy()
                pred_ocd = train_dataset.denormalize_ocd(pred_ocd)
                ocd = targets.cpu().numpy()
                ocd = train_dataset.denormalize_ocd(ocd)
                diff = abs(pred_ocd - ocd)
                pe = (diff / ocd) * 100.0
                retch_pe_list = np.concatenate((retch_pe_list, np.array(pe[:, 0])))
                Rtop_pe_list = np.concatenate((Rtop_pe_list, np.array(pe[:, 1])))
                Rbot_pe_list = np.concatenate((Rbot_pe_list, np.array(pe[:, 2])))
                n_pe_list = np.concatenate((n_pe_list, np.array(pe[:, 3])))
                height_pe_list = np.concatenate((height_pe_list, np.array(pe[:, 4])))
                b_pe_list = np.concatenate((b_pe_list, np.array(pe[:, 5])))

                val_spent_time = val_spent_time + time.time() - start_time

        total_val_loss = total_val_loss / (vi + 1)

        # Compute validation MAPE and STD per parameter
        retch_mape = np.mean(retch_pe_list)
        Rtop_mape = np.mean(Rtop_pe_list)
        Rbot_mape = np.mean(Rbot_pe_list)
        n_mape = np.mean(n_pe_list)
        height_mape = np.mean(height_pe_list)
        b_mape = np.mean(b_pe_list)

        retch_mape_std = np.std(retch_pe_list)
        Rtop_mape_std = np.std(Rtop_pe_list)
        Rbot_mape_std = np.std(Rbot_pe_list)
        n_mape_std = np.std(n_pe_list)
        height_mape_std = np.std(height_pe_list)
        b_mape_std = np.std(b_pe_list)

        logger.info('Val Loss =  {}'.format(total_val_loss))
        logger.info('Val MAPE (%) = <{:.4f}> <{:.4f}> <{:.4f}> <{:.4f}> <{:.4f}> <{:.4f}>'.format(
            retch_mape, Rtop_mape, Rbot_mape, n_mape, height_mape, b_mape))
        logger.info('Val STD (%) = <{:.4f}> <{:.4f}> <{:.4f}> <{:.4f}> <{:.4f}> <{:.4f}>'.format(
            retch_mape_std, Rtop_mape_std, Rbot_mape_std, n_mape_std, height_mape_std, b_mape_std))

        # Log validation metrics to TensorBoard
        writer.add_scalar('Val Loss', loss.item(), epoch)
        writer.add_scalar('Val retch MAPE', retch_mape, epoch)
        writer.add_scalar('Val Rtop MAPE', Rtop_mape, epoch)
        writer.add_scalar('Val Rbot MAPE', Rbot_mape, epoch)
        writer.add_scalar('Val n MAPE', n_mape, epoch)
        writer.add_scalar('Val height MAPE', height_mape, epoch)
        writer.add_scalar('Val b MAPE', b_mape, epoch)
        writer.add_scalar('Val retch STD', retch_mape_std, epoch)
        writer.add_scalar('Val Rtop STD', Rtop_mape_std, epoch)
        writer.add_scalar('Val Rbot STD', Rbot_mape_std, epoch)
        writer.add_scalar('Val n STD', n_mape_std, epoch)
        writer.add_scalar('Val height STD', height_mape_std, epoch)
        writer.add_scalar('Val b STD', b_mape_std, epoch)

        # ==================== Checkpoint saving ====================
        spent_time = train_spent_time + val_spent_time
        time_consumption.append(spent_time)

        # Best model criterion: sum of Rtop + Rbot + height MAPE
        val_mape = Rtop_mape + Rbot_mape + height_mape
        if best_regression_mape > val_mape:
            logger.info('*Epoch {} spent {} seconds. ({:.4f})'.format(epoch + 1, spent_time, best_regression_mape - val_mape))
            writer.flush()
            best_regression_mape = val_mape
            torch.save(model.state_dict(), join(MODEL, timestamp, 'model_ep-{}.pt'.format(epoch + 1)))
        else:
            logger.info('Epoch {} spent {} seconds'.format(epoch + 1, spent_time))

        logger.info('================================================')

        # Save time consumption CSV after epoch 11
        if epoch + 1 == 11:
            df = pd.DataFrame({'time_consumption': time_consumption})
            df.to_csv(join(MODEL, timestamp, 'training_time_consumption.csv'), index=True)

    writer.close()


def DUVLoss(preds, targets):
    """L1 loss (MAE) between predicted and target OCD parameters."""
    return F.l1_loss(preds, targets, reduction='mean')


class DUVDataset(Dataset):
    """PyTorch Dataset for SVD-compressed far-field data.

    Each sample loads Us and Vt from HDF5, pads Us to match Vt width,
    concatenates along height, and normalizes OCD targets to [0, 1].

    Input shape: (1, n_lambda*2, 361) = [pad(Us); Vt]
    """

    def __init__(self, data):
        self.data = data
        self.start_freq = 0
        self.step_freq = 27
        self.topN = 1  # SVD rank (number of singular vectors)
        # OCD normalization bounds: [retch, Rtop, Rbot, n, height, b]
        self.ocd_min = np.array([0.02, 0.27, 0.2, 100.0, 2.7, 0.05])
        self.ocd_max = np.array([0.1, 0.33, 0.27, 600.0, 3.3, 0.1])
        self.ocd_norm_minus = self.ocd_min
        self.ocd_norm_div = self.ocd_max - self.ocd_min

        # Read wavelength array from first file (for logging)
        with h5py.File(self.data[0], "r") as f:
            freq = f['freq'][()]
            self.freq = freq[self.start_freq::self.step_freq]

    def __len__(self):
        return len(self.data)

    def denormalize_ocd(self, ocd):
        """Convert normalized OCD values back to physical units."""
        return ocd * self.ocd_norm_div + self.ocd_norm_minus

    def __getitem__(self, index):
        file_path = self.data[index]
        with h5py.File(file_path, "r") as f:
            Us = f['Us'][::9, :, :self.topN]
            Vt = f['Vt'][::9, :self.topN]
            retch = f['retch'][()]
            Rtop = f['Rtop'][()]
            Rbot = f['Rbot'][()]
            n = f['n'][()]
            height = f['height'][()]
            b = f['b'][()]

        # Squeeze rank dimension: Us (89, 181), Vt (89, 361)
        Us = np.squeeze(Us)
        Vt = np.squeeze(Vt)

        # Pad Us from (89, 181) to (89, 361) to match Vt width
        f, n_row = Us.shape
        f, n_col = Vt.shape
        pad_Us = np.pad(Us, ((0, 0), (0, n_col - n_row)), mode='constant', constant_values=0)

        # Concatenate [pad(Us); Vt] along height → (1, 178, 361)
        images = np.expand_dims(np.concatenate((pad_Us, Vt), axis=0), axis=0)

        # Normalize OCD targets to [0, 1]
        ocd = np.array([retch, Rtop, Rbot, n, height, b])
        ocd = (ocd - self.ocd_norm_minus) / self.ocd_norm_div
        ocd = torch.tensor(ocd, dtype=torch.float)
        images = torch.tensor(images, dtype=torch.float)

        return images, ocd


def weight_histograms_conv2d(writer, step, weights, layer_number):
    """Log per-kernel weight histograms for a Conv2d layer."""
    weights_shape = weights.shape
    num_kernels = weights_shape[0]
    for k in range(num_kernels):
        flattened_weights = weights[k].flatten()
        tag = f"layer_{layer_number}/kernel_{k}"
        writer.add_histogram(tag, flattened_weights, global_step=step, bins='tensorflow')


def weight_histograms_linear(writer, step, weights, layer_number):
    """Log weight histogram for a Linear layer."""
    flattened_weights = weights.flatten()
    tag = f"layer_{layer_number}"
    writer.add_histogram(tag, flattened_weights, global_step=step, bins='tensorflow')


def weight_histograms(writer, step, model):
    """Log weight histograms for all Conv2d and Linear layers."""
    print("Visualizing model weights...")
    for idx, m in enumerate(model.modules()):
        if isinstance(m, nn.Conv2d):
            weight_histograms_conv2d(writer, step, m.weight, idx)
        elif isinstance(m, nn.Linear):
            weight_histograms_linear(writer, step, m.weight, idx)


def count_parameters(model):
    """Count total trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def find_files(directory):
    """Recursively find all HDF5 SVD data files in a directory."""
    match_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(HDF5_EXT_SVD):
                match_files.append(os.path.join(root, file))
    return match_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train SVD-compressed model (89 wavelengths, rank p=1)')

    parser.add_argument('-f', '--file',
                        help='Path to dataset root directory',
                        required=True)

    parser.add_argument('-c', '--ckpt',
                        help='Path to pre-trained checkpoint (optional)',
                        required=False)

    args = parser.parse_args()

    if args.ckpt:
        ckpt = abspath(args.ckpt)
    else:
        ckpt = None

    # Set up logging (file + console)
    log_path = join(MODEL, timestamp)
    os.makedirs(log_path, exist_ok=True)

    file_handler = logging.FileHandler(join(log_path, 'training_log.txt'))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    console = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    console.setLevel(logging.DEBUG)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console)

    logger.info('root dir = {}'.format(abspath(args.file)))
    logger.info('checkpoint file = {}'.format(ckpt))

    main(abspath(args.file), ckpt)

    file_handler.flush()