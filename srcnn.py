import csv
import os
import math
import random
import shutil
import time
from enum import Enum
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# ============================================================================
# CONFIGURATION
# ============================================================================
class CONFIG:
    """All configuration parameters for training"""
    
    # Random seeds for reproducibility
    RANDOM_SEED = 0
    
    # Device configuration
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    USE_CUDNN_BENCHMARK = True
    
    # Dataset paths - MODIFY THESE FOR YOUR CUSTOM DATASET
    # Training data: LR and HR image pairs
    TRAIN_LR_IMAGE_DIR = "/kaggle/input/screen-sr-data/Screen-SR-Dataset/Train/LR"  # Path to training low-resolution images
    TRAIN_HR_IMAGE_DIR = "/kaggle/input/screen-sr-data/Screen-SR-Dataset/Train/HR"  # Path to training high-resolution images
    
    # Test data: LR and HR image pairs
    TEST_LR_IMAGE_DIR = "/kaggle/input/screen-sr-data/Screen-SR-Dataset/Test/LR"   # Path to test low-resolution images
    TEST_HR_IMAGE_DIR = "/kaggle/input/screen-sr-data/Screen-SR-Dataset/Test/HR"   # Path to test high-resolution images
    
    # Model parameters
    UPSCALE_FACTOR = 2  # Super-resolution scale factor (2x, 3x, or 4x)
    
    # Training parameters
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    EPOCHS = 22  # Reduced for finetuning
    
    # Optimizer parameters (SGD)
    LEARNING_RATE = 1e-3  # Lower learning rate for finetuning
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    USE_NESTEROV = False
    
    # Pretrained model and finetuning strategy
    PRETRAINED_WEIGHTS_PATH = "/kaggle/input/epoch-150-face/pytorch/default/1/epoch_148_face.pth.tar"  # REQUIRED: Path to pretrained model
    FREEZE_PERCENTAGE = 0.0  # Freeze 80% of parameters (train only last 20%)
    
    # Checkpoint and logging
    EXP_NAME = "SRCNN_finetune"  # Experiment name for saving weights
    RESUME_PATH = ""  # Path to checkpoint to resume training (includes optimizer state, epoch, etc.)
    PRINT_FREQUENCY = 100  # Print training stats every N batches
    
    # Output directories
    SAMPLES_DIR = "./samples"
    RESULTS_DIR = "./results"
    
    # Logging and Visualization
    RESULTS_CSV = "results.csv"
    TRAIN_LOSS_PLOT = "train_loss.png"
    VAL_LOSS_PLOT = "val_loss.png"
    PSNR_PLOT = "psnr.png"
    SSIM_PLOT = "ssim.png"


# ============================================================================
# IMAGE PROCESSING UTILITIES
# ============================================================================
class ImageProcessor:
    """Image processing utilities for SRCNN training"""
    
    @staticmethod
    def image2tensor(image: np.ndarray, range_norm: bool = False, half: bool = False) -> torch.Tensor:
        """Convert image to PyTorch tensor (NCHW format)"""
        tensor = F.to_tensor(image)
        if range_norm:
            tensor = tensor.mul(2.0).sub(1.0)
        if half:
            tensor = tensor.half()
        return tensor
    
    @staticmethod
    def _cubic(x: Any) -> Any:
        """Bicubic interpolation kernel"""
        absx = torch.abs(x)
        absx2 = absx ** 2
        absx3 = absx ** 3
        return (1.5 * absx3 - 2.5 * absx2 + 1) * ((absx <= 1).type_as(absx)) + \
               (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * (((absx > 1) * (absx <= 2)).type_as(absx))
    
    @staticmethod
    def _calculate_weights_indices(in_length: int, out_length: int, scale: float,
                                   kernel_width: int, antialiasing: bool):
        """Calculate weights and indices for image resizing"""
        if (scale < 1) and antialiasing:
            kernel_width = kernel_width / scale
        
        x = torch.linspace(1, out_length, out_length)
        u = x / scale + 0.5 * (1 - 1 / scale)
        left = torch.floor(u - kernel_width / 2)
        p = math.ceil(kernel_width) + 2
        
        indices = left.view(out_length, 1).expand(out_length, p) + \
                  torch.linspace(0, p - 1, p).view(1, p).expand(out_length, p)
        distance_to_center = u.view(out_length, 1).expand(out_length, p) - indices
        
        if (scale < 1) and antialiasing:
            weights = scale * ImageProcessor._cubic(distance_to_center * scale)
        else:
            weights = ImageProcessor._cubic(distance_to_center)
        
        weights_sum = torch.sum(weights, 1).view(out_length, 1)
        weights = weights / weights_sum.expand(out_length, p)
        
        weights_zero_tmp = torch.sum((weights == 0), 0)
        if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
            indices = indices.narrow(1, 1, p - 2)
            weights = weights.narrow(1, 1, p - 2)
        if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
            indices = indices.narrow(1, 0, p - 2)
            weights = weights.narrow(1, 0, p - 2)
        
        weights = weights.contiguous()
        indices = indices.contiguous()
        sym_len_s = -indices.min() + 1
        sym_len_e = indices.max() - in_length
        indices = indices + sym_len_s - 1
        
        return weights, indices, int(sym_len_s), int(sym_len_e)
    
    @staticmethod
    def image_resize(image: Any, scale_factor: float, antialiasing: bool = True) -> Any:
        """Resize image using bicubic interpolation (Matlab-style)"""
        squeeze_flag = False
        if type(image).__module__ == np.__name__:
            numpy_type = True
            if image.ndim == 2:
                image = image[:, :, None]
                squeeze_flag = True
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        else:
            numpy_type = False
            if image.ndim == 2:
                image = image.unsqueeze(0)
                squeeze_flag = True
        
        in_c, in_h, in_w = image.size()
        out_h, out_w = math.ceil(in_h * scale_factor), math.ceil(in_w * scale_factor)
        kernel_width = 4
        
        weights_h, indices_h, sym_len_hs, sym_len_he = ImageProcessor._calculate_weights_indices(
            in_h, out_h, scale_factor, kernel_width, antialiasing)
        weights_w, indices_w, sym_len_ws, sym_len_we = ImageProcessor._calculate_weights_indices(
            in_w, out_w, scale_factor, kernel_width, antialiasing)
        
        # Process H dimension
        img_aug = torch.FloatTensor(in_c, in_h + sym_len_hs + sym_len_he, in_w)
        img_aug.narrow(1, sym_len_hs, in_h).copy_(image)
        
        sym_patch = image[:, :sym_len_hs, :]
        inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
        sym_patch_inv = sym_patch.index_select(1, inv_idx)
        img_aug.narrow(1, 0, sym_len_hs).copy_(sym_patch_inv)
        
        sym_patch = image[:, -sym_len_he:, :]
        inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
        sym_patch_inv = sym_patch.index_select(1, inv_idx)
        img_aug.narrow(1, sym_len_hs + in_h, sym_len_he).copy_(sym_patch_inv)
        
        out_1 = torch.FloatTensor(in_c, out_h, in_w)
        kernel_width = weights_h.size(1)
        for i in range(out_h):
            idx = int(indices_h[i][0])
            for j in range(in_c):
                out_1[j, i, :] = img_aug[j, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_h[i])
        
        # Process W dimension
        out_1_aug = torch.FloatTensor(in_c, out_h, in_w + sym_len_ws + sym_len_we)
        out_1_aug.narrow(2, sym_len_ws, in_w).copy_(out_1)
        
        sym_patch = out_1[:, :, :sym_len_ws]
        inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
        sym_patch_inv = sym_patch.index_select(2, inv_idx)
        out_1_aug.narrow(2, 0, sym_len_ws).copy_(sym_patch_inv)
        
        sym_patch = out_1[:, :, -sym_len_we:]
        inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
        sym_patch_inv = sym_patch.index_select(2, inv_idx)
        out_1_aug.narrow(2, sym_len_ws + in_w, sym_len_we).copy_(sym_patch_inv)
        
        out_2 = torch.FloatTensor(in_c, out_h, out_w)
        kernel_width = weights_w.size(1)
        for i in range(out_w):
            idx = int(indices_w[i][0])
            for j in range(in_c):
                out_2[j, :, i] = out_1_aug[j, :, idx:idx + kernel_width].mv(weights_w[i])
        
        if squeeze_flag:
            out_2 = out_2.squeeze(0)
        if numpy_type:
            out_2 = out_2.numpy()
            if not squeeze_flag:
                out_2 = out_2.transpose(1, 2, 0)
        
        return out_2
    
    @staticmethod
    def bgr2ycbcr(image: np.ndarray, only_use_y_channel: bool = True) -> np.ndarray:
        """Convert BGR image to YCbCr color space"""
        if only_use_y_channel:
            image = np.dot(image, [24.966, 128.553, 65.481]) + 16.0
        else:
            image = np.matmul(image, [[24.966, 112.0, -18.214],
                                     [128.553, -74.203, -93.786],
                                     [65.481, -37.797, 112.0]]) + [16, 128, 128]
        image /= 255.
        return image.astype(np.float32)
    
    @staticmethod
    def center_crop(image: np.ndarray, image_size: int) -> np.ndarray:
        """Crop image from center"""
        image_height, image_width = image.shape[:2]
        top = (image_height - image_size) // 2
        left = (image_width - image_size) // 2
        return image[top:top + image_size, left:left + image_size, ...]
    
    @staticmethod
    def random_crop(image: np.ndarray, image_size: int) -> np.ndarray:
        """Randomly crop image patch"""
        image_height, image_width = image.shape[:2]
        top = random.randint(0, image_height - image_size)
        left = random.randint(0, image_width - image_size)
        return image[top:top + image_size, left:left + image_size, ...]


# ============================================================================
# DATASET
# ============================================================================
class PairedImageDataset(Dataset):
    """Dataset for training with paired LR and HR images"""
    
    def __init__(self, lr_image_dir: str, hr_image_dir: str, upscale_factor: int):
        super(PairedImageDataset, self).__init__()
        # Get all image files from LR directory
        lr_files = sorted([f for f in os.listdir(lr_image_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        hr_files = sorted([f for f in os.listdir(hr_image_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        
        # Ensure same number of LR and HR images
        if len(lr_files) != len(hr_files):
            print(f"Warning: LR ({len(lr_files)}) and HR ({len(hr_files)}) image counts don't match!")
            # Use minimum count
            min_count = min(len(lr_files), len(hr_files))
            lr_files = lr_files[:min_count]
            hr_files = hr_files[:min_count]
        
        self.lr_image_paths = [os.path.join(lr_image_dir, f) for f in lr_files]
        self.hr_image_paths = [os.path.join(hr_image_dir, f) for f in hr_files]
        self.upscale_factor = upscale_factor
    
    def __getitem__(self, index: int):
        # Read LR and HR images
        lr_image = cv2.imread(self.lr_image_paths[index], cv2.IMREAD_UNCHANGED)
        hr_image = cv2.imread(self.hr_image_paths[index], cv2.IMREAD_UNCHANGED)
        
        if lr_image is None or hr_image is None:
            raise ValueError(f"Failed to load images at index {index}")
        
        # Normalize to [0, 1]
        lr_image = lr_image.astype(np.float32) / 255.
        hr_image = hr_image.astype(np.float32) / 255.
        
        # Convert to Y channel (luminance)
        lr_y_image = ImageProcessor.bgr2ycbcr(lr_image, only_use_y_channel=True)
        hr_y_image = ImageProcessor.bgr2ycbcr(hr_image, only_use_y_channel=True)
        
        # Resize LR to HR size (standard for SRCNN)
        if lr_y_image.shape != hr_y_image.shape:
            lr_y_image = ImageProcessor.image_resize(lr_y_image, self.upscale_factor)
            # If after resizing it still doesn't match exactly due to rounding, force it
            if lr_y_image.shape != hr_y_image.shape:
                lr_y_image = cv2.resize(lr_y_image, (hr_y_image.shape[1], hr_y_image.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        # Convert to tensor
        lr_y_tensor = ImageProcessor.image2tensor(lr_y_image, range_norm=False, half=False)
        hr_y_tensor = ImageProcessor.image2tensor(hr_y_image, range_norm=False, half=False)
        
        return {"lr": lr_y_tensor, "hr": hr_y_tensor}
    
    def __len__(self):
        return len(self.lr_image_paths)





class CUDAPrefetcher:
    """Data prefetcher for CUDA acceleration"""
    
    def __init__(self, dataloader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device
        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream() if device.type == 'cuda' else None
        self.preload()
    
    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None
        
        if self.device.type == 'cuda':
            with torch.cuda.stream(self.stream):
                for k, v in self.batch_data.items():
                    if torch.is_tensor(v):
                        self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)
        else:
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device)
    
    def next(self):
        if self.device.type == 'cuda':
            torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data
    
    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()
    
    def __len__(self):
        return len(self.original_dataloader)


# ============================================================================
# MODEL
# ============================================================================
class SRCNN(nn.Module):
    """Super-Resolution Convolutional Neural Network"""
    
    def __init__(self):
        super(SRCNN, self).__init__()
        
        # Feature extraction layer
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, (9, 9), (1, 1), (4, 4)),
            nn.ReLU(True)
        )
        
        # Non-linear mapping layer
        self.map = nn.Sequential(
            nn.Conv2d(64, 32, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(True)
        )
        
        # Reconstruction layer
        self.reconstruction = nn.Conv2d(32, 1, (5, 5), (1, 1), (2, 2))
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.map(out)
        out = self.reconstruction(out)
        return out
    
    def _initialize_weights(self):
        """Initialize network weights with Gaussian distribution"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight.data, 0.0, 
                              math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
                nn.init.zeros_(module.bias.data)
        
        nn.init.normal_(self.reconstruction.weight.data, 0.0, 0.001)
        nn.init.zeros_(self.reconstruction.bias.data)


# ============================================================================
# IMAGE QUALITY ASSESSMENT
# ============================================================================
class PSNR(nn.Module):
    """Peak Signal-to-Noise Ratio metric"""
    
    def __init__(self, upscale_factor: int, only_test_y_channel: bool = True):
        super(PSNR, self).__init__()
        self.upscale_factor = upscale_factor
        self.only_test_y_channel = only_test_y_channel
    
    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        # Crop border pixels
        crop_border = self.upscale_factor
        if self.only_test_y_channel:
            sr = sr[:, :, crop_border:-crop_border, crop_border:-crop_border]
            hr = hr[:, :, crop_border:-crop_border, crop_border:-crop_border]
        
        # Calculate MSE
        mse = torch.mean((sr - hr) ** 2)
        if mse == 0:
            return torch.tensor(100.0)
        
        # Calculate PSNR
        psnr = 10 * torch.log10(1.0 / mse)
        return psnr


class SSIM(nn.Module):
    """Structural Similarity Index Metric"""
    
    def __init__(self, upscale_factor: int, only_test_y_channel: bool = True):
        super(SSIM, self).__init__()
        self.upscale_factor = upscale_factor
        self.only_test_y_channel = only_test_y_channel
    
    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        # Crop border pixels
        crop_border = self.upscale_factor
        if self.only_test_y_channel:
            sr = sr[:, :, crop_border:-crop_border, crop_border:-crop_border]
            hr = hr[:, :, crop_border:-crop_border, crop_border:-crop_border]
        
        # SSIM parameters
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Calculate means
        mu_sr = torch.mean(sr)
        mu_hr = torch.mean(hr)
        
        # Calculate variances and covariance
        sigma_sr = torch.var(sr)
        sigma_hr = torch.var(hr)
        sigma_sr_hr = torch.mean((sr - mu_sr) * (hr - mu_hr))
        
        # Calculate SSIM
        numerator = (2 * mu_sr * mu_hr + C1) * (2 * sigma_sr_hr + C2)
        denominator = (mu_sr ** 2 + mu_hr ** 2 + C1) * (sigma_sr + sigma_hr + C2)
        ssim = numerator / denominator
        
        return ssim


# ============================================================================
# TRAINING UTILITIES
# ============================================================================
class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self, name: str, fmt: str = ":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        if self.summary_type is Summary.NONE:
            return ""
        elif self.summary_type is Summary.AVERAGE:
            return f"{self.name} {self.avg:.2f}"
        elif self.summary_type is Summary.SUM:
            return f"{self.name} {self.sum:.2f}"
        elif self.summary_type is Summary.COUNT:
            return f"{self.name} {self.count:.2f}"
        else:
            raise ValueError(f"Invalid summary type {self.summary_type}")


class ProgressMeter:
    """Progress meter for training/validation"""
    
    def __init__(self, num_batches: int, meters: list, prefix: str = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    
    def display(self, batch: int):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))
    
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))
    
    def _get_batch_fmtstr(self, num_batches: int):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


class CSVLogger:
    """Handles logging of epoch metrics to a CSV file"""
    def __init__(self, file_path: str, header: list):
        self.file_path = file_path
        self.header = header
        if not os.path.exists(file_path):
            with open(file_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
    
    def log(self, metrics: list):
        with open(self.file_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(metrics)


def save_plots(csv_path: str, output_dir: str, config: CONFIG):
    """Generates and saves line graphs for all metrics from the CSV log"""
    if plt is None:
        print("Warning: matplotlib not installed, skipping plots.")
        return

    epochs, train_losses, val_losses, psnrs, ssims = [], [], [], [], []
    
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))
            psnrs.append(float(row["psnr"]))
            ssims.append(float(row["ssim"]))

    def _plot(x, y, title, ylabel, save_name, color):
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, color=color, linewidth=2, marker='o', markersize=4)
        plt.title(f"{title} over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.savefig(os.path.join(output_dir, save_name))
        plt.close()

    _plot(epochs, train_losses, "Training Loss", "MSE Loss", config.TRAIN_LOSS_PLOT, "blue")
    _plot(epochs, val_losses, "Validation Loss", "MSE Loss", config.VAL_LOSS_PLOT, "orange")
    _plot(epochs, psnrs, "Validation PSNR", "PSNR (dB)", config.PSNR_PLOT, "green")
    _plot(epochs, ssims, "Validation SSIM", "SSIM", config.SSIM_PLOT, "purple")


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def load_dataset(config: CONFIG):
    """Load training and test datasets using paired LR/HR directories"""
    train_dataset = PairedImageDataset(
        config.TRAIN_LR_IMAGE_DIR,
        config.TRAIN_HR_IMAGE_DIR,
        config.UPSCALE_FACTOR
    )
    test_dataset = PairedImageDataset(
        config.TEST_LR_IMAGE_DIR,
        config.TEST_HR_IMAGE_DIR,
        config.UPSCALE_FACTOR
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if config.NUM_WORKERS > 0 else False
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True
    )
    
    train_prefetcher = CUDAPrefetcher(train_dataloader, config.DEVICE)
    test_prefetcher = CUDAPrefetcher(test_dataloader, config.DEVICE)
    
    return train_prefetcher, test_prefetcher


def build_model(config: CONFIG) -> nn.Module:
    """Build and initialize SRCNN model"""
    model = SRCNN()
    model = model.to(device=config.DEVICE, memory_format=torch.channels_last)
    return model


def define_loss(config: CONFIG) -> nn.MSELoss:
    """Define loss function"""
    criterion = nn.MSELoss()
    criterion = criterion.to(device=config.DEVICE, memory_format=torch.channels_last)
    return criterion


def define_optimizer(model: nn.Module, config: CONFIG) -> optim.SGD:
    """Define optimizer for only trainable parameters"""
    # Filter parameters for each group to only include those that require gradients
    # This prevents the KeyError: 'weight_decay' that occurs when param_groups are manually filtered later
    params_features = [p for p in model.features.parameters() if p.requires_grad]
    params_map = [p for p in model.map.parameters() if p.requires_grad]
    params_reconstruction = [p for p in model.reconstruction.parameters() if p.requires_grad]
    
    param_groups = []
    if params_features:
        param_groups.append({"params": params_features})
    if params_map:
        param_groups.append({"params": params_map})
    if params_reconstruction:
        param_groups.append({"params": params_reconstruction, "lr": config.LEARNING_RATE * 0.1})
        
    optimizer = optim.SGD(
        param_groups,
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
        nesterov=config.USE_NESTEROV
    )
    return optimizer


def train(model: nn.Module, train_prefetcher: CUDAPrefetcher, criterion: nn.MSELoss,
          optimizer: optim.SGD, epoch: int, scaler: torch.amp.GradScaler, config: CONFIG):
    """Training loop for one epoch"""
    batches = len(train_prefetcher)
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.6f")
    progress = ProgressMeter(batches, [batch_time, data_time, losses], prefix=f"Epoch: [{epoch + 1}]")
    
    model.train()
    batch_index = 0
    train_prefetcher.reset()
    batch_data = train_prefetcher.next()
    end = time.time()
    
    while batch_data is not None:
        data_time.update(time.time() - end)
        
        lr = batch_data["lr"].to(device=config.DEVICE, memory_format=torch.channels_last, non_blocking=True)
        hr = batch_data["hr"].to(device=config.DEVICE, memory_format=torch.channels_last, non_blocking=True)
        
        model.zero_grad(set_to_none=True)
        
        # Mixed precision training
        with torch.amp.autocast(device_type=config.DEVICE.type, enabled=config.DEVICE.type == 'cuda'):
            sr = model(lr)
            loss = criterion(sr, hr)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        losses.update(loss.item(), lr.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        
        if batch_index % config.PRINT_FREQUENCY == 0:
            progress.display(batch_index)
        
        batch_data = train_prefetcher.next()
        batch_index += 1
    
    return losses.avg


def validate(model: nn.Module, data_prefetcher: CUDAPrefetcher, epoch: int,
            psnr_model: nn.Module, ssim_model: nn.Module, criterion: nn.Module, config: CONFIG, mode: str = "Test"):
    """Validation loop"""
    batches = len(data_prefetcher)
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":6.6f")
    psnres = AverageMeter("PSNR", ":4.2f")
    ssimes = AverageMeter("SSIM", ":4.4f")
    progress = ProgressMeter(batches, [batch_time, losses, psnres, ssimes], prefix=f"{mode}: ")
    
    model.eval()
    batch_index = 0
    data_prefetcher.reset()
    batch_data = data_prefetcher.next()
    end = time.time()
    
    with torch.no_grad():
        while batch_data is not None:
            lr = batch_data["lr"].to(device=config.DEVICE, memory_format=torch.channels_last, non_blocking=True)
            hr = batch_data["hr"].to(device=config.DEVICE, memory_format=torch.channels_last, non_blocking=True)
            
            with torch.amp.autocast(device_type=config.DEVICE.type, enabled=config.DEVICE.type == 'cuda'):
                sr = model(lr)
                loss = criterion(sr, hr)
            
            psnr = psnr_model(sr, hr)
            ssim = ssim_model(sr, hr)
            
            losses.update(loss.item(), lr.size(0))
            psnres.update(psnr.item(), lr.size(0))
            ssimes.update(ssim.item(), lr.size(0))
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            if batch_index % (batches // 5 + 1) == 0:
                progress.display(batch_index)
            
            batch_data = data_prefetcher.next()
            batch_index += 1
        
        progress.display_summary()
    
    return losses.avg, psnres.avg, ssimes.avg


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================
def main():
    """Main training function"""
    # Configuration
    config = CONFIG()
    
    # Set random seeds
    random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    if torch.cuda.is_available() and config.USE_CUDNN_BENCHMARK:
        from torch.backends import cudnn
        cudnn.benchmark = True
    
    print("=" * 80)
    print("SRCNN Fine-tuning - All-in-One Script")
    print("=" * 80)
    print(f"Device: {config.DEVICE}")
    print(f"Upscale Factor: {config.UPSCALE_FACTOR}x")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print("=" * 80)
    
    # Initialize tracking variables
    start_epoch = 0
    best_psnr = 0.0
    best_ssim = 0.0
    
    # Load datasets
    print("\nLoading datasets...")
    train_prefetcher, test_prefetcher = load_dataset(config)
    print(f"Training samples: {len(train_prefetcher.original_dataloader.dataset)}")
    print(f"Test samples: {len(test_prefetcher.original_dataloader.dataset)}")
    
    # Build model
    print("\nBuilding SRCNN model...")
    model = build_model(config)
    print("Model built successfully.")
    
    # Define loss and optimizer
    criterion = define_loss(config)
    optimizer = define_optimizer(model, config)
    print("Loss function and optimizer defined.")
    
    # Load pretrained weights for finetuning (if specified)
    if config.PRETRAINED_WEIGHTS_PATH and os.path.exists(config.PRETRAINED_WEIGHTS_PATH):
        print(f"\nLoading pretrained weights from: {config.PRETRAINED_WEIGHTS_PATH}")
        try:
            checkpoint = torch.load(config.PRETRAINED_WEIGHTS_PATH, map_location=lambda storage, loc: storage)
            
            # Try to load state_dict (handles both checkpoint format and direct state_dict)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                # Full checkpoint format (from training script)
                state_dict = checkpoint["state_dict"]
                print("Loaded from checkpoint format (with metadata)")
            elif isinstance(checkpoint, dict):
                # Direct state_dict format
                state_dict = checkpoint
                print("Loaded direct state_dict format")
            else:
                raise ValueError("Unsupported checkpoint format")
            
            # Load weights into model (ignore size mismatches if any)
            model_state_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
            model_state_dict.update(pretrained_dict)
            model.load_state_dict(model_state_dict)
            
            print(f"✓ Loaded {len(pretrained_dict)}/{len(model_state_dict)} layers from pretrained model")
            
        except Exception as e:
            print(f"⚠ Warning: Could not load pretrained weights: {e}")
            print("Continuing with randomly initialized weights...")
    
    # Freeze a percentage of parameters according to FREEZE_PERCENTAGE
    freeze_ratio = getattr(config, "FREEZE_PERCENTAGE", 0.0)
    if freeze_ratio > 0:
        parameters = list(model.named_parameters())
        total_params = sum(p.numel() for _, p in parameters)
        target_freeze = int(total_params * freeze_ratio)
        
        print(f"\nSelective Finetuning: Targeting {freeze_ratio*100:.0f}% frozen parameters...")
        frozen_count = 0
        for name, param in parameters:
            if frozen_count + param.numel() <= target_freeze:
                param.requires_grad = False
                frozen_count += param.numel()
                print(f"  [FREEZE] {name}: {param.numel()} params")
            else:
                print(f"  [TRAIN]  {name}: {param.numel()} params")
        
        trainable_params = total_params - frozen_count
        print(f"Total parameters: {total_params}")
        print(f"Frozen parameters: {frozen_count} ({frozen_count/total_params*100:.1f}%)")
        print(f"Trainable parameters: {trainable_params} ({trainable_params/total_params*100:.1f}%)")
    
    # Resume from checkpoint if specified (this takes precedence over pretrained weights)
    if config.RESUME_PATH and os.path.exists(config.RESUME_PATH):
        print(f"\nResuming training from checkpoint: {config.RESUME_PATH}")
        checkpoint = torch.load(config.RESUME_PATH, map_location=lambda storage, loc: storage)
        start_epoch = checkpoint["epoch"]
        best_psnr = checkpoint.get("best_psnr", 0.0)
        best_ssim = checkpoint.get("best_ssim", 0.0)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"✓ Resumed from epoch {start_epoch}")
        print(f"  Previous best - PSNR: {best_psnr:.2f}, SSIM: {best_ssim:.4f}")

    
    # Create output directories
    samples_dir = os.path.join(config.SAMPLES_DIR, config.EXP_NAME)
    results_dir = os.path.join(config.RESULTS_DIR, config.EXP_NAME)
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize logger
    csv_log_path = os.path.join(results_dir, config.RESULTS_CSV)
    logger = CSVLogger(csv_log_path, ["epoch", "train_loss", "val_loss", "psnr", "ssim"])
    
    # Initialize gradient scaler for mixed precision
    scaler = torch.amp.GradScaler(device=config.DEVICE.type, enabled=config.DEVICE.type == 'cuda')
    
    # Create IQA models
    psnr_model = PSNR(config.UPSCALE_FACTOR, True).to(device=config.DEVICE, memory_format=torch.channels_last)
    ssim_model = SSIM(config.UPSCALE_FACTOR, True).to(device=config.DEVICE, memory_format=torch.channels_last)
    
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    
    # Training loop
    for epoch in range(start_epoch, config.EPOCHS):
        # Train for one epoch
        train_loss = train(model, train_prefetcher, criterion, optimizer, epoch, scaler, config)
        
        # Validate
        val_loss, psnr, ssim = validate(model, test_prefetcher, epoch, psnr_model, ssim_model, criterion, config, "Test")
        print("\n")
        
        # Log metrics to CSV
        logger.log([epoch + 1, train_loss, val_loss, psnr, ssim])
        
        # Update plots
        save_plots(csv_log_path, results_dir, config)
        
        # Save checkpoint
        is_best = psnr > best_psnr and ssim > best_ssim
        best_psnr = max(psnr, best_psnr)
        best_ssim = max(ssim, best_ssim)
        
        checkpoint = {
            "epoch": epoch + 1,
            "best_psnr": best_psnr,
            "best_ssim": best_ssim,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(samples_dir, f"epoch_{epoch + 1}.pth.tar"))
        
        # Save best model
        if is_best:
            shutil.copyfile(
                os.path.join(samples_dir, f"epoch_{epoch + 1}.pth.tar"),
                os.path.join(results_dir, "best.pth.tar")
            )
            print(f"✓ New best model saved! PSNR: {best_psnr:.2f}, SSIM: {best_ssim:.4f}")
        
        # Save final model
        if (epoch + 1) == config.EPOCHS:
            shutil.copyfile(
                os.path.join(samples_dir, f"epoch_{epoch + 1}.pth.tar"),
                os.path.join(results_dir, "last.pth.tar")
            )
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best PSNR: {best_psnr:.2f}")
    print(f"Best SSIM: {best_ssim:.4f}")
    print(f"Models saved in: {results_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
