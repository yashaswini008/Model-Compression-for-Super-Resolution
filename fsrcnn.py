import os
import shutil
import time
import random
import math
import queue
import threading
import csv
from enum import Enum
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.cuda import amp
from torch.backends import cudnn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class Config:
    # Image magnification factor
    upscale_factor = 2
    # Experiment name, easy to save weights and log files
    exp_name = f"fsrcnn_x{upscale_factor}"
    # Use GPU for training by default
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    
    # Dataset paths
    train_image_dir = f"/kaggle/input/datasets/emmanuelgeorgepiiitk/screen-sr-data/Screen-SR-Dataset/Train/HR"
    test_lr_image_dir = f"//kaggle/input/datasets/emmanuelgeorgepiiitk/screen-sr-data/Screen-SR-Dataset/Test/LR"
    test_hr_image_dir = f"/kaggle/input/datasets/emmanuelgeorgepiiitk/screen-sr-data/Screen-SR-Dataset/Test/HR"

    image_size = 20
    batch_size = 16
    num_workers = 4

    # Incremental training
    start_epoch = 0
    resume = ""

    # Total number of epochs
    epochs = 10

    # SGD optimizer parameters
    model_lr = 1e-3
    model_momentum = 0.9
    model_weight_decay = 1e-4
    model_nesterov = False

    print_frequency = 200

    # Logging and Visualization
    results_csv = "results.csv"
    train_loss_plot = "train_loss.png"
    val_loss_plot = "val_loss.png"
    psnr_plot = "psnr.png"
    ssim_plot = "ssim.png"


# ==============================================================================
# Image Processing Utils (from imgproc.py)
# ==============================================================================

def image2tensor(image: np.ndarray, range_norm: bool, half: bool) -> torch.Tensor:
    tensor = F.to_tensor(image)
    if range_norm:
        tensor = tensor.mul_(2.0).sub_(1.0)
    if half:
        tensor = tensor.half()
    return tensor


def cubic(x: Any):
    absx = torch.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * ((absx <= 1).type_as(absx)) + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * (
        ((absx > 1) * (absx <= 2)).type_as(absx))


def calculate_weights_indices(in_length: int, out_length: int, scale: float, kernel_width: int, antialiasing: bool):
    if (scale < 1) and antialiasing:
        kernel_width = kernel_width / scale
    x = torch.linspace(1, out_length, out_length)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = torch.floor(u - kernel_width / 2)
    p = math.ceil(kernel_width) + 2
    indices = left.view(out_length, 1).expand(out_length, p) + torch.linspace(0, p - 1, p).view(1, p).expand(out_length, p)
    distance_to_center = u.view(out_length, 1).expand(out_length, p) - indices
    if (scale < 1) and antialiasing:
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)
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


def imresize(image: Any, scale_factor: float, antialiasing: bool = True) -> Any:
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
    weights_h, indices_h, sym_len_hs, sym_len_he = calculate_weights_indices(in_h, out_h, scale_factor, kernel_width, antialiasing)
    weights_w, indices_w, sym_len_ws, sym_len_we = calculate_weights_indices(in_w, out_w, scale_factor, kernel_width, antialiasing)
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


def bgr2ycbcr(image: np.ndarray, use_y_channel: bool = False) -> np.ndarray:
    if use_y_channel:
        image = np.dot(image, [24.966, 128.553, 65.481]) + 16.0
    else:
        image = np.matmul(image, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]
    image /= 255.
    image = image.astype(np.float32)
    return image


def center_crop(lr_image: np.ndarray, hr_image: np.ndarray, hr_image_size: int, upscale_factor: int) -> [np.ndarray, np.ndarray]:
    hr_image_height, hr_image_width = hr_image.shape[:2]
    hr_top = (hr_image_height - hr_image_size) // 2
    hr_left = (hr_image_width - hr_image_size) // 2
    lr_top = hr_top // upscale_factor
    lr_left = hr_left // upscale_factor
    lr_image_size = hr_image_size // upscale_factor
    patch_lr_image = lr_image[lr_top:lr_top + lr_image_size, lr_left:lr_left + lr_image_size, ...]
    patch_hr_image = hr_image[hr_top:hr_top + hr_image_size, hr_left:hr_left + hr_image_size, ...]
    return patch_lr_image, patch_hr_image


def random_crop(lr_image: np.ndarray, hr_image: np.ndarray, hr_image_size: int, upscale_factor: int) -> [np.ndarray, np.ndarray]:
    hr_image_height, hr_image_width = hr_image.shape[:2]
    hr_top = random.randint(0, hr_image_height - hr_image_size)
    hr_left = random.randint(0, hr_image_width - hr_image_size)
    
    # Crucial: align HR crop coordinates to exact multiples of the upscale_factor 
    # to prevent subsampling phase shift which causes the model to learn blurred translation
    hr_top -= hr_top % upscale_factor
    hr_left -= hr_left % upscale_factor
    
    lr_top = hr_top // upscale_factor
    lr_left = hr_left // upscale_factor
    lr_image_size = hr_image_size // upscale_factor
    patch_lr_image = lr_image[lr_top:lr_top + lr_image_size, lr_left:lr_left + lr_image_size, ...]
    patch_hr_image = hr_image[hr_top:hr_top + hr_image_size, hr_left:hr_left + hr_image_size, ...]
    return patch_lr_image, patch_hr_image


def random_rotate(lr_image: np.ndarray, hr_image: np.ndarray, angles: list, lr_center=None, hr_center=None, scale_factor: float = 1.0) -> [np.ndarray, np.ndarray]:
    lr_image_height, lr_image_width = lr_image.shape[:2]
    hr_image_height, hr_image_width = hr_image.shape[:2]
    if lr_center is None:
        lr_center = (lr_image_width // 2, lr_image_height // 2)
    if hr_center is None:
        hr_center = (hr_image_width // 2, hr_image_height // 2)
    angle = random.choice(angles)
    lr_matrix = cv2.getRotationMatrix2D(lr_center, angle, scale_factor)
    hr_matrix = cv2.getRotationMatrix2D(hr_center, angle, scale_factor)
    rotated_lr_image = cv2.warpAffine(lr_image, lr_matrix, (lr_image_width, lr_image_height))
    rotated_hr_image = cv2.warpAffine(hr_image, hr_matrix, (hr_image_width, hr_image_height))
    return rotated_lr_image, rotated_hr_image


# ==============================================================================
# Dataset Classes (from dataset.py)
# ==============================================================================

class TrainImageDataset(Dataset):
    def __init__(self, image_dir: str, image_size: int, upscale_factor: int) -> None:
        super(TrainImageDataset, self).__init__()
        self.image_file_names = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        self.image_size = image_size
        self.upscale_factor = upscale_factor

    def __getitem__(self, batch_index: int) -> dict:
        hr_image = cv2.imread(self.image_file_names[batch_index], cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        lr_image = imresize(hr_image, 1 / self.upscale_factor)
        # Data augment
        lr_image, hr_image = random_crop(lr_image, hr_image, self.image_size, self.upscale_factor)
        lr_image, hr_image = random_rotate(lr_image, hr_image, angles=[0, 90, 180, 270])
        
        lr_y_image = bgr2ycbcr(lr_image, use_y_channel=True)
        hr_y_image = bgr2ycbcr(hr_image, use_y_channel=True)
        lr_y_tensor = image2tensor(lr_y_image, range_norm=False, half=False)
        hr_y_tensor = image2tensor(hr_y_image, range_norm=False, half=False)
        return {"lr": lr_y_tensor, "hr": hr_y_tensor}

    def __len__(self) -> int:
        return len(self.image_file_names)


class TestImageDataset(Dataset):
    def __init__(self, test_lr_image_dir: str, test_hr_image_dir: str, upscale_factor: int) -> None:
        super(TestImageDataset, self).__init__()
        self.lr_image_file_names = sorted([os.path.join(test_lr_image_dir, x) for x in os.listdir(test_lr_image_dir) if x.endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        self.hr_image_file_names = sorted([os.path.join(test_hr_image_dir, x) for x in os.listdir(test_hr_image_dir) if x.endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        self.upscale_factor = upscale_factor

    def __getitem__(self, batch_index: int) -> dict:
        lr_image = cv2.imread(self.lr_image_file_names[batch_index], cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        hr_image = cv2.imread(self.hr_image_file_names[batch_index], cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        lr_y_image = bgr2ycbcr(lr_image, use_y_channel=True)
        hr_y_image = bgr2ycbcr(hr_image, use_y_channel=True)
        lr_y_tensor = image2tensor(lr_y_image, range_norm=False, half=False)
        hr_y_tensor = image2tensor(hr_y_image, range_norm=False, half=False)
        return {"lr": lr_y_tensor, "hr": hr_y_tensor}

    def __len__(self) -> int:
        return len(self.lr_image_file_names)


class CUDAPrefetcher:
    def __init__(self, dataloader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device
        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None
        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)


# ==============================================================================
# Model Definition (from model.py)
# ==============================================================================

class FSRCNN(nn.Module):
    def __init__(self, upscale_factor: int) -> None:
        super(FSRCNN, self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(1, 56, (5, 5), (1, 1), (2, 2)),
            nn.PReLU(56)
        )
        self.shrink = nn.Sequential(
            nn.Conv2d(56, 12, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(12)
        )
        self.map = nn.Sequential(
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12),
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12),
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12),
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12)
        )
        self.expand = nn.Sequential(
            nn.Conv2d(12, 56, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(56)
        )
        self.deconv = nn.ConvTranspose2d(56, 1, (9, 9), (upscale_factor, upscale_factor), (4, 4), (upscale_factor - 1, upscale_factor - 1))
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.feature_extraction(x)
        out = self.shrink(out)
        out = self.map(out)
        out = self.expand(out)
        out = self.deconv(out)
        return out

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(self.deconv.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.deconv.bias.data)


# ==============================================================================
# Training Logic (from train.py)
# ==============================================================================

# ==============================================================================
# Image Quality Assessment
# ==============================================================================

class PSNR(nn.Module):
    def __init__(self, upscale_factor: int, only_test_y_channel: bool = True):
        super(PSNR, self).__init__()
        self.upscale_factor = upscale_factor
        self.only_test_y_channel = only_test_y_channel

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        crop_border = self.upscale_factor
        if self.only_test_y_channel:
            sr = sr[:, :, crop_border:-crop_border, crop_border:-crop_border]
            hr = hr[:, :, crop_border:-crop_border, crop_border:-crop_border]
        mse = torch.mean((sr - hr) ** 2)
        if mse == 0:
            return torch.tensor(100.0)
        return 10 * torch.log10(1.0 / mse)


class SSIM(nn.Module):
    def __init__(self, upscale_factor: int, only_test_y_channel: bool = True):
        super(SSIM, self).__init__()
        self.upscale_factor = upscale_factor
        self.only_test_y_channel = only_test_y_channel

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        crop_border = self.upscale_factor
        if self.only_test_y_channel:
            sr = sr[:, :, crop_border:-crop_border, crop_border:-crop_border]
            hr = hr[:, :, crop_border:-crop_border, crop_border:-crop_border]
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        mu_sr = torch.mean(sr)
        mu_hr = torch.mean(hr)
        sigma_sr = torch.var(sr)
        sigma_hr = torch.var(hr)
        sigma_sr_hr = torch.mean((sr - mu_sr) * (hr - mu_hr))
        numerator = (2 * mu_sr * mu_hr + C1) * (2 * sigma_sr_hr + C2)
        denominator = (mu_sr ** 2 + mu_hr ** 2 + C1) * (sigma_sr + sigma_hr + C2)
        return numerator / denominator


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
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
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.2f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.2f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.2f}"
        else:
            raise ValueError(f"Invalid summary type {self.summary_type}")
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


class CSVLogger:
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


def save_plots(csv_path: str, output_dir: str):
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

    _plot(epochs, train_losses, "Training Loss", "MSE Loss", Config.train_loss_plot, "blue")
    _plot(epochs, val_losses, "Validation Loss", "MSE Loss", Config.val_loss_plot, "orange")
    _plot(epochs, psnrs, "Validation PSNR", "PSNR (dB)", Config.psnr_plot, "green")
    _plot(epochs, ssims, "Validation SSIM", "SSIM", Config.ssim_plot, "purple")


def load_dataset() -> [CUDAPrefetcher, CUDAPrefetcher]:
    train_datasets = TrainImageDataset(Config.train_image_dir, Config.image_size, Config.upscale_factor)
    test_datasets = TestImageDataset(Config.test_lr_image_dir, Config.test_hr_image_dir, Config.upscale_factor)

    train_dataloader = DataLoader(train_datasets, batch_size=Config.batch_size, shuffle=True,
                                  num_workers=Config.num_workers, pin_memory=True, drop_last=True, persistent_workers=True)
    test_dataloader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=1,
                                 pin_memory=True, drop_last=False, persistent_workers=False)

    if Config.device.type == "cuda":
        train_prefetcher = CUDAPrefetcher(train_dataloader, Config.device)
        test_prefetcher = CUDAPrefetcher(test_dataloader, Config.device)
    else:
        # Fallback for CPU
        class CPUPrefetcher:
            def __init__(self, dataloader):
                self.dataloader = dataloader
                self.data = iter(dataloader)
            def next(self):
                try: return next(self.data)
                except StopIteration: return None
            def reset(self): self.data = iter(self.dataloader)
            def __len__(self): return len(self.dataloader)
        train_prefetcher = CPUPrefetcher(train_dataloader)
        test_prefetcher = CPUPrefetcher(test_dataloader)

    return train_prefetcher, test_prefetcher


def build_model() -> nn.Module:
    model = FSRCNN(Config.upscale_factor).to(Config.device)
    return model


def define_loss() -> nn.MSELoss:
    pixel_criterion = nn.MSELoss().to(Config.device)
    return pixel_criterion


def define_optimizer(model) -> optim.SGD:
    optimizer = optim.SGD([{"params": model.feature_extraction.parameters()},
                           {"params": model.shrink.parameters()},
                           {"params": model.map.parameters()},
                           {"params": model.expand.parameters()},
                           {"params": model.deconv.parameters(), "lr": Config.model_lr * 0.1}],
                          lr=Config.model_lr, momentum=Config.model_momentum,
                          weight_decay=Config.model_weight_decay, nesterov=Config.model_nesterov)
    return optimizer


def train(model, train_prefetcher, pixel_criterion, optimizer, epoch, scaler) -> float:
    batches = len(train_prefetcher)
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.6f")
    progress = ProgressMeter(batches, [batch_time, data_time, losses], prefix=f"Epoch: [{epoch + 1}]")

    model.train()
    batch_index = 0
    end = time.time()
    train_prefetcher.reset()
    batch_data = train_prefetcher.next()
    while batch_data is not None:
        data_time.update(time.time() - end)
        lr = batch_data["lr"].to(Config.device, non_blocking=True)
        hr = batch_data["hr"].to(Config.device, non_blocking=True)
        model.zero_grad()
        with amp.autocast():
            sr = model(lr)
            loss = pixel_criterion(sr, hr)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.update(loss.item(), lr.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_index % Config.print_frequency == 0:
            progress.display(batch_index)
        batch_data = train_prefetcher.next()
        batch_index += 1
    return losses.avg


def validate(model, valid_prefetcher, psnr_model, ssim_model, pixel_criterion, epoch, mode) -> [float, float, float]:
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    losses = AverageMeter("Loss", ":6.6f", Summary.AVERAGE)
    psnres = AverageMeter("PSNR", ":4.2f", Summary.AVERAGE)
    ssimes = AverageMeter("SSIM", ":4.4f", Summary.AVERAGE)
    progress = ProgressMeter(len(valid_prefetcher), [batch_time, losses, psnres, ssimes], prefix=f"{mode}: ")
    model.eval()
    batch_index = 0
    end = time.time()
    with torch.no_grad():
        valid_prefetcher.reset()
        batch_data = valid_prefetcher.next()
        while batch_data is not None:
            lr = batch_data["lr"].to(Config.device, non_blocking=True)
            hr = batch_data["hr"].to(Config.device, non_blocking=True)
            with amp.autocast():
                sr = model(lr)
                loss = pixel_criterion(sr, hr)
            
            # Clamp to [0, 1] range to avoid artificially lowered PSNR/SSIM from small overflows
            sr_clamped = torch.clamp(sr, 0.0, 1.0)
            psnr = psnr_model(sr_clamped, hr)
            ssim = ssim_model(sr_clamped, hr)
            losses.update(loss.item(), lr.size(0))
            psnres.update(psnr.item(), lr.size(0))
            ssimes.update(ssim.item(), lr.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if batch_index % Config.print_frequency == 0:
                progress.display(batch_index)
            batch_data = valid_prefetcher.next()
            batch_index += 1
    progress.display_summary()
    return losses.avg, psnres.avg, ssimes.avg


def main():
    # Random seed to maintain reproducible results
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    cudnn.benchmark = True

    best_psnr = 0.0
    train_prefetcher, test_prefetcher = load_dataset()
    print("Load datasets successfully.")

    model = build_model()
    print("Build FSRCNN model successfully.")

    pixel_criterion = define_loss()
    print("Define loss functions successfully.")

    optimizer = define_optimizer(model)
    print("Define optimizer successfully.")

    if Config.resume:
        checkpoint = torch.load(Config.resume, map_location=lambda storage, loc: storage)
        Config.start_epoch = checkpoint["epoch"]
        best_psnr = checkpoint["best_psnr"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"Loaded pretrained model weights from {Config.resume}.")

    samples_dir = os.path.join("samples", Config.exp_name)
    results_dir = os.path.join("results", Config.exp_name)
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Initialize logger
    csv_log_path = os.path.join(results_dir, Config.results_csv)
    logger = CSVLogger(csv_log_path, ["epoch", "train_loss", "val_loss", "psnr", "ssim"])

    # Create IQA models
    psnr_model = PSNR(Config.upscale_factor, True).to(Config.device)
    ssim_model = SSIM(Config.upscale_factor, True).to(Config.device)

    scaler = amp.GradScaler()

    for epoch in range(Config.start_epoch, Config.epochs):
        train_loss = train(model, train_prefetcher, pixel_criterion, optimizer, epoch, scaler)
        val_loss, psnr, ssim = validate(model, test_prefetcher, psnr_model, ssim_model, pixel_criterion, epoch, "Test")
        print("\n")

        # Log metrics to CSV
        logger.log([epoch + 1, train_loss, val_loss, psnr, ssim])

        # Update plots
        save_plots(csv_log_path, results_dir)

        is_best = psnr > best_psnr
        best_psnr = max(psnr, best_psnr)
        torch.save({"epoch": epoch + 1, "best_psnr": best_psnr, "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "scheduler": None},
                   os.path.join(samples_dir, f"epoch_{epoch + 1}.pth.tar"))
        if is_best:
            shutil.copyfile(os.path.join(samples_dir, f"epoch_{epoch + 1}.pth.tar"), os.path.join(results_dir, "best.pth.tar"))
        if (epoch + 1) == Config.epochs:
            shutil.copyfile(os.path.join(samples_dir, f"epoch_{epoch + 1}.pth.tar"), os.path.join(results_dir, "last.pth.tar"))


if __name__ == "__main__":
    main()
