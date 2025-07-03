from pathlib import Path
from typing import Tuple, Optional, Iterable, Callable, List, Dict
from collections import OrderedDict
from copy import deepcopy
from itertools import product
from dataclasses import dataclass

import tyro
import cv2
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import (
    swin_s,
    Swin_S_Weights,
    efficientnet_b7,
    EfficientNet_B7_Weights,
    swin_v2_t,
    Swin_V2_T_Weights,
    swin_v2_s,
    Swin_V2_S_Weights,
    densenet201,
    DenseNet201_Weights,
    googlenet,
    GoogLeNet_Weights,
)
from torchvision.models.swin_transformer import Permute
from torchvision.io import read_image
from torchvision import transforms
from torchvision.ops import FeaturePyramidNetwork

from tqdm import tqdm


DATA_ROOT = Path("data/thread/depth")

TRAIN_DIR = DATA_ROOT / "orig_train"
TEST_DIR = DATA_ROOT / "orig_test"
SYNTHETIC_DIR = DATA_ROOT / "synthetic"

CKPT_SAVE_DIR = Path("checkpoints/depth_classification")

CKPT_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Classification parameters
MIN_DEPTH = 1.5
MAX_DEPTH = 9.5
STEP_SIZE = 0.25
NUM_CLASSES = int((MAX_DEPTH - MIN_DEPTH) / STEP_SIZE) + 1


class ThreadDataset(Dataset):
    def __init__(
        self, data_root_dirs: List[str], transform: Optional[nn.Module] = None
    ):
        self.data_root_dirs = [Path(data_root_dir) for data_root_dir in data_root_dirs]

        self.image_paths = []
        self.labels = []
        for data_root_dir in self.data_root_dirs:
            for path in data_root_dir.iterdir():
                image_path = str(path)
                label = float(path.stem.split("_")[1])
                self.image_paths.append(image_path)
                self.labels.append(label)

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = read_image(str(image_path))
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def discretize_depth(
    depth: torch.Tensor,
    min_depth: float = MIN_DEPTH,
    max_depth: float = MAX_DEPTH,
    step: float = STEP_SIZE,
) -> torch.Tensor:
    n_classes = int((max_depth - min_depth) / step) + 1
    depth_bins = torch.linspace(min_depth, max_depth, n_classes, device=depth.device)
    depth_bins[0] = -torch.inf
    depth_bins[-1] = torch.inf

    return torch.bucketize(depth, depth_bins)


def undiscritize_depth(
    depth: torch.Tensor,
    min_depth: float = MIN_DEPTH,
    max_depth: float = MAX_DEPTH,
    step: float = STEP_SIZE,
) -> torch.Tensor:
    n_classes = int((max_depth - min_depth) / step) + 1
    return depth.to(torch.float32) * step + min_depth


class Clahe(nn.Module):
    def __init__(self, clip_limit: float = 10.0):
        super().__init__()
        self.clip_limit = clip_limit
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit)

    def _apply_clahe(self, img: np.ndarray) -> np.ndarray:
        lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab_img[:, :, 0] = self.clahe.apply(lab_img[:, :, 0])
        return cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        x_npy = x.cpu().numpy().transpose(1, 2, 0)
        x_clahe = self._apply_clahe(x_npy).transpose(2, 0, 1)

        return torch.tensor(x_clahe, device=device)


def train_fn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    start_lr: float = 1e-4,
    use_scheduler: bool = False,
    gradient_clip: Optional[float] = None,
):
    best_val_metric = -torch.inf
    best_model = deepcopy(model).cpu()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = F.cross_entropy
    optimizer = optim.Adam(model.parameters(), lr=start_lr)

    if use_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs + 1
        )

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0
        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch [{epoch + 1}/{num_epochs}] Training",
            total=len(train_loader),
        )
        for images, labels in train_pbar:
            images = images.to(device)
            labels = labels.to(device, torch.float32)
            labels_discretized = discretize_depth(
                labels, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, step=STEP_SIZE
            )

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels_discretized)
            loss.backward()

            if gradient_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            optimizer.step()

            running_loss += loss.item()
            train_pbar.set_postfix(loss=running_loss / (train_pbar.n + 1))

        model.eval()

        residuals = []
        eval_pbar = tqdm(
            val_loader,
            desc=f"Epoch [{epoch + 1}/{num_epochs}] Evalutaion",
            total=len(val_loader),
        )
        with torch.no_grad():
            for images, labels in eval_pbar:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                preds = torch.argmax(logits, dim=1).to(torch.float32)
                preds = undiscritize_depth(
                    preds, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, step=STEP_SIZE
                )
                residuals.extend(torch.abs(preds - labels).detach().cpu().tolist())

        residuals = torch.tensor(residuals)
        val_metric = torch.mean(torch.where(residuals <= 1, 1.0, 0.0))

        tqdm.write(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            + f"Validation MAE: {torch.mean(residuals).item():.4f}, "
            + f"Fraction of errors <= 1: {val_metric.item():.4f}, "
            + f"0.9th quantile: {torch.quantile(residuals, 0.9).item():.4f}"
        )

        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_model = deepcopy(model).cpu()

        if use_scheduler:
            scheduler.step()

    model.cpu()
    return best_model, best_val_metric


def get_model(model_name: str) -> nn.Module:
    if model_name == "swin_v2_s":
        model = swin_v2_s(weights=Swin_V2_S_Weights.IMAGENET1K_V1)
        model.head = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, NUM_CLASSES))
    elif model_name == "swin_s":
        model = swin_s(weights=Swin_S_Weights.IMAGENET1K_V1)
        model.head = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, NUM_CLASSES))
    elif model_name == "effnet_b7":
        model = efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
        model.classifier = nn.Sequential(
            nn.Linear(2560, 512), nn.SiLU(), nn.Linear(512, NUM_CLASSES)
        )
    elif model_name == "swin_v2_t":
        model = swin_v2_t(weights=Swin_V2_T_Weights.IMAGENET1K_V1)
        model.features[0][0] = nn.Conv2d(
            in_channels=3, out_channels=96, kernel_size=(5, 5), stride=(2, 2)
        )
        model.head = nn.Sequential(nn.Linear(768, 512), nn.GELU(), nn.Linear(512, NUM_CLASSES))
    elif model_name == "densenet201":
        model = densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)
        model.classifier = nn.Sequential(
            nn.Linear(1920, 512), nn.ReLU(), nn.Linear(512, NUM_CLASSES)
        )
    elif model_name == "googlenet":
        model = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
        model.fc = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, NUM_CLASSES))
    else:
        raise ValueError(f"Model {model_name} not found")
    return model


@dataclass
class Config:
    model_name: str
    gradient_clip: Optional[float] = None
    start_lr: float = 1e-4
    use_scheduler: bool = False
    with_synthetic: bool = False
    with_clahe: bool = False
    with_aug: bool = False
    num_epochs: int = 25
    save_dir: str = "checkpoints/depth_classification"


def main():
    config = tyro.cli(Config)

    base_transform = transforms.Compose(
        [
            lambda x: x / 255,
            transforms.Resize(
                (512, 512), interpolation=transforms.InterpolationMode.BICUBIC
            ),
        ]
    )

    def clahe_transform(transform: nn.Module):
        return transforms.Compose(
            [
                Clahe(),
                transform,
            ]
        )

    def aug_transform(transform: nn.Module):
        return transforms.Compose(
            [
                transform,
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomAffine(
                    degrees=10,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1),
                    shear=10,
                ),
            ]
        )

    model = get_model(config.model_name)
    train_transform = base_transform
    val_transform = base_transform

    if config.with_aug:
        train_transform = aug_transform(train_transform)
    if config.with_clahe:
        train_transform = clahe_transform(train_transform)
        val_transform = clahe_transform(val_transform)

    train_dataset = ThreadDataset(
        [TRAIN_DIR, SYNTHETIC_DIR] if config.with_synthetic else [TRAIN_DIR],
        transform=train_transform,
    )
    val_dataset = ThreadDataset(
        [TEST_DIR],
        transform=val_transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=8)

    best_model, score = train_fn(
        model, 
        train_loader, 
        val_loader, 
        config.num_epochs,
        config.start_lr,
        config.use_scheduler,
        config.gradient_clip
    )

    ckpt_name = f"{config.model_name}_{score:.4f}"
    if config.with_synthetic:
        ckpt_name += "_synthetic"
    if config.with_aug:
        ckpt_name += "_aug"
    if config.with_clahe:
        ckpt_name += "_clahe"
    if config.gradient_clip is not None:
        ckpt_name += f"_clip{config.gradient_clip}"
    if config.start_lr != 1e-4:
        ckpt_name += f"_lr{config.start_lr}"
    if config.use_scheduler:
        ckpt_name += "_scheduler"

    ckpt_name += f"_{config.num_epochs}e"
    ckpt_name += f".pt"

    save_path = Path(config.save_dir) / ckpt_name
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(best_model.state_dict(), save_path)


if __name__ == "__main__":
    main()