from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import datetime

import tyro

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image

import pytorch_lightning as pl

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


@dataclass
class Args:
    train_roots: List[str]
    val_roots: List[str]

    model_name: str = "swin_v2_t"
    size: int = 512
    batch_size: int = 8
    num_epochs: int = 25
    lr: float = 1e-4
    gradient_clip_val: float = 1.0
    gradient_clip_algorithm: str = "norm"
    num_workers: int = 8
    seed: int = 42

    ckpt_dir: str = "depth_checkpoints"
    no_aug: bool = False


class ThreadDepthDataset(Dataset):
    def __init__(self, roots: List[str], transform: Optional[transforms.Compose]):
        super().__init__()
        self.transform = transform
        self.image_paths: List[str] = []
        self.labels: List[float] = []

        for root in map(Path, roots):
            for img_path in root.iterdir():
                try:
                    depth_val = float(img_path.stem.split("_")[1])
                except (IndexError, ValueError):
                    continue
                self.image_paths.append(str(img_path))
                self.labels.append(depth_val)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = read_image(self.image_paths[idx]).float()
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def build_transforms(size: int, do_aug: bool) -> transforms.Compose:
    base = [
        lambda x: x / 255.0,
        transforms.Resize(
            (size, size), interpolation=transforms.InterpolationMode.BILINEAR
        ),
    ]
    aug = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(
            degrees=45,
            translate=(0.25, 0.25),
            scale=(0.8, 1.2),
            shear=15,
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=1,
        ),
    ]
    return transforms.Compose(base + (aug if do_aug else []))


def get_model(model_name: str) -> nn.Module:
    if model_name == "swin_v2_s":
        model = swin_v2_s(weights=Swin_V2_S_Weights.IMAGENET1K_V1)
        model.head = nn.Linear(768, 1)
    elif model_name == "swin_s":
        model = swin_s(weights=Swin_S_Weights.IMAGENET1K_V1)
        model.head = nn.Linear(768, 1)
    elif model_name == "effnet_b7":
        model = efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
        model.classifier = nn.Linear(2560, 1)
    elif model_name == "swin_v2_t":
        model = swin_v2_t(weights=Swin_V2_T_Weights.IMAGENET1K_V1)
        model.features[0][0] = nn.Conv2d(
            in_channels=3, out_channels=96, kernel_size=(5, 5), stride=(2, 2)
        )
        model.head = nn.Linear(768, 1)
    elif model_name == "densenet201":
        model = densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)
        model.classifier = nn.Linear(1920, 1)
    elif model_name == "googlenet":
        model = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(1024, 1)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model


class DepthRegressionModule(pl.LightningModule):
    def __init__(self, args: Args):
        super().__init__()
        self.save_hyperparameters(vars(args))
        self.args = args

        self.model = get_model(args.model_name)
        self.criterion = nn.MSELoss()

    @staticmethod
    def _metrics(outputs: torch.Tensor, labels: torch.Tensor):
        residuals = torch.abs(outputs - labels)
        mae = residuals.mean()
        frac_le1 = (residuals <= 1.0).float().mean()
        return mae, frac_le1

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        images = images.float()
        labels = labels.unsqueeze(1)

        preds = self(images).exp()
        loss = self.criterion(preds, labels)

        mae, frac_le1 = self._metrics(preds, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_mae", mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train_frac_le1", frac_le1, on_step=False, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images = images.float()
        labels = labels.unsqueeze(1)

        preds = self(images).exp()
        loss = self.criterion(preds, labels)
        mae, frac_le1 = self._metrics(preds, labels)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mae", mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_frac_le1", frac_le1, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.args.lr)


def main():
    args = tyro.cli(Args)

    pl.seed_everything(args.seed, workers=True)

    train_tf = build_transforms(args.size, do_aug=not args.no_aug)
    val_tf = build_transforms(args.size, do_aug=False)

    train_ds = ThreadDepthDataset(args.train_roots, transform=train_tf)
    val_ds = ThreadDepthDataset(args.val_roots, transform=val_tf)

    print(f"Train dataset size: {len(train_ds)}")
    print(f"Val dataset size:   {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    module = DepthRegressionModule(args)

    ckpt_dir = (
        Path(args.ckpt_dir)
        / f"checkpoints-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="model-{epoch:03d}-{val_frac_le1:.5f}",
        monitor="val_frac_le1",
        mode="max",
        save_top_k=10,
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
    )

    print("Starting training…")
    trainer.fit(module, train_loader, val_loader)
    print(f"Checkpoints saved to: {ckpt_dir}")


if __name__ == "__main__":
    main()
