"""
Training script for PhysFormerLite on the UBFC-rPPG dataset.

Usage
-----
python src/training/train_rppg.py \
    --data /path/to/ubfc-rppg \
    --epochs 50 \
    --batch-size 4 \
    --lr 1e-4 \
    --output src/models/weights/physformer_lite.pth
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Allow running from repo root: python src/training/train_rppg.py
_SRC = Path(__file__).parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data.ubfc_dataset import UBFCDataset
from models.physformer_lite import PhysFormerLite
from config import PHYSFORMER_WEIGHTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def neg_pearson_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Negative Pearson correlation loss.

    Both *pred* and *target* have shape (B, T).
    Pearson r is computed per sample; mean over batch is returned.
    """
    pred_mu = pred.mean(dim=-1, keepdim=True)
    tgt_mu = target.mean(dim=-1, keepdim=True)
    pred_c = pred - pred_mu
    tgt_c = target - tgt_mu
    cov = (pred_c * tgt_c).mean(dim=-1)
    std = (pred_c.std(dim=-1) * tgt_c.std(dim=-1)).clamp(min=1e-8)
    r = cov / std
    return -r.mean()


def train(
    data_root: str,
    epochs: int = 50,
    batch_size: int = 4,
    lr: float = 1e-4,
    val_split: float = 0.1,
    output_path: Path = PHYSFORMER_WEIGHTS,
    device_str: str = "cpu",
) -> None:
    device = torch.device(device_str)
    logger.info("Training PhysFormerLite on %s  (device=%s)", data_root, device)

    dataset = UBFCDataset(data_root, augment=True)
    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    model = PhysFormerLite().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for rgb, bvp in train_loader:
            rgb = rgb.to(device)   # (B, T, 3)
            bvp = bvp.to(device)   # (B, T)
            optimizer.zero_grad()
            pred = model(rgb)
            loss = neg_pearson_loss(pred, bvp)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for rgb, bvp in val_loader:
                rgb, bvp = rgb.to(device), bvp.to(device)
                pred = model(rgb)
                val_loss += neg_pearson_loss(pred, bvp).item()
        val_loss /= len(val_loader)

        logger.info(
            "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f",
            epoch, epochs, train_loss, val_loss,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_path)
            logger.info("  → Saved best model to %s", output_path)

    logger.info("Training complete. Best val loss: %.4f", best_val_loss)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PhysFormerLite on UBFC-rPPG")
    parser.add_argument("--data", required=True, help="Path to UBFC-rPPG root directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument(
        "--output", default=str(PHYSFORMER_WEIGHTS), help="Output .pth path"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train(
        data_root=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        output_path=Path(args.output),
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
