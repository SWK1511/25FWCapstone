# dl/training/train_unet_ae.py
import os
import time
import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from dl.dataset import RDDataset
from dl.models.unet_ae import UNetAutoEncoder


def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net AutoEncoder on RD maps")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data_rd/HUMAN",
        help="RD .npy files root dir (예: data_rd/HUMAN)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--save-dir",
        type=str,
        default="dl_ckpt",
        help="모델 저장 폴더",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="디버그용: 사용할 최대 파일 수 (None이면 전체)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    dataset = RDDataset(args.data_dir, max_files=args.max_files)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device = {device}")

    model = UNetAutoEncoder(in_channels=1, base_channels=32).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        t0 = time.time()

        for batch_idx, x in enumerate(loader):
            x = x.to(device)

            optimizer.zero_grad()
            x_hat = model(x)
            loss = criterion(x_hat, x)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"[Epoch {epoch}/{args.epochs}] "
                    f"step {batch_idx+1}/{len(loader)}  "
                    f"loss={loss.item():.4f}"
                )

        epoch_loss = running_loss / len(loader)
        dt = time.time() - t0
        print(
            f"[Epoch {epoch}] mean loss={epoch_loss:.4f}  "
            f"({dt:.1f} s, {len(dataset)} samples)"
        )

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_path = os.path.join(args.save_dir, "unet_ae_best.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_loss,
                },
                save_path,
            )
            print(f"[SAVE] best model updated: {save_path}")

    print("[DONE] training finished.")


if __name__ == "__main__":
    main()