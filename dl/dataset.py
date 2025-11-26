# dl/dataset.py
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class RDDataset(Dataset):
    """
    RD map (.npy)들을 읽어서 (1, H, W) 텐서로 반환하는 Dataset.
    root_dir 아래의 모든 .npy 파일을 재귀적으로 스캔.
    """

    def __init__(self, root_dir: str, max_files: int | None = None):
        self.root_dir = root_dir
        pattern = os.path.join(root_dir, "**", "*.npy")
        self.files = sorted(glob.glob(pattern, recursive=True))

        if not self.files:
            raise RuntimeError(f"No .npy files found under: {root_dir}")

        if max_files is not None:
            self.files = self.files[:max_files]

        print(f"[RDDataset] {len(self.files)} files loaded from {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        arr = np.load(path).astype(np.float32)

        # (H, W, C) 형태면 첫 채널만 사용
        if arr.ndim == 3:
            arr = arr[..., 0]

        # 간단 정규화
        mean = arr.mean()
        std = arr.std()
        if std > 0:
            arr = (arr - mean) / std

        # (1, H, W)
        arr = np.expand_dims(arr, axis=0)

        return torch.from_numpy(arr)