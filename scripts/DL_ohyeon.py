import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# ============================================================
# 1. IQ Dataset 정의
# ============================================================

class IqDataset(Dataset):
    def __init__(self, root_dir: str):
        """
        root_dir 예:
          /home/jorin/FWCapstone/data_iq

        내부 구조:
          data_iq/HUMAN/*.npy
          data_iq/HUMAN_MOVE/*.npy
          data_iq/WALL2/*.npy
        """
        self.samples = []
        self.label_map = {
            "HUMAN": 0,
            "HUMAN_MOVE": 1,
            "WALL2": 2,
        }

        for cls_name, lbl in self.label_map.items():
            cls_dir = os.path.join(root_dir, cls_name)
            paths = sorted(glob.glob(os.path.join(cls_dir, "*.npy")))
            for p in paths:
                self.samples.append((p, lbl))

        if not self.samples:
            raise RuntimeError(f"[ERROR] {root_dir} 아래에서 .npy 파일을 하나도 찾지 못함")

        # 각 클래스 개수 출력
        print("[INFO] 총 샘플 수 :", len(self.samples))
        for cls_name, lbl in self.label_map.items():
            cnt = sum(1 for _, y in self.samples if y == lbl)
            print(f"  - {cls_name}: {cnt}개")

        # 첫 샘플 길이를 기준으로 입력 길이(N) 고정
        first_iq = np.load(self.samples[0][0])
        self.N = len(first_iq)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        iq = np.load(path)  # shape: (N,), dtype: complex

        # 길이 맞추기 (너무 길면 자르고, 짧으면 0 padding)
        if len(iq) > self.N:
            iq = iq[:self.N]
        elif len(iq) < self.N:
            pad = self.N - len(iq)
            iq = np.pad(iq, (0, pad), mode="constant")

        # 복소수 → 2채널 실수 텐서 [2, N] (real, imag)
        x = np.stack([iq.real, iq.imag], axis=0).astype(np.float32)
        x = torch.from_numpy(x)  # [2, N]
        y = torch.tensor(label, dtype=torch.long)
        return x, y


# ============================================================
# 2. 1D CNN 모델 정의
# ============================================================

class RadarNet(nn.Module):
    def __init__(self, num_classes: int = 3, input_len: int = 1024):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),  # N -> N/2

            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),  # N/2 -> N/4

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),  # N/4 -> N/8
        )

        # conv 출력 크기 계산
        dummy = torch.zeros(1, 2, input_len)  # [B=1, C=2, N]
        with torch.no_grad():
            out = self.conv(dummy)
        self.flatten_dim = out.numel()

        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: [B, 2, N]
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ============================================================
# 3. 학습 루프
# ============================================================

def train():
    # -----------------------------------
    # (1) data_iq 경로를 프로젝트 루트 기준으로 자동 설정
    # -----------------------------------
    scripts_dir = os.path.dirname(__file__)          # .../FWCapstone/scripts
    project_root = os.path.dirname(scripts_dir)      # .../FWCapstone
    root_dir = os.path.join(project_root, "data_iq") # .../FWCapstone/data_iq

    print("[DEBUG] data root :", root_dir)

    dataset = IqDataset(root_dir)
    input_len = dataset.N  # IQ 길이

    # -----------------------------------
    # (2) Train / Val 분할
    # -----------------------------------
    total_len = len(dataset)
    val_len = max(1, int(total_len * 0.2))
    train_len = total_len - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    print(f"[INFO] Train: {train_len}개, Val: {val_len}개")

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    # -----------------------------------
    # (3) 모델/손실함수/옵티마이저
    # -----------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 사용 디바이스: {device}")

    model = RadarNet(num_classes=3, input_len=input_len).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 20

    # -----------------------------------
    # (4) 에폭 루프
    # -----------------------------------
    for epoch in range(1, num_epochs + 1):
        # ----- Train -----
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device)  # [B, 2, N]
            y = y.to(device)  # [B]

            optimizer.zero_grad()
            logits = model(x)        # [B, 3]
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # ----- Validation -----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                logits = model(x)
                loss = criterion(logits, y)

                val_loss += loss.item() * x.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"[Epoch {epoch:02d}] "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:5.1f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:5.1f}%")

    # -----------------------------------
    # (5) 모델 저장
    # -----------------------------------
    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", "radar_classifier.pth")
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_len": input_len,
        },
        save_path,
    )

    print(f"\n[INFO] 모델 저장 완료 : {save_path}")
    print("라벨 매핑 :")
    print("  0: HUMAN")
    print("  1: HUMAN_MOVE")
    print("  2: WALL2")


if __name__ == "__main__":
    train()
