import os, json, math, random, argparse, time
from datetime import datetime
from glob import glob
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

# -------------------------
# Config
# -------------------------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data_iq", type=str)
    ap.add_argument("--classes", nargs="+", default=["WALL", "WALL_HUMAN"])
    ap.add_argument("--fft_len", type=int, default=4096)          # 스펙트럼 해상도
    ap.add_argument("--win_len", type=int, default=4096)          # 입력 IQ 길이(부족하면 zero-pad, 넘치면 중앙자르기)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=0)         # Jetson/RPi면 0 권장
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_amp", action="store_true")
    ap.add_argument("--cache_dir", default="cache", type=str)     # 중간산출물 저장(선택)
    ap.add_argument("--runs_dir", default="runs", type=str)
    ap.add_argument("--val_ratio", type=float, default=0.2)       # 세션 단위 분할
    return ap.parse_args()

# -------------------------
# Utils
# -------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def list_session_files(root, cls):
    # e.g., data_iq/WALL/<세션폴더>/frame_000.npy
    sess_dirs = sorted([d for d in glob(os.path.join(root, cls, "*")) if os.path.isdir(d)])
    sessions = []
    for sd in sess_dirs:
        files = sorted(glob(os.path.join(sd, "*.npy")))
        if files: sessions.append(files)
    return sessions  # list of list

def central_fix_len(x, L):
    # x: 1D complex np.ndarray
    n = x.shape[0]
    if n == L:
        return x
    if n > L:
        start = (n - L)//2
        return x[start:start+L]
    # pad
    pad = L - n
    left = pad//2
    right = pad - left
    return np.pad(x, (left, right), mode="constant")

def mag_spec(iq, fft_len=4096):
    # iq: 1D complex
    spec = np.fft.fftshift(np.fft.fft(iq, n=fft_len))
    mag = np.abs(spec)
    mag = mag / (mag.mean() + 1e-9)      # 간단 정규화
    # (1, H, W) 형태로 변환 (H=fft_len, W=1 인 세로 스펙트럼) -> CNN 호환 위해 (H, W) reshape
    mag = mag.astype(np.float32)
    # 길이가 너무 길면 세로 strip -> (H, 64)로 타일링
    W = 64
    tile = np.tile(mag[:, None], (1, W))
    return tile  # (H, W)

# -------------------------
# Dataset
# -------------------------
class IQSpectDataset(Dataset):
    def __init__(self, file_list, fft_len=4096, win_len=4096, label=0, cache_dir=None, augment=False):
        self.files = file_list
        self.fft_len = fft_len
        self.win_len = win_len
        self.label = label
        self.cache_dir = cache_dir
        self.augment = augment
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        if self.cache_dir:
            key = f.replace("/", "_")
            cpath = os.path.join(self.cache_dir, f"{key}.npz")
            if os.path.exists(cpath):
                arr = np.load(cpath)["arr"]
                x = arr
            else:
                iq = np.load(f)  # complex128/complex64 예상
                iq = iq.astype(np.complex64)
                iq = central_fix_len(iq, self.win_len)
                x = mag_spec(iq, self.fft_len)   # (H, W)
                np.savez_compressed(cpath, arr=x)
        else:
            iq = np.load(f)
            iq = iq.astype(np.complex64)
            iq = central_fix_len(iq, self.win_len)
            x = mag_spec(iq, self.fft_len)

        # 간단 증강(선택): 작은 가우시안 노이즈
        if self.augment:
            x = x + np.random.normal(0, 0.02, size=x.shape).astype(np.float32)

        x = x[None, ...]               # (1, H, W)
        y = np.int64(self.label)
        return torch.from_numpy(x), torch.tensor(y)

# -------------------------
# Model: 간단 2D CNN
# -------------------------
class SmallSpecCNN(nn.Module):
    def __init__(self, in_ch=1, ncls=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8, 128), nn.ReLU(),
            nn.Linear(128, ncls)
        )
    def forward(self, x):
        return self.fc(self.net(x))

# -------------------------
# Train / Eval
# -------------------------
def split_by_session(sessA, sessB, val_ratio=0.2, seed=42):
    random.seed(seed)
    def split(sess_list):
        n = len(sess_list)
        k = max(1, int(math.ceil(n*val_ratio)))
        val_idx = set(random.sample(range(n), k))
        tr, va = [], []
        for i, files in enumerate(sess_list):
            (va if i in val_idx else tr).extend(files)
        return tr, va
    trA, vaA = split(sessA)
    trB, vaB = split(sessB)
    return trA, vaA, trB, vaB

def main():
    args = get_args(); set_seed(args.seed)
    # 세션 수집
    sess_wall = list_session_files(args.data_root, args.classes[0])
    sess_human = list_session_files(args.data_root, args.classes[1])
    assert sess_wall and sess_human, "WALL / WALL_HUMAN 데이터가 필요합니다."

    trA, vaA, trB, vaB = split_by_session(sess_wall, sess_human, args.val_ratio, args.seed)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.runs_dir, run_id); os.makedirs(run_dir, exist_ok=True)
    cacheA = os.path.join(args.cache_dir, "WALL"); cacheB = os.path.join(args.cache_dir, "WALL_HUMAN")

    # 파일 목록 저장(감사 추적)
    open(os.path.join(run_dir, "train_files.txt"), "w", encoding="utf-8").write("\n".join(trA+trB))
    open(os.path.join(run_dir, "val_files.txt"), "w", encoding="utf-8").write("\n".join(vaA+vaB))
    json.dump(vars(args), open(os.path.join(run_dir, "cfg.json"), "w"), ensure_ascii=False, indent=2)

    # Dataset/DataLoader
    ds_tr = torch.utils.data.ConcatDataset([
        IQSpectDataset(trA, args.fft_len, args.win_len, 0, cacheA, augment=True),
        IQSpectDataset(trB, args.fft_len, args.win_len, 1, cacheB, augment=True),
    ])
    ds_va = torch.utils.data.ConcatDataset([
        IQSpectDataset(vaA, args.fft_len, args.win_len, 0, cacheA, augment=False),
        IQSpectDataset(vaB, args.fft_len, args.win_len, 1, cacheB, augment=False),
    ])
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmallSpecCNN(1, 2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    ce = nn.CrossEntropyLoss()

    best_f1, best_path = -1, os.path.join(run_dir, "model.pt")
    for ep in range(1, args.epochs+1):
        model.train()
        tr_loss = 0.0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                logits = model(x)
                loss = ce(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tr_loss += loss.item()*x.size(0)
        sched.step()

        # Validation
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x, y in dl_va:
                x = x.to(device)
                logits = model(x)
                prob = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
                ys.extend(y.numpy().tolist()); ps.extend(prob.tolist())

        # Metrics
        pred = (np.array(ps) >= 0.5).astype(int)
        acc = accuracy_score(ys, pred)
        f1  = f1_score(ys, pred)
        try:
            auc = roc_auc_score(ys, ps)
        except ValueError:
            auc = float("nan")

        print(f"[{ep:02d}/{args.epochs}] loss={tr_loss/len(ds_tr):.4f} acc={acc:.3f} f1={f1:.3f} auc={auc:.3f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save({"model": model.state_dict(), "cfg": vars(args)}, best_path)

    # 최종 평가 및 지표 저장
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"]); model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in dl_va:
            x = x.to(device)
            prob = torch.softmax(model(x), dim=1)[:,1].cpu().numpy()
            ys.extend(y.numpy().tolist()); ps.extend(prob.tolist())
    pred = (np.array(ps) >= 0.5).astype(int)
    cm = confusion_matrix(ys, pred).tolist()
    metrics = {
        "best_f1": float(best_f1),
        "acc": float(accuracy_score(ys, pred)),
        "f1": float(f1_score(ys, pred)),
        "auc": float(roc_auc_score(ys, ps)) if len(set(ys))>1 else None,
        "confusion_matrix": cm
    }
    json.dump(metrics, open(os.path.join(run_dir, "metrics.json"), "w"), indent=2, ensure_ascii=False)

    # ONNX 내보내기(배포용)
    dummy = torch.randn(1,1,args.fft_len,64, device=device)
    onnx_dir = os.path.join(run_dir, "onnx"); os.makedirs(onnx_dir, exist_ok=True)
    torch.onnx.export(model, dummy, os.path.join(onnx_dir, "model.onnx"),
                      input_names=["x"], output_names=["logits"], opset_version=12)
    print(f"[Done] best model saved to: {best_path} | metrics: {metrics}")

if __name__ == "__main__":
    main()
