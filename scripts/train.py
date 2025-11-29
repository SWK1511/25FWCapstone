import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split

# ==========================================
# 1. 설정 (Configuration) - 경로 확인 필수!
# ==========================================
# 사용자 경로 기반으로 설정 (Empty 경로는 확인됨)
EMPTY_DIR = "/home/jorin/FWCapstone/radar_dataset/class_0_empty"

# [!] 중요: Human 데이터가 있는 폴더 이름을 확인해서 아래를 수정하세요!
# 예: .../class_1_human 또는 .../human 등
HUMAN_DIR = "/home/jorin/FWCapstone/radar_dataset/class_1_human" 

INPUT_SIZE = 4096    # 확인된 Shape
BATCH_SIZE = 16
LEARNING_RATE = 0.0001 # 학습률을 조금 낮춰서 안정적으로 학습
EPOCHS = 30

# 젯슨 오린 나노 GPU(cuda) 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 사용하는 장치: {device}")

# ==========================================
# 2. 데이터셋 클래스 (전처리 포함)
# ==========================================
class RadarDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # 1. .npy 파일 로드 (complex128)
        data = np.load(self.file_paths[idx])
        
        # 2. 전처리: 복소수 -> 실수 변환 (절댓값 사용)
        # 딥러닝은 복소수를 바로 못 다루므로 신호의 강도(Magnitude)만 취합니다.
        data = np.abs(data) 
        
        # 3. 정규화 (Normalization) - 선택 사항이지만 학습에 도움됨
        # 데이터가 너무 크면 log를 씌우거나 최대값으로 나눔. 
        # 여기서는 간단하게 float32 변환만 수행 (필요 시 수정)
        data = data.astype(np.float32)
        
        return torch.from_numpy(data), torch.tensor(self.labels[idx], dtype=torch.float32)

# ==========================================
# 3. 데이터 로드 및 분할
# ==========================================
print("📂 데이터 파일 검색 중...")

empty_files = glob.glob(os.path.join(EMPTY_DIR, "*.npy"))
human_files = glob.glob(os.path.join(HUMAN_DIR, "*.npy"))

print(f"   - Empty 파일: {len(empty_files)}개 발견")
print(f"   - Human 파일: {len(human_files)}개 발견")

if len(empty_files) == 0 or len(human_files) == 0:
    print("\n❌ 오류: 데이터를 찾을 수 없습니다.")
    print(f"   - 확인한 Empty 경로: {EMPTY_DIR}")
    print(f"   - 확인한 Human 경로: {HUMAN_DIR}")
    print("👉 폴더 경로가 맞는지 다시 한번 확인해주세요.")
    exit()

# 전체 파일 리스트와 라벨 합치기 (Empty=0, Human=1)
all_files = empty_files + human_files
all_labels = [0] * len(empty_files) + [1] * len(human_files)

# 학습용(80%) / 검증용(20%) 분리
X_train, X_val, y_train, y_val = train_test_split(
    all_files, all_labels, test_size=0.2, random_state=42, shuffle=True
)

train_dataset = RadarDataset(X_train, y_train)
val_dataset = RadarDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 4. 모델 정의 (간단한 MLP)
# ==========================================
class SimpleRadarNet(nn.Module):
    def __init__(self, input_size):
        super(SimpleRadarNet, self).__init__()
        # 입력(4096) -> 은닉층 -> 출력(1)
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2), # 과적합 방지
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() # 0~1 사이 확률 출력
        )

    def forward(self, x):
        return self.net(x)

model = SimpleRadarNet(INPUT_SIZE).to(device)

# ==========================================
# 5. 학습 루프
# ==========================================
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\n🔥 학습 시작...")
print("-" * 50)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.unsqueeze(1) # 차원 맞추기

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    # 검증 (Validation)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.unsqueeze(1)
            
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {acc:.2f}%")

# ==========================================
# 6. 모델 저장
# ==========================================
# scripts 폴더 말고 프로젝트 루트나 models 폴더에 저장하는 것이 관리하기 편함
save_path = "radar_model.pth"
torch.save(model.state_dict(), save_path)
print("-" * 50)
print(f"💾 학습 완료! 모델이 저장되었습니다: {save_path}")