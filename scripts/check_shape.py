"""본 파일은 npy의 차원수를 확인하는 프로그램임."""

import numpy as np
import os
import glob

# 설정된 저장 경로 (config에 따라 다를 수 있으니 확인 필요)
# 예: FWCapstone/data/human 또는 similar path
sample_dir = "/home/jorin/FWCapstone/radar_dataset/class_0_empty"  # 실제 데이터가 있는 폴더로 경로 수정 필요

# .npy 파일 하나 찾기
files = glob.glob(os.path.join(sample_dir, "*.npy"))

if files:
    data = np.load(files[0])
    print(f"✅ 파일명: {files[0]}")
    print(f"✅ 데이터 타입: {data.dtype}")
    print(f"✅ 데이터 형태(Shape): {data.shape}")
else:
    print("❌ 해당 폴더에 .npy 파일이 없습니다. 경로를 확인해주세요.")