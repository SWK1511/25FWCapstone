import time
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 기존 모듈 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fmcw.config import FMCWConfig
from fmcw.pluto_iface import PlutoInterface
from fmcw.waveform import make_chirp
from fmcw.processor import FMCWProcessor

def main():
    # ---------------------------------------------------------
    # 1. 데이터 저장 설정 (사용자 입력)
    # ---------------------------------------------------------
    print("=== FMCW Radar Data Collection Tool ===")
    label_input = input("데이터 라벨을 입력하세요 (0: 빈 방/클러터, 1: 사람/타겟): ")
    
    if label_input == '0':
        save_dir = "./dataset/clutter"
        print(f"-> '[Clutter/Empty]' 모드로 저장합니다.")
    elif label_input == '1':
        save_dir = "./dataset/target"
        print(f"-> '[Target/Human]' 모드로 저장합니다.")
    else:
        print("잘못된 입력입니다. 프로그램을 종료합니다.")
        return

    # 폴더가 없으면 생성
    os.makedirs(save_dir, exist_ok=True)

    # 수집할 프레임 수 설정
    try:
        max_frames = int(input("수집할 프레임 수를 입력하세요 (예: 1000): "))
    except ValueError:
        max_frames = 1000
    
    print(f"-> 총 {max_frames}개의 프레임을 {save_dir} 에 저장합니다.")
    print("------------------------------------------------------------")

    # ---------------------------------------------------------
    # 2. 하드웨어 및 프로세서 초기화
    # ---------------------------------------------------------
    cfg = FMCWConfig()
    
    # 필요시 Config 수정 (속도 최적화)
    cfg.num_chirps = 128 
    cfg.fft_size = 1024
    cfg.rx_buffer_size = 128 * 1024
    
    pluto = PlutoInterface(cfg.uri)
    pluto.connect()
    pluto.configure_common(cfg.sample_rate)
    pluto.configure_tx(cfg.fc, cfg.B * 2, cfg.tx_gain)
    pluto.configure_rx(cfg.fc, cfg.B * 2, cfg.rx_buffer_size)

    chirp = make_chirp(cfg)
    proc = FMCWProcessor(cfg)

    # 시각화 설정 (실시간 확인용 - 현재 주석 처리됨)
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    img_plot = None

    try:
        for i in range(max_frames):
            t0 = time.time()

            # 1) 프레임 수집 (Rx -> Beat Signal -> Range FFT)
            try:
                frame = proc.collect_frame(pluto, chirp)
            except OSError as e:
                print(f"[ERROR] Pluto connection lost: {e}")
                break

            # 2) Doppler FFT 수행 (Range-Doppler Map 생성)
            rd_map = proc.doppler_fft(frame)

            # -----------------------------------------------------
            # [추가된 기능] 사람 움직임 탐지 로직
            # -----------------------------------------------------
            is_detected, energy, threshold, _ = proc.detect_human_presence(
                rd_map,
                range_gate_m=(0.5, 6.0),  # 탐지 거리: 0.5m ~ 6.0m 사이만 감시
                doppler_exclude_bins=2,   # 정지 물체(벽, 책상) 제거를 위해 중앙 속도 성분 무시
                threshold_scale=1.2       # 민감도 (낮을수록 예민함)
            )

            # 상태 메시지 설정
            if is_detected:
                status_msg = "🚨 움직임 감지됨! (Moving)"
            else:
                status_msg = "   정지 상태 (Static)"
            # -----------------------------------------------------
            # 3) 중요: 데이터 저장
            # -----------------------------------------------------
            file_name = os.path.join(save_dir, f"data_{int(time.time()*1000)}.npy")
            np.save(file_name, rd_map)

            # -----------------------------------------------------
            # 4) 실시간 화면 표시 (속도를 위해 주석 처리 유지)
            # -----------------------------------------------------
            """
            if i % 5 == 0:
                rd_map_db = 20 * np.log10(rd_map + 1e-9)
                if img_plot is None:
                    img_plot = ax.imshow(rd_map_db, aspect='auto', cmap='jet', origin='lower', vmin=60, vmax=120)
                    plt.title("Range-Doppler Map (Real-time)")
                    plt.xlabel("Range Bins")
                    plt.ylabel("Doppler Bins")
                    plt.colorbar(img_plot, ax=ax)
                else:
                    img_plot.set_data(rd_map_db)
                plt.pause(0.001)
            """

            dt = time.time() - t0
            
            # 기존 코드
            # print(f"[{i+1}/{max_frames}] {status_msg} | Saved: {file_name} ({dt*1000:.1f} ms)")

            # 수정 코드 (에너지와 임계값 수치를 같이 출력)
            print(f"[{i+1}/{max_frames}] {status_msg} | E:{energy:.1f} / TH:{threshold:.1f} | Saved... ({dt*1000:.1f} ms)")

    except KeyboardInterrupt:
        print("\n[STOP] 수집을 중단합니다.")
    
    finally:
        pluto.close()
        plt.close()
        print("Done.")

if __name__ == "__main__":
    main()