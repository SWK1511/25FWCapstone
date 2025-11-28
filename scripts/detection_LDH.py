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
    print("=== FMCW Radar Real-time Detection Test (Fixed Range) ===")
    print("-> 거리 설정을 7.5m로 확장하여 사람을 탐지합니다.")
    print("------------------------------------------------------------")

    # 1. 하드웨어 설정 초기화
    cfg = FMCWConfig()
    
    # [핵심 수정 1] 대역폭을 50MHz -> 10MHz로 줄여야 멀리(7.5m) 보입니다.
    cfg.B = 25e6          # 10 MHz (최대 거리 7.5m)
    
    # [설정 유지] 
    cfg.num_chirps = 128  # 걷는 속도 감지용
    cfg.fft_size = 1024
    cfg.rx_buffer_size = 128 * 1024
    
    pluto = PlutoInterface(cfg.uri)
    pluto.connect()
    pluto.configure_common(cfg.sample_rate)
    pluto.configure_tx(cfg.fc, cfg.B * 2, cfg.tx_gain)
    pluto.configure_rx(cfg.fc, cfg.B * 2, cfg.rx_buffer_size)

    chirp = make_chirp(cfg)
    proc = FMCWProcessor(cfg)

    print(f"\n[INFO] Radar Started! (Bandwidth: {cfg.B/1e6} MHz, Max Range: ~7.5m)")
    print("[INFO] Press Ctrl+C to stop.\n")

    try:
        frame_count = 0
        while True:
            t0 = time.time()
            frame_count += 1

            # 1) 프레임 수집
            try:
                frame = proc.collect_frame(pluto, chirp)
            except OSError as e:
                print(f"[ERROR] Pluto connection lost: {e}")
                break

            # 2) Doppler FFT
            rd_map = proc.doppler_fft(frame)

            # -----------------------------------------------------
            # [튜닝된 감지 로직]
            # -----------------------------------------------------
            is_detected, energy, threshold, _ = proc.detect_human_presence(
                rd_map,
                range_gate_m=(0.5, 7.0),  # 이제 7m까지 데이터가 유효합니다.
                
                # [핵심 수정 2] 걷는 속도도 잡도록 필터 완화 (2 -> 1)
                doppler_exclude_bins=1,   
                
                # [핵심 수정 3] 초기 테스트를 위해 민감도 확보 (1.5 -> 1.3)
                # 너무 예민하면 1.5로 올리세요.
                threshold_scale=1.5       
            )

            # 상태 메시지
            if is_detected:
                status_msg = "🚨 움직임 감지됨! (Moving)"
            else:
                status_msg = "   정지 상태 (Static)"

            dt = time.time() - t0
            
            print(f"[{frame_count}] {status_msg} | E:{energy:.1f} / TH:{threshold:.1f} | Loop: {dt*1000:.1f} ms")

    except KeyboardInterrupt:
        print("\n[STOP] 테스트를 종료합니다.")
    
    finally:
        pluto.close()
        plt.close()
        print("Done.")

if __name__ == "__main__":
    main()