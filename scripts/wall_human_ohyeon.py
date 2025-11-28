import os
import sys
import time
import csv
from datetime import datetime
import numpy as np

# --------- fmcw 모듈 import 경로 추가 (노트북/라즈파이 공통) ---------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from fmcw.config import FMCWConfig
from fmcw.pluto_iface import PlutoInterface
from fmcw.waveform import make_chirp
from fmcw.processor import FMCWProcessor


# ============================================================
# (1) IQ 1프레임 특징 추출 (벽+사람 환경에도 공통 사용)
# ============================================================
def extract_features(iq: np.ndarray):
    """
    IQ 1프레임(복소 1D)에서 간단 도플러 스펙트럼 기반 특징 추출.

    반환:
      peak_hz      : 가장 강한 도플러 주파수 (상대 Hz)
      motion_ratio : 0Hz(정지) 이외 에너지 비율
      signal_power : IQ 절댓값 평균 (신호 세기)
    """
    N = len(iq)

    # 도플러 FFT
    doppler = np.fft.fftshift(np.fft.fft(iq))
    doppler_abs = np.abs(doppler)

    # 도플러 축 (상대 단위, -10~+10 Hz를 N 포인트로 매핑한다고 가정)
    doppler_freqs = np.linspace(-10, 10, N)

    # 1) 가장 강한 도플러 주파수
    peak_idx = int(np.argmax(doppler_abs))
    peak_hz = float(doppler_freqs[peak_idx])

    # 2) 정지(0Hz) 주변 vs 나머지 → motion_ratio
    zero_idx = N // 2
    band = 3  # 중앙 ±3 bin을 정지 영역으로 간주
    static_power = float(doppler_abs[zero_idx - band : zero_idx + band].sum())
    total_power = float(doppler_abs.sum()) + 1e-9
    motion_power = max(0.0, total_power - static_power)
    motion_ratio = motion_power / total_power

    # 3) 전체 신호 세기
    signal_power = float(np.mean(np.abs(iq)))

    return peak_hz, motion_ratio, signal_power


# ============================================================
# (2) 메인: WALL_HUMAN (벽 뒤 사람) 수집 + 저장 + 통계
# ============================================================
def main():
    NUM_FRAMES = 100  # 필요 시 조정

    # ---- 저장 경로: FWCapstone/data_iq/WALL_HUMAN/yyyymmdd_HHMMSS ----
    project_root = BASE_DIR
    base_dir = os.path.join(project_root, "data_iq", "WALL_HUMAN")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(base_dir, run_id)
    os.makedirs(save_dir, exist_ok=True)

    print("\n========== WALL_HUMAN (벽 뒤 사람) IQ 데이터 수집 시작 ==========")
    print(f"[INFO] 저장 경로 : {save_dir}")
    print(f"[INFO] 총 프레임 수 : {NUM_FRAMES}\n")

    # ---- FMCW / Pluto 초기화 ----
    cfg = FMCWConfig()
    print(f"[INFO] 사용 URI : {cfg.uri}")

    pluto = PlutoInterface(cfg.uri)
    try:
        pluto.connect()
    except Exception as e:
        print(f"[ERROR] Pluto 연결 실패 : {e}")
        print("  - Pluto USB/네트워크 연결 확인")
        print("  - fmcw/config.py 의 uri 값을 환경에 맞게 설정")
        return

    pluto.configure_common(cfg.sample_rate)
    pluto.configure_tx(cfg.fc, cfg.B * 2, cfg.tx_gain)
    pluto.configure_rx(cfg.fc, cfg.B * 2, cfg.rx_buffer_size)

    chirp = make_chirp(cfg)
    proc = FMCWProcessor(cfg)

    # ---- 통계 및 로그용 ----
    peak_list, motion_list, power_list = [], [], []

    # 특징 로그 CSV
    csv_path = os.path.join(save_dir, "features_wall_human.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["frame_idx", "timestamp", "peak_hz", "motion_ratio", "signal_power", "n_iq"])

        try:
            for i in range(NUM_FRAMES):
                t0 = time.time()

                # 1) 레이더 프레임 수집
                frame = proc.collect_frame(pluto, chirp)   # shape: [n_chirps, n_samples] (가정)
                iq = frame.reshape(-1)                     # 1D로 변환

                # 2) IQ 저장 (.npy)
                npy_path = os.path.join(save_dir, f"frame_{i:03d}.npy")
                np.save(npy_path, iq)

                # 3) 특징 추출
                peak_hz, motion_ratio, signal_power = extract_features(iq)
                peak_list.append(peak_hz)
                motion_list.append(motion_ratio)
                power_list.append(signal_power)

                # 4) CSV 로그 기록
                ts = datetime.now().isoformat(timespec="seconds")
                writer.writerow([i, ts, f"{peak_hz:.6f}", f"{motion_ratio:.6f}", f"{signal_power:.6f}", int(iq.size)])

                # 5) 콘솔 출력
                dt_ms = (time.time() - t0) * 1000.0
                print(
                    f"[{i+1:03d}/{NUM_FRAMES}] "
                    f"peak_dopp = {peak_hz:+6.2f} Hz | "
                    f"motion_ratio = {motion_ratio:5.3f} | "
                    f"signal_power = {signal_power:6.4f} | "
                    f"{dt_ms:5.1f} ms"
                )

                # 수집 안정화를 위해 약간의 간격
                time.sleep(0.02)

        except KeyboardInterrupt:
            print("\n[INTERRUPT] 사용자 중단")
        finally:
            pluto.close()

    # ========================================================
    # (3) 최종 통계 요약
    # ========================================================
    peak_arr = np.array(peak_list) if peak_list else np.array([0.0])
    motion_arr = np.array(motion_list) if motion_list else np.array([0.0])
    power_arr = np.array(power_list) if power_list else np.array([0.0])

    print("\n========== WALL_HUMAN 특징 통계 요약 ==========")
    print(
        f"- 평균 peak_doppler  : {peak_arr.mean():+7.3f} Hz "
        f"(표준편차 {peak_arr.std():.3f})"
    )
    print(
        f"- 평균 motion_ratio  : {motion_arr.mean():7.4f} "
        f"(표준편차 {motion_arr.std():.4f})"
    )
    print(
        f"- 평균 signal_power  : {power_arr.mean():7.4f} "
        f"(표준편차 {power_arr.std():.4f})"
    )
    print(f"\n[파일] IQ: {save_dir}/frame_###.npy, 특징 로그: {csv_path}")
    print("========== WALL_HUMAN 수집 종료 ==========\n")


if __name__ == "__main__":
    main()
