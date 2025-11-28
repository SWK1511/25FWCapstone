import os
import sys
import time
import numpy as np

# --------- fmcw 모듈 import를 위한 경로 추가 (노트북에서도 동일) ---------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from fmcw.config import FMCWConfig
from fmcw.pluto_iface import PlutoInterface
from fmcw.waveform import make_chirp
from fmcw.processor import FMCWProcessor


# ============================================================
# (1) IQ 1프레임에서 벽 특징 추출 함수
# ============================================================

def extract_wall_features(iq: np.ndarray):
    """
    IQ 1프레임(복소수 1D)에서 벽(스티로폼)의 특징을 계산.

    반환:
      peak_hz      : 가장 강한 도플러 주파수 (상대 Hz)
      motion_ratio : 0Hz(정지) 이외 에너지 비율
      signal_power : IQ 절댓값 평균 (신호 세기)
    """
    N = len(iq)

    # 간단한 도플러 FFT
    doppler = np.fft.fftshift(np.fft.fft(iq))
    doppler_abs = np.abs(doppler)

    # 도플러 축 (상대 단위, -10 ~ +10 Hz 구간으로 가정)
    doppler_freqs = np.linspace(-10, 10, N)

    # 1) 가장 강한 도플러 주파수
    peak_idx = int(np.argmax(doppler_abs))
    peak_hz = float(doppler_freqs[peak_idx])

    # 2) 정지(0Hz) 주변 vs 나머지 → motion_ratio
    zero_idx = N // 2
    band = 3  # 중앙 ±3 bin을 정지 영역으로 봄
    static_power = float(doppler_abs[zero_idx - band : zero_idx + band].sum())
    total_power = float(doppler_abs.sum()) + 1e-9
    motion_power = max(0.0, total_power - static_power)
    motion_ratio = motion_power / total_power

    # 3) 전체 신호 세기
    signal_power = float(np.mean(np.abs(iq)))

    return peak_hz, motion_ratio, signal_power


# ============================================================
# (2) 메인: WALL(스티로폼) 100프레임 수집 + 저장 + 통계
# ============================================================

def main():
    NUM_FRAMES = 100

    # ---- 저장 경로: FWCapstone/data_iq/WALL ----
    project_root = BASE_DIR                    # .../FWCapstone
    save_dir = os.path.join(project_root, "data_iq", "WALL")
    os.makedirs(save_dir, exist_ok=True)

    print("\n========== WALL (스티로폼) IQ 데이터 수집 시작 ==========")
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
        print("  - 노트북에 Pluto USB/네트워크 연결 상태 확인")
        print("  - fmcw/config.py 의 uri 값을 노트북 기준으로 다시 설정")
        return

    pluto.configure_common(cfg.sample_rate)
    pluto.configure_tx(cfg.fc, cfg.B * 2, cfg.tx_gain)
    pluto.configure_rx(cfg.fc, cfg.B * 2, cfg.rx_buffer_size)

    chirp = make_chirp(cfg)
    proc = FMCWProcessor(cfg)

    # ---- 통계 저장용 리스트 ----
    peak_list = []
    motion_list = []
    power_list = []

    try:
        for i in range(NUM_FRAMES):
            t0 = time.time()

            # 1) 레이더 프레임 수집 (복소수 IQ 2D 배열)
            frame = proc.collect_frame(pluto, chirp)   # shape: [n_chirps, n_samples] (가정)
            iq = frame.reshape(-1)                     # 1D 로 펼쳐서 저장/분석

            # 2) IQ 저장 (.npy)
            save_path = os.path.join(save_dir, f"frame_{i:03d}.npy")
            np.save(save_path, iq)

            # 3) 특징 추출
            peak_hz, motion_ratio, signal_power = extract_wall_features(iq)

            peak_list.append(peak_hz)
            motion_list.append(motion_ratio)
            power_list.append(signal_power)

            dt = (time.time() - t0) * 1000.0  # ms

            # 4) 터미널 출력
            print(
                f"[{i+1:03d}/{NUM_FRAMES}] "
                f"peak_dopp = {peak_hz:+6.2f} Hz | "
                f"motion_ratio = {motion_ratio:5.3f} | "
                f"signal_power = {signal_power:6.4f} | "
                f"{dt:5.1f} ms"
            )

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n[INTERRUPT] 사용자 중단")
    finally:
        pluto.close()

    # ========================================================
    # (3) 최종 통계 요약: 벽(스티로폼)의 대표 파라미터
    # ========================================================
    peak_arr = np.array(peak_list)
    motion_arr = np.array(motion_list)
    power_arr = np.array(power_list)

    print("\n========== WALL (스티로폼) 특징 통계 요약 ==========")
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

    print("\n[해석 예시]")
    print("  • 벽(스티로폼)은 peak_doppler가 0Hz 근처에 몰려 있고,")
    print("  • motion_ratio가 매우 낮게 나와야 정상적인 '정적 표적' 패턴이다.")
    print("  • 이 평균값들을 이후 딥러닝/보고서에서 '벽 파라미터 기준값'으로 사용할 수 있다.")
    print("\n========== WALL 수집 종료 ==========\n")


if __name__ == "__main__":
    main()
