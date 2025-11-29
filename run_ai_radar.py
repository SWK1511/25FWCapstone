import time
import os
import sys
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2

# 모듈 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fmcw.config import FMCWConfig
from fmcw.pluto_iface import PlutoInterface
from fmcw.waveform import make_chirp
from fmcw.processor import FMCWProcessor

# --- 설정 ---
ENGINE_PATH = "radar_unet.trt"

# [감도 조절]
# 아까 빈 방 점수가 -33000 정도였으므로, 그보다 높은 값으로 설정
# 사람이 들어오면 이 값이 급격히 올라갑니다. 테스트하면서 조절하세요.
THRESHOLD = -10000.0 

# ---------------------------------------------------------
# TensorRT 엔진 클래스 (최신 API 호환)
# ---------------------------------------------------------
class TRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        print(f">>> TensorRT 엔진 로딩 중... ({engine_path})")
        
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        self.inputs = []
        self.outputs = []
        
        # 최신 TensorRT API (v10.x 대응)
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(tensor_name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.context.set_tensor_address(tensor_name, int(device_mem))
            
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.inputs.append({'host': host_mem, 'device': device_mem, 'name': tensor_name})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem, 'name': tensor_name})

    def infer(self, img_np):
        np.copyto(self.inputs[0]['host'], img_np.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        return self.outputs[0]['host']

# ---------------------------------------------------------
# 메인 함수
# ---------------------------------------------------------
def main():
    if not os.path.exists(ENGINE_PATH):
        print(f"❌ 에러: {ENGINE_PATH} 파일이 없습니다.")
        return
    ai_model = TRTInference(ENGINE_PATH)

    print(">>> PlutoSDR 연결 중...")
    cfg = FMCWConfig()
    pluto = PlutoInterface(cfg.uri)
    pluto.connect()
    
    pluto.configure_common(cfg.sample_rate)
    pluto.configure_tx(cfg.fc, cfg.B * 2, cfg.tx_gain)
    pluto.configure_rx(cfg.fc, cfg.B * 2, cfg.rx_buffer_size)
    
    chirp = make_chirp(cfg)
    proc = FMCWProcessor(cfg)

    print("\n" + "="*60)
    print("   🚀 AI 레이더 가동 (Ctrl+C로 종료)")
    print("="*60)
    
    try:
        while True:
            t0 = time.time()
            
            # 1. 데이터 수신 & 처리
            frame = proc.collect_frame(pluto, chirp)
            rd_map = proc.doppler_fft(frame)
            
            # 2. AI 입력 전처리 (Resize 128x128)
            rd_db = 20 * np.log10(np.abs(rd_map) + 1e-9)
            rd_norm = rd_db / 150.0
            rd_norm = np.clip(rd_norm, 0, 1)
            rd_resized = cv2.resize(rd_norm, (128, 128))
            input_tensor = rd_resized.astype(np.float32)

            # 3. AI 추론
            output = ai_model.infer(input_tensor)
            
            # 4. 점수 계산
            score = np.sum(output)
            is_human = score > THRESHOLD
            
            # 5. 화면 출력 (깨짐 방지)
            fps = 1.0 / (time.time() - t0)
            
            if is_human:
                msg = f"\033[91m🚨 사람 감지!\033[0m"
            else:
                msg = f"\033[92m   안전 구역  \033[0m"
            
            # 시각적 바 (값 스케일 조정)
            # -30000 ~ 0 범위를 시각화하기 위해 오프셋 적용
            bar_val = int((score + 40000) / 1000) 
            if bar_val < 0: bar_val = 0
            if bar_val > 20: bar_val = 20
            
            bar = "█" * bar_val
            space = " " * (20 - bar_val)

            sys.stdout.write(f"\r{msg} | 점수: {score:8.1f} [{bar}{space}] | FPS: {fps:.1f}    ")
            sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n종료합니다.")
    finally:
        pluto.close()

if __name__ == "__main__":
    main()