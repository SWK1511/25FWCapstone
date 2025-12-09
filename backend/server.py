import asyncio
import json
import sys
import os
import time
import gc
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ------------------------------------------------------
# 경로 설정
# ------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.join(current_dir, "scripts")
if scripts_dir not in sys.path:
    sys.path.append(scripts_dir)

# ------------------------------------------------------
# 모듈 임포트
# ------------------------------------------------------
try:
    from cw_logic import MotionDetector as CWRadar
    from fmcw_logic import FMCWDetector as FMCWRadar
    print("✅ 모듈 로드 성공")
except ImportError as e:
    print(f"⚠️ 모듈 로드 실패: {e}")
    CWRadar = None
    FMCWRadar = None

# ------------------------------------------------------
# FastAPI 초기화
# ------------------------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------
# 글로벌 상태
# ------------------------------------------------------
current_radar = None
current_mode = "CW"

# 🔒 모드 변경 중복 방지 락
mode_change_lock = asyncio.Lock()


class ModeRequest(BaseModel):
    mode: str


# ------------------------------------------------------
# 기본 정보
# ------------------------------------------------------
@app.get("/")
def read_root():
    return {"status": "Running", "mode": current_mode}


# ------------------------------------------------------
# 🔥 모드 변경 (async + Lock 적용)
# ------------------------------------------------------
@app.post("/set_mode")
async def set_mode(req: ModeRequest):
    global current_radar, current_mode

    new_mode = req.mode.upper()
    print(f"\n🔄 모드 변경 요청: {current_mode} -> {new_mode}")

    # Lock 진입 (동시 요청 방지)
    async with mode_change_lock:

        # 같은 모드는 변경 필요 없음
        if new_mode == current_mode:
            print("⏸ 이미 해당 모드입니다.")
            return {"status": "Already in this mode"}

        # 🔧 기존 레이더 종료
        if current_radar:
            try:
                current_radar.close()
            except:
                pass

            del current_radar
            current_radar = None
            gc.collect()
            time.sleep(1.5)  # 하드웨어 안정화 시간

        # 🔧 새 모드 생성
        if new_mode == "CW" and CWRadar:
            current_radar = CWRadar()
        elif new_mode == "FMCW" and FMCWRadar:
            current_radar = FMCWRadar()
        else:
            return {"status": "Error", "message": "Module Not Found"}

        # 🔧 하드웨어 연결
        if not current_radar.connect():
            print("❌ 하드웨어 연결 실패")
            return {"status": "Connection Failed"}

        # 🔧 캘리브레이션
        current_radar.calibrate()

        # 모드 갱신
        current_mode = new_mode

        print(f"✔ 모드 변경 완료 → {current_mode}")
        return {"status": "Mode Changed", "current_mode": current_mode}


# ------------------------------------------------------
# 서버 시작 시 CW 레이더 초기화
# ------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    global current_radar, current_mode
    print("\n>>> [System] 서버 시작 (기본: CW)")

    if CWRadar:
        current_radar = CWRadar()
        if current_radar.connect():
            current_radar.calibrate()
            current_mode = "CW"
            print("✔ 기본 CW 모드 준비완료")
        else:
            print("❌ 기본 CW 초기화 실패")


# ------------------------------------------------------
# 서버 종료 시 리소스 해제
# ------------------------------------------------------
@app.on_event("shutdown")
async def shutdown_event():
    global current_radar
    if current_radar:
        try:
            current_radar.close()
        except:
            pass


# ------------------------------------------------------
# WebSocket 실시간 데이터 스트림
# ------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🔌 클라이언트 연결됨")

    try:
        while True:
            if current_radar:
                try:
                    result = current_radar.process_frame()

                    if result:
                        result["current_mode"] = current_mode

                        # CW → probability 계산
                        if current_mode == "CW":
                            score = result.get("score", 0)
                            max_score = result.get("max_score", 20)
                            result["probability"] = min((score / max_score) * 100, 100)

                        # FMCW → ratio 변환
                        elif current_mode == "FMCW":
                            ratio = result.get("ratio", 0)
                            result["probability"] = min(ratio * 100, 100)

                        await websocket.send_text(json.dumps(result))

                    else:
                        await asyncio.sleep(0.05)

                except Exception:
                    await asyncio.sleep(0.1)

            else:
                await asyncio.sleep(0.5)

            await asyncio.sleep(0.03)

    except WebSocketDisconnect:
        print("🔌 연결 끊김")
    except Exception:
        pass