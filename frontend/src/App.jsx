import { useState, useEffect, useRef } from 'react'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Filler,
  BarElement,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import './App.css'

// 젯슨 IP
const JETSON_IP = "10.204.220.59"; 
const API_URL = `http://${JETSON_IP}:8000`;
const WS_URL = `ws://${JETSON_IP}:8000/ws`;

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Filler);

/**
 * Range–Doppler 2D 맵을 그리는 캔버스 컴포넌트
 * rdMap: [doppler][range] 구조의 2D 배열 (dB 값)
 */
function RangeDopplerCanvas({ rdMap }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!rdMap || rdMap.length === 0) return;
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rows = rdMap.length;        // doppler 방향
    const cols = rdMap[0].length;     // range 방향

    // 캔버스 내부 해상도는 데이터 크기와 동일하게
    canvas.width = cols;
    canvas.height = rows;

    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(cols, rows);

    // dB 값 범위 계산
    let minVal = Infinity;
    let maxVal = -Infinity;
    for (let y = 0; y < rows; y++) {
      const row = rdMap[y];
      for (let x = 0; x < cols; x++) {
        const v = row[x];
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
      }
    }
    if (!isFinite(minVal) || !isFinite(maxVal) || minVal === maxVal) {
      minVal = minVal || 0;
      maxVal = maxVal || minVal + 1;
    }
    const span = maxVal - minVal;

    // 간단한 컬러맵: 어두운 초록 → 밝은 노랑
    const data = imgData.data;
    let i = 0;
    for (let y = 0; y < rows; y++) {
      const row = rdMap[y];
      for (let x = 0; x < cols; x++) {
        const v = row[x];
        const norm = Math.min(1, Math.max(0, (v - minVal) / span)); // 0~1

        // 0~0.5: 검녹 → 초록, 0.5~1: 초록 → 노랑
        let r, g, b;
        if (norm < 0.5) {
          const t = norm / 0.5;       // 0~1
          r = 0;
          g = Math.round(128 + t * 127); // 128~255
          b = 0;
        } else {
          const t = (norm - 0.5) / 0.5; // 0~1
          r = Math.round(t * 255);      // 0~255
          g = 255;
          b = 0;
        }

        data[i++] = r;
        data[i++] = g;
        data[i++] = b;
        data[i++] = 255; // alpha
      }
    }

    ctx.putImageData(imgData, 0, 0);
  }, [rdMap]);

  return (
    <div style={{ width: '100%', height: '100%' }}>
      <canvas
        ref={canvasRef}
        style={{
          width: '100%',
          height: '100%',
          imageRendering: 'pixelated',
          border: '1px solid #333'
        }}
      />
    </div>
  );
}

function App() {
  const [currentMode, setCurrentMode] = useState("CW");
  const [radarData, setRadarData] = useState(null);
  const [cwHistory, setCwHistory] = useState(new Array(300).fill(0)); 
  const [isConnected, setIsConnected] = useState(false);
  const [connectionError, setConnectionError] = useState("");
  
  const ws = useRef(null);
  const lastUpdateRef = useRef(0); // 프론트 업데이트 throttle 용

  // 모드 변경 함수
  const changeMode = async (mode) => {
    if (mode === currentMode) return;
    try {
      console.log(`Sending mode change request to ${API_URL}/set_mode`);
      const response = await fetch(`${API_URL}/set_mode`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: mode })
      });
      
      const res = await response.json();
      console.log("Mode Changed:", res);
      
      setCurrentMode(mode);
      if (mode === "CW") {
        setCwHistory(new Array(300).fill(0));
      }
      setRadarData(null);
    } catch (e) {
      console.error("Mode change failed:", e);
      alert(`모드 변경 실패!\n서버 주소(${API_URL})에 연결할 수 없습니다.`);
    }
  };

  // 웹소켓 연결
  useEffect(() => {
    console.log(`📡 Connecting to WebSocket: ${WS_URL}`);
    
    const connectWS = () => {
      ws.current = new WebSocket(WS_URL);
      
      ws.current.onopen = () => {
        console.log("✅ WebSocket Connected!");
        setIsConnected(true);
        setConnectionError("");
      };

      ws.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          // 서버 모드와 동기화
          if (data.current_mode && data.current_mode !== currentMode) {
            setCurrentMode(data.current_mode);
          }

          // CW 히스토리는 가볍기 때문에 그냥 바로 업데이트
          if (data.current_mode === "CW" || (!data.current_mode && currentMode === "CW")) {
            const val = data.is_detected ? 1 : 0;
            setCwHistory(prev => [...prev.slice(1), val]);
          }

          // 렌더링 부하 줄이기 위해 상태 업데이트는 최대 10~12 FPS로 제한
          const now = performance.now();
          if (now - lastUpdateRef.current > 80) { // 80ms 이상일 때만
            lastUpdateRef.current = now;
            setRadarData(data);
          }
        } catch (e) {
          console.error("Data Parse Error:", e);
        }
      };

      ws.current.onclose = () => {
        console.log("❌ WebSocket Disconnected");
        setIsConnected(false);
        setConnectionError("연결 끊김 (Retrying...)");
        setTimeout(connectWS, 3000);
      };

      ws.current.onerror = (err) => {
        console.error("⚠️ WebSocket Error:", err);
        setConnectionError("연결 오류");
        ws.current.close();
      };
    };
    
    connectWS();
    return () => { if (ws.current) ws.current.close(); };
  }, []); 

  // ---------------- CW 그래프 ----------------
  const cwOptions = {
    responsive: true,
    animation: false,
    maintainAspectRatio: false,
    scales: {
      x: { display: false },
      y: { 
        min: -0.2, max: 1.2, 
        grid: { color: '#333' }, 
        ticks: { 
          color: '#00ff00', stepSize: 1, maxTicksLimit: 2,
          callback: (v) => v===0?'Safe(0)':v===1?'Detected(1)':'' 
        } 
      }
    },
    plugins: { legend: {display:false} },
    elements: { 
      point: {radius:0}, 
      line: {borderWidth:3, borderColor:'#00ff00', tension:0.4} 
    }
  };

  const cwChartData = {
    labels: cwHistory.map((_, i) => i),
    datasets: [{
      fill: true,
      data: cwHistory,
      backgroundColor: (context) => {
        const ctx = context.chart.ctx;
        const gradient = ctx.createLinearGradient(0, 0, 0, 400);
        const isDet = radarData?.is_detected || false;
        
        if (isDet) {
          gradient.addColorStop(0, 'rgba(255, 0, 0, 0.5)');
          gradient.addColorStop(1, 'rgba(255, 0, 0, 0)');
        } else {
          gradient.addColorStop(0, 'rgba(0, 255, 0, 0.2)');
          gradient.addColorStop(1, 'rgba(0, 255, 0, 0)');
        }
        return gradient;
      },
      borderColor: (radarData?.is_detected) ? '#ff0000' : '#00ff00',
    }],
  };

  // ---------------- FMCW 그래프 (1D 프로파일) ----------------
  const fmcwSignalRaw =
    (currentMode === "FMCW" && Array.isArray(radarData?.signal))
      ? radarData.signal
      : [];

  // 포인트 수 줄여서 렉 줄이기 (예: 앞 256개만 사용)
  const MAX_POINTS = 256;
  const fmcwSignal =
    fmcwSignalRaw.length > MAX_POINTS
      ? fmcwSignalRaw.slice(0, MAX_POINTS)
      : fmcwSignalRaw;

  const displaySignal =
    fmcwSignal.length > 0 ? fmcwSignal : new Array(100).fill(0);

  // 데이터 기반 Y축 범위
  let fmcwYMin = 0;
  let fmcwYMax = 120;
  if (fmcwSignal.length > 0) {
    const minVal = Math.min(...fmcwSignal);
    const maxVal = Math.max(...fmcwSignal);
    fmcwYMin = Math.max(0, Math.floor(minVal));
    fmcwYMax = Math.min(120, Math.ceil(maxVal + 5));
    if (fmcwYMax - fmcwYMin < 20) {
      fmcwYMax = fmcwYMin + 20;
    }
  }

  const fmcwOptions = {
    responsive: true,
    animation: false,
    maintainAspectRatio: false,
    scales: {
      x: { 
        display: true, 
        title: {display:true, text:'Distance (Range Bin)', color:'#00ffff'}, 
        grid: {display:false}, 
        ticks: {color:'#00ffff'} 
      },
      y: { 
        min: fmcwYMin, 
        max: fmcwYMax, 
        title: {display:true, text:'Signal Strength (dB)', color:'#00ffff'}, 
        grid: {color:'#333'}, 
        ticks: {color:'#00ffff'} 
      }
    },
    plugins: { legend: {display:false} },
    elements: { point: {radius:0}, line: {borderWidth:2} }
  };

  const fmcwChartData = {
    labels: displaySignal.map((_, i) => i),
    datasets: [{
      type: 'line',
      data: displaySignal,
      borderColor: '#00ffff',
      backgroundColor: 'rgba(0, 255, 255, 0.2)',
      fill: true,
      tension: 0.1
    }]
  };

  // ---------------- Range–Doppler 맵 ----------------
  const rdMap =
    currentMode === "FMCW" &&
    Array.isArray(radarData?.rd_map) &&
    radarData.rd_map.length > 0
      ? radarData.rd_map
      : null;

  // ---------------- 공통 표시용 값 ----------------
  const isDet = radarData?.is_detected || false;
  const peakVal = (radarData?.peak_val !== undefined)
    ? radarData.peak_val.toFixed(1)
    : "0.0";
  
  let percent = 0;
  if (currentMode === "CW") {
    percent = radarData?.probability || 0;
  } else {
    percent = (radarData?.ratio || 0) * 100;
  }

  return (
    <div className={`container ${isDet ? 'alert-mode' : 'safe-mode'}`}>
      <header>
        <h1>LKNKL RADAR - {currentMode} MODE</h1>
        <div className="mode-switch">
          <button
            className={currentMode === "CW" ? "active cw-btn" : "cw-btn"}
            onClick={() => changeMode("CW")}
          >
            CW MODE
          </button>
          <button
            className={currentMode === "FMCW" ? "active fmcw-btn" : "fmcw-btn"}
            onClick={() => changeMode("FMCW")}
          >
            FMCW MODE
          </button>
        </div>
      </header>

      {!isConnected && (
        <div style={{
          backgroundColor: 'red',
          color: 'white',
          padding: '10px',
          width: '100%',
          textAlign: 'center'
        }}>
          ⚠️ 서버 연결 실패 ({connectionError}) - {WS_URL} 확인 필요
        </div>
      )}

      <main>
        <div className="chart-container">
          {currentMode === "CW" ? (
            <Line data={cwChartData} options={cwOptions} />
          ) : (
            <>
              <div style={{ height: '45%' }}>
                <Line data={fmcwChartData} options={fmcwOptions} />
              </div>
              {rdMap && (
                <div style={{ height: '45%', marginTop: 10 }}>
                  <RangeDopplerCanvas rdMap={rdMap} />
                </div>
              )}
            </>
          )}
        </div>

        <div className="info-panel">
          <div className="metric">
            <span className="label">
              {currentMode === "CW" ? "Detection Status" : "Max Signal Strength"}
            </span>
            <span
              className="value"
              style={{
                color: isDet
                  ? '#ff0000'
                  : (currentMode === "CW" ? '#00ff00' : '#00ffff')
              }}
            >
              {currentMode === "CW"
                ? (isDet ? "탐지됨 (Detected)" : "미탐지 (Scanning)")
                : `${peakVal} dB`}
            </span>
            <div className="progress-bar-bg">
              <div
                className="progress-bar-fill"
                style={{
                  width: `${percent}%`,
                  backgroundColor: isDet
                    ? 'red'
                    : (currentMode === "CW" ? '#00ff00' : '#00ffff')
                }}
              />
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}

export default App