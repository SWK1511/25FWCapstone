프론트엔드를 실행한다.
cd frontend
npm run dev

터미널을 하나 더 켠 후에, 백엔드를 실행한다

cd backend
uvicorn server:app --reload --port 8000 --host 0.0.0.0


창을 확인해본다. 