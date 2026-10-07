@echo off
setlocal
cd /d "%~dp0"
if not exist "frontend\node_modules" (
  echo Installing frontend dependencies...
  cd frontend
  call npm install
  if errorlevel 1 exit /b 1
  cd ..
)
if not exist "frontend\dist\index.html" (
  echo Building React frontend...
  cd frontend
  call npm run build
  if errorlevel 1 exit /b 1
  cd ..
)
echo Starting RippleAI...
python app.py
