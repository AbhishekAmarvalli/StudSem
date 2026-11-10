@echo off
REM StudSem launcher for Windows cmd / PowerShell.
REM Works even when the "streamlit" command is not on PATH: this runs
REM "python -m streamlit", and picks a Python that actually has Streamlit.
setlocal
cd /d "%~dp0"

set "PYTHON=python"
if exist "venv\Scripts\python.exe" (
  "venv\Scripts\python.exe" -c "import streamlit" >nul 2>&1 && set "PYTHON=venv\Scripts\python.exe"
)

"%PYTHON%" -c "import streamlit" >nul 2>&1
if errorlevel 1 (
  echo Streamlit is not installed for %PYTHON%.
  echo Run:  "%PYTHON%" -m pip install -r requirements.txt
  exit /b 1
)

echo Starting StudSem on http://localhost:8501  (press Ctrl+C to stop)
"%PYTHON%" -m streamlit run app.py %*
