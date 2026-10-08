@echo off
set TARGET=%1
if "%TARGET%"=="" set TARGET=help

if "%TARGET%"=="help" (
    echo Available commands:
    echo   make setup   - Install project dependencies
    echo   make data    - Ingest, validate, clean raw NASA battery data
    echo   make train   - Train VAE anomaly detector and temperature predictor
    echo   make test    - Run test suite via pytest
    echo   make app     - Launch Streamlit interactive dashboard
    echo   make report  - Generate Model Card and Auto-Heal reports
    exit /b 0
)

if "%TARGET%"=="setup" (
    python -m pip install -r requirements.txt
    exit /b %ERRORLEVEL%
)

if "%TARGET%"=="data" (
    python scripts/build_dataset.py
    exit /b %ERRORLEVEL%
)

if "%TARGET%"=="train" (
    python scripts/train_vae.py
    if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
    python scripts/train_temp.py
    exit /b %ERRORLEVEL%
)

if "%TARGET%"=="test" (
    pytest tests/ -v
    exit /b %ERRORLEVEL%
)

if "%TARGET%"=="app" (
    streamlit run app/Home.py
    exit /b %ERRORLEVEL%
)

if "%TARGET%"=="report" (
    python tools/make_model_card.py
    if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
    python tools/make_autoheal_report.py
    exit /b %ERRORLEVEL%
)

if "%TARGET%"=="colab-pack" (
    python scripts/make_colab_pack.py
    exit /b %ERRORLEVEL%
)

echo Unknown target: %TARGET%
exit /b 1
