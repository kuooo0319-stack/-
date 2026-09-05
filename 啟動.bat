@echo off
chcp 65001 >nul
cd /d "%~dp0"
title 文案產生器

set "PYTHON_CMD="

where python >nul 2>nul
if %errorlevel%==0 (
    python --version >nul 2>nul
    if %errorlevel%==0 set "PYTHON_CMD=python"
)

if "%PYTHON_CMD%"=="" (
    where py >nul 2>nul
    if %errorlevel%==0 (
        py --version >nul 2>nul
        if %errorlevel%==0 set "PYTHON_CMD=py"
    )
)

if "%PYTHON_CMD%"=="" (
    echo [錯誤] 找不到 Python，請先雙擊「安裝.bat」完成安裝。
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   這個視窗是伺服器本體，使用網頁的時候請不要關閉這個視窗！
echo   可以縮到最小，但按 X 關閉的話，網頁就會變成「無法連線」。
echo ============================================================
echo.

%PYTHON_CMD% app.py

echo.
echo 伺服器已結束（如果不是你自己關的，把上面的錯誤訊息複製起來回報）。
pause
