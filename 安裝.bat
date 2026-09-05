@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo   文案產生器 - 安裝環境
echo ============================================
echo.

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
    echo [錯誤] 找不到可以正常執行的 Python。
    echo.
    echo 如果你剛剛打 python 指令時，Windows 跳出「打開 Microsoft Store」的視窗，
    echo 那是 Windows 內建的假替身，不是真正的 Python，一定要移除或改用下面的方式安裝：
    echo.
    echo 請到 https://www.python.org/downloads/ 下載安裝 Python 3.10 以上版本，
    echo 安裝時務必勾選最下面的「Add Python to PATH」，裝完後重新執行這個檔案。
    pause
    exit /b 1
)

echo 找到 Python：
%PYTHON_CMD% --version
echo.

echo 安裝套件中，請稍候（第一次安裝可能要 1-2 分鐘）...
%PYTHON_CMD% -m pip install --upgrade pip --user
%PYTHON_CMD% -m pip install -r requirements.txt --user
if %errorlevel% neq 0 (
    echo.
    echo [提示] 剛剛用 --user 方式安裝失敗，改試試看一般方式安裝...
    %PYTHON_CMD% -m pip install -r requirements.txt
)

echo.
echo 檢查套件是否安裝成功...
%PYTHON_CMD% -c "import flask, anthropic, openai" 2>nul
if not %errorlevel%==0 (
    echo.
    echo [錯誤] 套件似乎沒有安裝成功，請看上面藍色/白色文字的錯誤訊息。
    echo 常見原因：沒有網路連線、公司電腦網路有限制、或 pip 版本太舊。
    echo 你也可以自己手動打開這個資料夾，執行：
    echo     %PYTHON_CMD% -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo ============================================
echo   安裝完成！接下來請雙擊「啟動.bat」執行程式
echo ============================================
pause
