@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0train.ps1" %*
set "TRAIN_EXIT=%ERRORLEVEL%"
echo.
if "%TRAIN_EXIT%"=="0" (
    echo Training finished. Review the evaluation report and model status above.
) else (
    echo Training did not complete. Read the error message above.
)
pause
exit /b %TRAIN_EXIT%
