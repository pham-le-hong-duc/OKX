@echo off
echo Starting all modules...

REM Start fundingRate module
start "fundingRate" cmd /c "cd /d %~dp0\fundingRate && app.bat"

REM Wait 2 seconds before starting next module
timeout /t 2 /nobreak > nul

REM Start indexPriceKlines module
start "indexPriceKlines" cmd /c "cd /d %~dp0\indexPriceKlines && app.bat"

REM Wait 2 seconds before starting next module  
timeout /t 2 /nobreak > nul

REM Start markPriceKlines module  
start "markPriceKlines" cmd /c "cd /d %~dp0\markPriceKlines && app.bat"

REM Wait 2 seconds before starting next module
timeout /t 2 /nobreak > nul

REM Start trades module
start "trades" cmd /c "cd /d %~dp0\trades && app.bat"

REM Wait 2 seconds before starting next module
timeout /t 2 /nobreak > nul

REM Start orderBook module
start "orderBook" cmd /c "cd /d %~dp0\orderBook && app.bat"

echo All modules started!
pause