@echo off
start "markPriceKlines" cmd /c "cd /d %~dp0 && echo Start WebSocketStream && python WebSocketStream.py && echo End WebSocketStream"
start "markPriceKlines" cmd /c "cd /d %~dp0 && timeout /t 5 /nobreak >nul 2>&1 && echo Start RestAPI && python RestAPI.py && echo End RestAPI"