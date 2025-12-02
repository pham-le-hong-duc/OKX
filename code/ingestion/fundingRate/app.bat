@echo off
start "fundingRate" cmd /c "cd /d %~dp0 && echo Start WebSocketStream && python WebSocketStream.py && echo End WebSocketStream"
start "fundingRate" cmd /c "cd /d %~dp0 && timeout /t 5 /nobreak >nul 2>&1 && echo Start Download && python Download.py && echo End Download && echo. && echo Start RestAPI && python RestAPI.py && echo End RestAPI"