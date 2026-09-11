@echo off
REM Double-click this to run the Smart Traffic software stack without physical controllers.
REM
REM Any extra arguments are passed through to start.py / the CV node.
REM Examples:
REM   START HERE.bat --source road.mp4
REM   START HERE.bat --no-display
REM
REM The launcher defaults to --no-arduino so the repository runs as software-only.
cd /d "%~dp0"
python "start.py" --no-arduino %*
echo.
echo The system has stopped. Press any key to close this window.
pause > nul
