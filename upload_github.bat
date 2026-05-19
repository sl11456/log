@echo off
cd /d "%~dp0"
set NODE=d:\cursor\cursor\resources\app\resources\helpers\node.exe
if not exist "%NODE%" (
  echo Node not found. Install Node.js or edit NODE path in this script.
  exit /b 1
)
"%NODE%" upload_to_github.mjs %*
pause
