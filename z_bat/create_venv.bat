@ECHO off
TITLE VENV: Create and Install

SET "path_env=%~dp0..\.venv\Scripts"
ECHO Enviroment path: %path_env%

ECHO Create .venv
python -m venv %~dp0..\.venv

CALL "%path_env%\activate.bat"
"%path_env%\python.exe" -m pip list
"%path_env%\python.exe" -m pip install -U pip
"%path_env%\python.exe" -m pip install -r "%~dp0../requirements.txt"
"%path_env%\python.exe" -m pip list
CALL "%path_env%\deactivate.bat"

TIMEOUT 5