@ECHO off
TITLE VENV: Create and Install

SET "path_env=%~dp0..\.venv\Scripts"
ECHO Enviroment path: %path_env%

ECHO Create .venv
python -m venv %~dp0..\.venv

CALL "%path_env%\activate.bat"
python -m pip list
python -m pip install -U pip
python -m pip install -r "%~dp0../requirements.txt"
python -m pip list
CALL "%path_env%\deactivate.bat"

TIMEOUT 5