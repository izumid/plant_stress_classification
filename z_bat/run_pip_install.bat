@echo off
TITLE PIP: INSTALL

SET "path_env=%~dp0..\.venv\Scripts"
ECHO Enviroment path: %path_env%

CALL "%path_env%\activate.bat"
python -m pip list
python -m pip install -U pip
python -m pip install -r "%~dp0../requirements.txt"
python -m pip list
CALL "%path_env%\deactivate.bat"

TIMEOUT 10