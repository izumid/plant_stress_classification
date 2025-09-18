@echo off
TITLE PIP: INSTALL

SET "path_env=%~dp0..\.venv\Scripts"
ECHO Enviroment path: %path_env%

CALL "%path_env%\activate.bat"
"%path_env%\python.exe" -m pip list
"%path_env%\python.exe" -m pip install -U pip
"%path_env%\python.exe" -m pip install -r "%~dp0../requirements.txt"
"%path_env%\python.exe" -m pip list
CALL "%path_env%\deactivate.bat"

TIMEOUT 10