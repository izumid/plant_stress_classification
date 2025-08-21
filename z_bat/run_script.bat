@echo off
TITLE RUN PROG: 

SET "path_env=%~dp0..\.venv\Scripts"
ECHO Enviroment path: %path_env%

CALL "%path_env%\activate.bat"
CALL "%path_env%\python.exe" "%~dp0main.py" /popup
CALL "%path_env%\deactivate.bat"

TIMEOUT 10
PAUSE