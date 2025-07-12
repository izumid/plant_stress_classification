@echo off
TITLE PIP: LIST LIBRARIES

SET "path_env=%~dp0..\.venv\Scripts"
ECHO Enviroment path: %path_env%

CALL "%path_env%\activate.bat"
python -m pip list
CALL "%path_env%\deactivate.bat"

PAUSE