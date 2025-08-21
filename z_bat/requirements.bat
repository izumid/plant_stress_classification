@ECHO off
TITLE REQUIREMENTS

SET "path_env=%~dp0..\venv\Scripts"
ECHO Enviroment path: %path_env%

CALL "%path_env%\activate.bat"
pip freeze > "%~dp0..\requirements.txt"
CALL "%path_env%\deactivate.bat"
ECHO Generated file!!

TIMEOUT 5