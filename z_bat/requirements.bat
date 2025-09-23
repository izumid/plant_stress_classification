@ECHO off
TITLE REQUIREMENTS

REM Path to your virtual environment's Scripts folder
SET "path_env=%~dp0..\.venv\Scripts"
ECHO Environment path: %path_env%

REM Check if the venv exists
IF NOT EXIST "%path_env%\python.exe" (
    ECHO [ERROR] Could not find python.exe in %path_env%
    ECHO Make sure the virtual environment exists and the path is correct.
    PAUSE
    EXIT /B 1
)

REM Show which Python will be used
ECHO.
ECHO === Checking Python version and location ===
"%path_env%\python.exe" --version
"%path_env%\python.exe" -c "import sys; print(sys.executable)"

REM Show which pip will be used
ECHO.
ECHO === Checking pip version and location ===
"%path_env%\python.exe" -m pip --version

REM Freeze only local packages (ignore system site packages)
ECHO.
ECHO === Generating requirements.txt from venv ===
"%path_env%\python.exe" -m pip freeze --local > "%~dp0..\requirements.txt"

ECHO.
ECHO Generated file: %~dp0..\requirements.txt
TIMEOUT 5