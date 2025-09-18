@ECHO off
TITLE [EXE: Based On Venv]

set "path_env=%~dp0..\.venv\Scripts"
set "output_folder=%~dp0..\exe_file"
set "PYINSTALLER_CONFIG_DIR=%output_folder%"

echo Enviroment path: %path_env%

CALL "%path_env%\activate"
CALL "%path_env%\python.exe" -m PyInstaller  ^
	--clean ^
    --onefile ^
	--hidden-import=xgboost ^
	--collect-all xgboost ^
	--exclude-module hypothesis ^
	--exclude-module pytest ^
    --icon="%~dp0..\z_img\icon.ico" ^
    --name "plant_stress_classification" ^
    --workpath "%output_folder%" ^
    --distpath "%output_folder%" ^
    --specpath "%output_folder%" ^
    "%~dp0..\main.py"

CALL "%path_env%\deactivate"
TIMEOUT 10
EXIT