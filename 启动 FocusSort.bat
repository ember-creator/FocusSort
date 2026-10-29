@echo off
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%~dp0vendor"
set "PYTHONUTF8=1"
set "PYW="

if exist "%~dp0python\pythonw.exe" set "PYW=%~dp0python\pythonw.exe"
if not defined PYW if exist "%LOCALAPPDATA%\Python\bin\pythonw.exe" set "PYW=%LOCALAPPDATA%\Python\bin\pythonw.exe"
if not defined PYW if exist "%LOCALAPPDATA%\Python\pythoncore-3.14-64\pythonw.exe" set "PYW=%LOCALAPPDATA%\Python\pythoncore-3.14-64\pythonw.exe"
if not defined PYW if exist "%LOCALAPPDATA%\Programs\Python\Python312\pythonw.exe" set "PYW=%LOCALAPPDATA%\Programs\Python\Python312\pythonw.exe"
if not defined PYW if exist "%ProgramFiles%\Python312\pythonw.exe" set "PYW=%ProgramFiles%\Python312\pythonw.exe"

if not defined PYW for /f "delims=" %%P in ('where pythonw 2^>nul') do (
    if not defined PYW echo %%P | find /i "WindowsApps" >nul || set "PYW=%%P"
)
if not defined PYW for /f "delims=" %%P in ('where python 2^>nul') do (
    if not defined PYW echo %%P | find /i "WindowsApps" >nul || set "PYW=%%P"
)
if not defined PYW for /f "delims=" %%P in ('py -3 -c "import sys;print(sys.executable)" 2^>nul') do (
    if exist "%%~dpPpythonw.exe" set "PYW=%%~dpPpythonw.exe"
)
if not defined PYW goto nopython

start "" "%PYW%" "%~dp0run.py" %*
exit /b 0

:nopython
echo.
echo FocusSort could not find Python.
echo Install Python 3.10+ from https://www.python.org/downloads/
echo and enable "Add python.exe to PATH", then run this file again.
echo.
pause
exit /b 1
