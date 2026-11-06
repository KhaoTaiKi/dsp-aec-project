@echo off
title Jupyter Notebook (DSP)
cd /d "%~dp0"
echo ===================================================
echo Starting Jupyter Notebook...
echo Working Directory: %CD%
echo ===================================================
"C:\Users\khaok\dsp_env\Scripts\jupyter.exe" notebook
pause