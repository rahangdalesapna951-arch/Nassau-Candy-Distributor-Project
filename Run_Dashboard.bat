@echo off
title Nassau Candy Dashboard
echo ============================================
echo    Nassau Candy Factory Optimizer
echo    Starting Streamlit Dashboard...
echo ============================================
echo.

cd /d C:\Users\sap58\project2

echo Checking files...
if not exist nassau_dashboard.py (
    echo ERROR: nassau_dashboard.py not found!
    echo Make sure this file is in C:\Users\sap58\project2
    pause
    exit
)

echo Starting dashboard... Please wait...
echo Browser will open at http://localhost:8501
echo.
echo DO NOT close this window while using dashboard!
echo.

call streamlit run nassau_dashboard.py

pause