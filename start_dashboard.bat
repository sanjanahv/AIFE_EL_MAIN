@echo off
echo Starting LightGBM SHAP Dashboard...
echo Make sure you have run the setup once before starting!
echo.

:: Open the default web browser to the local address
start http://localhost:5000

:: Run the Flask application
python app.py

pause
