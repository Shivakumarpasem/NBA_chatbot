@echo off
title NBA Chatbot
cd /d "%~dp0"
echo Starting NBA Chatbot...
echo.
python -m streamlit run app.py
pause
