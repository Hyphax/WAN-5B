@echo off
echo ======================================
echo FIXING YOUR GITHUB REPOSITORY
echo ======================================
echo.

echo Step 1: Adding all files...
git add .

echo Step 2: Committing with the missing app/main.py...
git commit -m "CRITICAL FIX: Add missing app/main.py - Main FastAPI application for A100 deployment"

echo Step 3: Force pushing to GitHub...
git push origin main --force

echo.
echo ======================================
echo DONE! Check your GitHub repository now!
echo Go to: https://github.com/Hyphax/WAN-5B
echo You should see the app/ folder with main.py
echo ======================================
pause
