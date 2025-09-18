@echo off
echo ======================================
echo VERIFYING ALL FILES ARE PRESENT
echo ======================================
echo.

echo Checking critical files:
echo.

if exist "app\main.py" (
    echo [✓] app\main.py - FOUND
    echo    Size: 
    dir "app\main.py" | find "main.py"
) else (
    echo [✗] app\main.py - MISSING!
)

if exist "requirements.txt" (
    echo [✓] requirements.txt - FOUND
) else (
    echo [✗] requirements.txt - MISSING!
)

if exist "Dockerfile" (
    echo [✓] Dockerfile - FOUND
) else (
    echo [✗] Dockerfile - MISSING!
)

if exist "README.md" (
    echo [✓] README.md - FOUND
) else (
    echo [✗] README.md - MISSING!
)

echo.
echo Git status:
git status

echo.
echo Files tracked by git:
git ls-files

echo.
echo ======================================
pause
