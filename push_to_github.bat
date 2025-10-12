@echo off
echo ========================================
echo Pushing 112-Analytics to GitHub
echo ========================================
echo.

REM Initialize git if not already initialized
if not exist .git (
    echo Initializing git repository...
    git init
    echo.
)

REM Add all files
echo Adding all files to git...
git add .
echo.

REM Create initial commit
echo Creating commit...
git commit -m "Initial commit: 112-Analytics project"
echo.

REM Prompt for GitHub repository URL
echo.
echo Please create a new repository on GitHub first, then enter the repository URL below.
echo Example: https://github.com/yourusername/112-Analytics.git
echo.
set /p REPO_URL="Enter your GitHub repository URL: "
echo.

REM Add remote origin
echo Adding remote origin...
git remote add origin %REPO_URL%
echo.

REM Push to GitHub
echo Pushing to GitHub...
git branch -M main
git push -u origin main
echo.

echo ========================================
echo Done! Your project has been pushed to GitHub.
echo ========================================
pause
