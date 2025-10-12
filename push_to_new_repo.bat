@echo off
echo ========================================
echo Push to NEW GitHub Repository
echo ========================================
echo.
echo Current remote: https://github.com/demistifying/112-Analytics.git
echo.
echo Please create a NEW repository on GitHub, then enter the URL below.
echo Example: https://github.com/yourusername/new-repo-name.git
echo.
set /p NEW_REPO_URL="Enter your NEW GitHub repository URL: "
echo.

echo Removing old remote...
git remote remove origin
echo.

echo Adding new remote...
git remote add origin %NEW_REPO_URL%
echo.

echo Pushing to new repository...
git push -u origin main
echo.

echo ========================================
echo Done! Your project is now on the new repository.
echo ========================================
pause
