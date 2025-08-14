@echo off
echo Cleaning up simulation environment...

:: Find and kill all relevant processes
taskkill /f /im CarlaUE4.exe
taskkill /f /im python.exe /fi "WINDOWTITLE eq *detector_server*"
taskkill /f /im python.exe /fi "WINDOWTITLE eq *module_7*"

echo Cleanup complete!
