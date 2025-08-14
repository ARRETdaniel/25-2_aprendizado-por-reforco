@echo off
echo Starting Self-Driving Car Simulation Environment...


:: Start CARLA simulator in a new window
start cmd /k call start_carla.bat

:: Wait for CARLA to initialize
echo Waiting for CARLA simulator to initialize...
timeout /t 8 /nobreak

:: Start the YOLO detection server in a new window
start cmd /k call start_detector.bat

:: Wait for detector to initialize
echo Waiting for detector server to initialize...
timeout /t 5 /nobreak


:: Start the module_7 client
start cmd /k call start_client.bat

echo All components started successfully!
