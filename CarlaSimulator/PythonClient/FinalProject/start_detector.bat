@echo off
echo Starting YOLO Detection Server...
cd %~dp0\detector_socket
call conda activate yolo_gpu
python detector_server.py
