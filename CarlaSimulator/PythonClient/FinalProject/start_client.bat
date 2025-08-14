@echo off
echo Starting CARLA Client Module...
cd C:\Users\danie\Documents\Documents\MESTRADO\25-2_aprendizado-por-reforco\CarlaSimulator\PythonClient\FinalProject
echo Current directory: %CD%
:: Add the PythonAPI path to PYTHONPATH
set PYTHONPATH=%PYTHONPATH%;C:\Users\danie\Documents\Documents\MESTRADO\25-2_aprendizado-por-reforco\CarlaSimulator\PythonAPI
:: Run the module
py -3.6 module_7.py
