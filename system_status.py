#!/usr/bin/env python3
"""
CARLA DRL System Status Checker
Displays real-time status of all components
"""

import psutil
import subprocess
import time
import sys

def check_carla_server():
    """Check if CARLA server is running."""
    try:
        for proc in psutil.process_iter(['pid', 'name']):
            if 'CarlaUE4' in proc.info['name']:
                return True, proc.info['pid']
        return False, None
    except:
        return False, None

def check_python_processes():
    """Check for Python DRL processes."""
    processes = []
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if 'python' in proc.info['name'].lower():
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                if any(keyword in cmdline for keyword in ['carla', 'drl', 'ppo', 'visual_monitor']):
                    processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cmdline': cmdline
                    })
    except:
        pass
    return processes

def check_gpu_usage():
    """Check GPU usage (if nvidia-smi available)."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            line = result.stdout.strip()
            parts = line.split(', ')
            if len(parts) >= 3:
                gpu_util = parts[0]
                mem_used = parts[1]
                mem_total = parts[2]
                return True, f"{gpu_util}%", f"{mem_used}/{mem_total} MB"
        return False, "N/A", "N/A"
    except:
        return False, "N/A", "N/A"

def check_port_usage():
    """Check if key ports are in use."""
    ports = {2000: "CARLA Server", 5555: "ZMQ Data", 5556: "ZMQ Actions"}
    port_status = {}
    
    for port, description in ports.items():
        try:
            for conn in psutil.net_connections():
                if conn.laddr.port == port and conn.status == 'LISTEN':
                    port_status[port] = f"✅ {description}"
                    break
            else:
                port_status[port] = f"❌ {description}"
        except:
            port_status[port] = f"⚠️ {description}"
    
    return port_status

def main():
    """Main status display loop."""
    print("🔍 CARLA DRL System Status Monitor")
    print("=" * 60)
    print("Press Ctrl+C to exit\n")
    
    try:
        while True:
            # Clear screen (Windows)
            subprocess.run('cls', shell=True)
            
            print("🔍 CARLA DRL System Status Monitor")
            print("=" * 60)
            print(f"⏰ Status at: {time.strftime('%H:%M:%S')}\n")
            
            # Check CARLA server
            carla_running, carla_pid = check_carla_server()
            if carla_running:
                print(f"🚗 CARLA Server: ✅ Running (PID: {carla_pid})")
            else:
                print(f"🚗 CARLA Server: ❌ Not running")
            
            # Check Python processes
            python_procs = check_python_processes()
            print(f"\n🐍 Python DRL Processes: {len(python_procs)} found")
            for proc in python_procs:
                cmdline_short = proc['cmdline'][:80] + "..." if len(proc['cmdline']) > 80 else proc['cmdline']
                print(f"   PID {proc['pid']}: {cmdline_short}")
            
            # Check GPU usage
            gpu_available, gpu_util, gpu_mem = check_gpu_usage()
            if gpu_available:
                print(f"\n🎮 GPU Status: ✅ Utilization: {gpu_util}, Memory: {gpu_mem}")
            else:
                print(f"\n🎮 GPU Status: ⚠️ nvidia-smi not available")
            
            # Check port usage
            print(f"\n🌐 Port Status:")
            port_status = check_port_usage()
            for port in [2000, 5555, 5556]:
                if port in port_status:
                    print(f"   Port {port}: {port_status[port]}")
            
            # System performance
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            print(f"\n💻 System Performance:")
            print(f"   CPU Usage: {cpu_percent:.1f}%")
            print(f"   Memory Usage: {memory.percent:.1f}% ({memory.used // (1024**3):.1f}GB / {memory.total // (1024**3):.1f}GB)")
            
            print("\n" + "=" * 60)
            print("Press Ctrl+C to exit")
            
            # Wait before next update
            time.sleep(3)
            
    except KeyboardInterrupt:
        print("\n\n👋 Status monitor stopped.")

if __name__ == "__main__":
    main()
