#!/usr/bin/env python3
"""
Quick test to check if visual monitor is working and display a status summary
"""

import time
import subprocess
import psutil

def main():
    print("🎥 CARLA DRL Visual System Status")
    print("=" * 50)
    
    # Check running processes
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
            if any(keyword in cmdline.lower() for keyword in ['carla', 'visual_monitor', 'real_carla_ppo']):
                processes.append({
                    'pid': proc.info['pid'],
                    'name': proc.info['name'],
                    'cmd': cmdline[:60] + "..." if len(cmdline) > 60 else cmdline
                })
        except:
            pass
    
    print(f"📋 Active Components ({len(processes)} found):")
    for proc in processes:
        print(f"  • PID {proc['pid']}: {proc['cmd']}")
    
    # Check ports
    active_ports = []
    for conn in psutil.net_connections():
        try:
            if conn.laddr.port in [2000, 5555, 5556] and conn.status == 'LISTEN':
                active_ports.append(conn.laddr.port)
        except:
            pass
    
    print(f"\n🌐 Active Ports: {active_ports}")
    
    # GPU status
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            line = result.stdout.strip()
            parts = line.split(', ')
            if len(parts) >= 2:
                gpu_util = parts[0]
                mem_used = parts[1]
                print(f"🎮 GPU Status: {gpu_util}% utilization, {mem_used} MB memory")
    except:
        print("🎮 GPU Status: nvidia-smi not available")
    
    print(f"\n💡 Visual System Status:")
    print(f"  • CARLA Server: {'✅' if 2000 in active_ports else '❌'}")
    print(f"  • ZMQ Data Feed: {'✅' if 5555 in active_ports else '❌'}")
    print(f"  • Visual Monitor: {'✅' if any('visual_monitor' in p['cmd'] for p in processes) else '❌'}")
    print(f"  • DRL Training: {'✅' if any('real_carla_ppo' in p['cmd'] for p in processes) else '❌'}")
    
    print(f"\n🖼️ To see the camera feed window:")
    print(f"  1. Check if OpenCV window 'CARLA DRL - Live Camera Feed' is open")
    print(f"  2. Look for window on taskbar or alt-tab")
    print(f"  3. Press ESC or Q in the window to close")
    print(f"  4. Press S in the window to save screenshot")
    
    print(f"\n📊 Expected Behavior:")
    print(f"  • Real-time camera feed from CARLA simulation")
    print(f"  • Vehicle position, rotation, speed displayed")
    print(f"  • FPS counter showing refresh rate")
    print(f"  • Training metrics overlay")

if __name__ == "__main__":
    main()
