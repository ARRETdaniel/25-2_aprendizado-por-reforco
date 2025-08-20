#!/usr/bin/env python3
"""
CARLA-DRL Integration Demonstration
Shows the complete pipeline working with real CARLA simulation
"""

import os
import sys
import time
import subprocess
import threading
import signal
from pathlib import Path

# Add paths for imports
workspace_dir = Path(__file__).parent
sys.path.append(str(workspace_dir / "communication"))
sys.path.append(str(workspace_dir / "drl_agent"))

print("🚀 CARLA-DRL Integration Demonstration")
print("=" * 50)

class SystemOrchestrator:
    """Orchestrates the complete CARLA-DRL system."""
    
    def __init__(self):
        self.workspace_dir = Path(__file__).parent
        self.carla_process = None
        self.client_process = None
        self.training_process = None
        self.running = True
        
        # Setup signal handling
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print("\n🛑 Shutdown signal received. Cleaning up...")
        self.running = False
        self.cleanup()
    
    def check_prerequisites(self):
        """Check if all prerequisites are met."""
        print("🔍 Checking prerequisites...")
        
        checks = []
        
        # Check CARLA directory
        carla_dir = self.workspace_dir / "CarlaSimulator"
        carla_exe = carla_dir / "CarlaUE4.exe"
        checks.append(("CARLA executable", carla_exe.exists()))
        
        # Check Python 3.6
        try:
            result = subprocess.run(["py", "-3.6", "--version"], 
                                  capture_output=True, text=True, timeout=5)
            py36_available = result.returncode == 0
        except:
            py36_available = False
        checks.append(("Python 3.6", py36_available))
        
        # Check conda environment
        try:
            result = subprocess.run(["conda", "env", "list"], 
                                  capture_output=True, text=True, timeout=5)
            conda_env_exists = "carla_drl_py312" in result.stdout
        except:
            conda_env_exists = False
        checks.append(("carla_drl_py312 environment", conda_env_exists))
        
        # Check key files
        zmq_client = self.workspace_dir / "carla_client_py36" / "carla_zmq_client.py"
        zmq_bridge = self.workspace_dir / "communication" / "zmq_bridge.py"
        drl_trainer = self.workspace_dir / "drl_agent" / "real_carla_ppo_trainer.py"
        
        checks.append(("ZMQ CARLA client", zmq_client.exists()))
        checks.append(("ZMQ bridge", zmq_bridge.exists()))
        checks.append(("DRL trainer", drl_trainer.exists()))
        
        # Print results
        all_passed = True
        for name, passed in checks:
            status = "✅" if passed else "❌"
            print(f"  {status} {name}")
            if not passed:
                all_passed = False
        
        return all_passed
    
    def start_carla_server(self):
        """Start CARLA server."""
        carla_dir = self.workspace_dir / "CarlaSimulator"
        carla_exe = carla_dir / "CarlaUE4.exe"
        
        if not carla_exe.exists():
            print("❌ CARLA executable not found")
            return False
        
        print("🏁 Starting CARLA server...")
        
        try:
            # Start CARLA with optimized settings
            cmd = [
                str(carla_exe),
                "-carla-server",
                "-windowed",
                "-ResX=800",
                "-ResY=600",
                "-quality-level=Low",
                "-carla-no-networking"
            ]
            
            self.carla_process = subprocess.Popen(
                cmd,
                cwd=str(carla_dir),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Wait for CARLA to start
            print("⏳ Waiting for CARLA to initialize (15 seconds)...")
            time.sleep(15)
            
            # Check if process is still running
            if self.carla_process.poll() is None:
                print("✅ CARLA server started successfully")
                return True
            else:
                print("❌ CARLA server failed to start")
                return False
                
        except Exception as e:
            print(f"❌ Error starting CARLA: {e}")
            return False
    
    def start_carla_client(self):
        """Start CARLA ZMQ client."""
        client_script = self.workspace_dir / "carla_client_py36" / "carla_zmq_client.py"
        
        if not client_script.exists():
            print("❌ CARLA ZMQ client script not found")
            return False
        
        print("🔗 Starting CARLA ZMQ client...")
        
        try:
            cmd = ["py", "-3.6", str(client_script)]
            
            self.client_process = subprocess.Popen(
                cmd,
                cwd=str(client_script.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait a bit for client to connect
            time.sleep(5)
            
            # Check if process is still running
            if self.client_process.poll() is None:
                print("✅ CARLA ZMQ client started successfully")
                return True
            else:
                print("❌ CARLA ZMQ client failed to start")
                return False
                
        except Exception as e:
            print(f"❌ Error starting CARLA client: {e}")
            return False
    
    def start_drl_training(self):
        """Start DRL training."""
        trainer_script = self.workspace_dir / "drl_agent" / "real_carla_ppo_trainer.py"
        
        if not trainer_script.exists():
            print("❌ DRL trainer script not found")
            return False
        
        print("🧠 Starting DRL training...")
        
        try:
            # Use conda to activate environment and run training
            cmd = [
                "conda", "run", "-n", "carla_drl_py312",
                "python", str(trainer_script)
            ]
            
            self.training_process = subprocess.Popen(
                cmd,
                cwd=str(trainer_script.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            print("✅ DRL training started successfully")
            return True
                
        except Exception as e:
            print(f"❌ Error starting DRL training: {e}")
            return False
    
    def monitor_training(self):
        """Monitor training output."""
        if not self.training_process:
            return
        
        print("\n📊 Training Output:")
        print("-" * 40)
        
        try:
            while self.running and self.training_process.poll() is None:
                output = self.training_process.stdout.readline()
                if output:
                    print(output.strip())
                time.sleep(0.1)
        except Exception as e:
            print(f"❌ Error monitoring training: {e}")
    
    def cleanup(self):
        """Clean up all processes."""
        print("\n🧹 Cleaning up processes...")
        
        # Stop training process
        if self.training_process and self.training_process.poll() is None:
            print("  Stopping DRL training...")
            try:
                self.training_process.terminate()
                self.training_process.wait(timeout=5)
            except:
                self.training_process.kill()
        
        # Stop client process
        if self.client_process and self.client_process.poll() is None:
            print("  Stopping CARLA client...")
            try:
                self.client_process.terminate()
                self.client_process.wait(timeout=5)
            except:
                self.client_process.kill()
        
        # Stop CARLA server
        if self.carla_process and self.carla_process.poll() is None:
            print("  Stopping CARLA server...")
            try:
                self.carla_process.terminate()
                self.carla_process.wait(timeout=10)
            except:
                self.carla_process.kill()
        
        print("✅ Cleanup complete")
    
    def run_demonstration(self):
        """Run the complete demonstration."""
        print("\n🎬 Starting CARLA-DRL Integration Demonstration")
        print("=" * 50)
        
        try:
            # Check prerequisites
            if not self.check_prerequisites():
                print("\n❌ Prerequisites not met. Please check installation.")
                return False
            
            print("\n✅ All prerequisites met!")
            
            # Start CARLA server
            if not self.start_carla_server():
                print("\n❌ Failed to start CARLA server")
                return False
            
            # Start CARLA client
            if not self.start_carla_client():
                print("\n❌ Failed to start CARLA client")
                self.cleanup()
                return False
            
            # Start DRL training
            if not self.start_drl_training():
                print("\n❌ Failed to start DRL training")
                self.cleanup()
                return False
            
            print("\n🎉 All systems started successfully!")
            print("\n📱 System Status:")
            print("  🏁 CARLA Server: Running")
            print("  🔗 ZMQ Client: Connected")
            print("  🧠 DRL Training: Active")
            print("\nPress Ctrl+C to stop the demonstration")
            
            # Monitor training
            self.monitor_training()
            
            return True
            
        except KeyboardInterrupt:
            print("\n\n🛑 Demonstration stopped by user")
            return True
        except Exception as e:
            print(f"\n❌ Demonstration failed: {e}")
            return False
        finally:
            self.cleanup()

def main():
    """Main demonstration function."""
    print("🎯 Real CARLA-DRL Integration System")
    print("🔬 This demonstration shows:")
    print("   • CARLA 0.8.4 simulation running")
    print("   • ZMQ bridge connecting Python 3.6 ↔ 3.12") 
    print("   • Real-time PPO training on live CARLA data")
    print("   • Multimodal observations (camera + vehicle state)")
    print("   • GPU-accelerated deep learning")
    
    orchestrator = SystemOrchestrator()
    
    print("\n⚠️ IMPORTANT NOTES:")
    print("   • This will start CARLA server automatically")
    print("   • Training will run for ~5 minutes")
    print("   • Press Ctrl+C to stop at any time")
    print("   • Monitor system resources during training")
    
    input("\nPress Enter to start the demonstration...")
    
    success = orchestrator.run_demonstration()
    
    if success:
        print("\n🎉 Demonstration completed successfully!")
        print("\n📊 What you just saw:")
        print("   ✅ CARLA simulation with realistic physics")
        print("   ✅ Real-time sensor data transmission")
        print("   ✅ DRL agent learning to drive")
        print("   ✅ Cross-version Python communication")
        print("   ✅ GPU-accelerated neural network training")
        
        print("\n🔍 Check the logs:")
        print(f"   📁 TensorBoard: drl_agent/logs/real_carla_training")
        print(f"   📊 Command: tensorboard --logdir drl_agent/logs")
        
        print("\n🚀 Next steps:")
        print("   • Experiment with different reward functions")
        print("   • Try different DRL algorithms (SAC, TD3)")
        print("   • Add more complex scenarios")
        print("   • Deploy trained models")
    else:
        print("\n❌ Demonstration failed")
        print("💡 Try running the system health check:")
        print("   start_real_carla_system.bat → Option 4")

if __name__ == "__main__":
    main()
