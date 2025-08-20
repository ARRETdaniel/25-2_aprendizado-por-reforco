#!/usr/bin/env python3
"""
Simple OpenCV Window Test
Tests if OpenCV windows can be displayed
"""

import cv2
import numpy as np
import time

def main():
    print("🖼️ OpenCV Window Test")
    print("This will open a test window to verify OpenCV display capability")
    print("Press ESC or Q to close the window\n")
    
    # Create a test image
    width, height = 640, 480
    
    try:
        for i in range(100):  # Show for ~10 seconds
            # Create dynamic test image
            image = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Add animated elements
            cv2.circle(image, (int(width/2 + 100*np.sin(i*0.1)), int(height/2)), 50, (0, 255, 0), -1)
            cv2.putText(image, f"OpenCV Test Frame {i}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, "Visual system is working!", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(image, "Press ESC or Q to close", (10, height-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.imshow('CARLA DRL Visual System Test', image)
            
            key = cv2.waitKey(100) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or Q
                print("✅ User closed window")
                break
        
        print("✅ OpenCV window test completed successfully")
        print("✅ Visual system is capable of displaying camera feeds")
        
    except Exception as e:
        print(f"❌ OpenCV window test failed: {e}")
        print("⚠️ Visual display may not be available")
        
    finally:
        cv2.destroyAllWindows()
        print("🧹 Windows closed")

if __name__ == "__main__":
    main()
