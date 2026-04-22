import cv2
import numpy as np
import math
import sys

def start_visualizer(state_queue):
    print("DEBUG: OpenCV Visualizer started.")
    canvas_size = 600
    center = canvas_size // 2
    scale = 20.0 # pixels per meter
    
    while True:
        if not state_queue.empty():
            # Get latest data and clear queue
            data = None
            while not state_queue.empty():
                data = state_queue.get()
            
            if data is None: continue
            
            lidar, goal_dist, goal_heading, dev_x, dev_y, action = data
            
            # Create Black Canvas
            img = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
            img[:] = (30, 30, 30)
            
            # Draw Grid Rings
            for r in range(1, 6):
                radius = int(r * 2 * scale)
                cv2.circle(img, (center, center), radius, (60, 60, 60), 1)
                cv2.putText(img, f"{r*2}m", (center + radius, center), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

            # Draw LiDAR Scan
            # Training env: index 0 is Behind (-180), index 64 is Forward (0)
            num_rays = len(lidar)
            angle_step = 360.0 / num_rays
            
            for i, val in enumerate(lidar):
                # val is normalized clearance [0, 1]. Total dist = val * 12.0 + 0.5
                actual_dist = (val * 12.0) + 0.5
                angle_deg = -180.0 + i * angle_step
                # Correct Left/Right flip: ROS/Training is CCW, so we negate the angle 
                # before mapping Forward (0) to UP (-90).
                angle_rad = math.radians(-angle_deg - 90.0)
                
                px = int(center + actual_dist * scale * math.cos(angle_rad))
                py = int(center + actual_dist * scale * math.sin(angle_rad))
                
                # Draw ray line (faint)
                cv2.line(img, (center, center), (px, py), (50, 50, 150), 1)
                # Draw hit point
                cv2.circle(img, (px, py), 2, (0, 0, 255), -1)

            # Draw Goal (dev_x = Forward, dev_y = Left)
            # Screen coords: Forward is UP (-y), Left is LEFT (-x)
            goal_px = int(center - dev_y * scale)
            goal_py = int(center - dev_x * scale)
            cv2.circle(img, (goal_px, goal_py), 8, (0, 255, 0), -1)
            cv2.putText(img, "GOAL", (goal_px + 10, goal_py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Draw Action Vector
            act_px = int(center - action[1] * 50)
            act_py = int(center - action[0] * 50)
            cv2.line(img, (center, center), (act_px, act_py), (0, 255, 255), 2)

            # Draw Drone
            cv2.circle(img, (center, center), 10, (255, 150, 0), -1)
            cv2.line(img, (center, center), (center, center - 15), (255, 255, 255), 2) # Heading

            # Draw Info Text
            info = f"Dist: {goal_dist:.2f}m | HeadDiff: {math.degrees(goal_heading):.1f}deg"
            cv2.putText(img, info, (20, canvas_size - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show dev_x/y for debugging
            dev_info = f"devX: {dev_x:.2f} (Fwd) | devY: {dev_y:.2f} (Left)"
            cv2.putText(img, dev_info, (20, canvas_size - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Drone RL Visualizer (OpenCV)", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import queue
    q = queue.Queue()
    # Dummy data test
    lidar = np.random.rand(128)
    q.put((lidar, 5.0, 0.5, 4.0, 3.0, [0.5, -0.2]))
    start_visualizer(q)
