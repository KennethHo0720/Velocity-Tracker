import cv2
import numpy as np

def create_video():
    width, height = 640, 480
    fps = 30
    duration = 3
    output_file = 'dummy_video.mp4'
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    for i in range(fps * duration):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw a "plate" (calibration object) - Static
        cv2.rectangle(frame, (50, 200), (100, 250), (200, 0, 0), -1)
        
        # Draw a "barbell" (moving object)
        # Move up and down
        y = int(200 + 100 * np.sin(i / 10))
        cv2.circle(frame, (320, y), 20, (0, 255, 0), -1)
        
        out.write(frame)
        
    out.release()
    print(f"Created {output_file}")

if __name__ == "__main__":
    create_video()
