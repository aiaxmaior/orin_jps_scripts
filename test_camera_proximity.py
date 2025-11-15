#!/usr/bin/env python3
# Camera test with proximity overlay (no DeepStream)

import cv2
import numpy as np
from advanced_proximity_sensor import AdvancedProximitySensor, ObjectClass

sensor = AdvancedProximitySensor(
    focal_length=700.0,
    image_width=1280,
    image_height=720,
    camera_height=1.2,
    camera_tilt_angle=5.0
)

# CSI camera pipeline
gst_str = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! "
    "appsink drop=1"
)

cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit(1)

print("Camera opened successfully!")
print("Press 'q' to quit")
print("\nDrawing simulated detections with proximity estimation...")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame")
        break
    
    frame_count += 1
    
    # Simulated detections (would come from DashCamNet in real pipeline)
    simulated_detections = [
        ((500, 300, 700, 500), ObjectClass.CAR, "Car"),
        ((200, 400, 350, 550), ObjectClass.PERSON, "Person"),
        ((800, 200, 1000, 400), ObjectClass.UNKNOWN, "Unknown"),
    ]
    
    for bbox, obj_class, label in simulated_detections:
        x1, y1, x2, y2 = bbox
        
        # Distance calc
        result = sensor.estimate_distance_multi_method(bbox, obj_class)
        distance = result['distance']
        
        # Collision metrics
        ego_velocity = 13.89  # m/s (~50 km/h)
        collision_prob = sensor.calculate_collision_probability(bbox, distance, ego_velocity)
        ttc = distance / ego_velocity if ego_velocity > 0 else float('inf')
        
        # Warning level
        if distance < 3.0 or ttc < 1.5:
            warning = 'CRITICAL'
            color = (0, 0, 255)  # Red BGR
        elif distance < 10.0 or ttc < 3.0:
            warning = 'WARNING'
            color = (0, 165, 255)  # Orange BGR
        elif distance < 20.0:
            warning = 'CAUTION'
            color = (0, 255, 255)  # Yellow BGR
        else:
            warning = 'SAFE'
            color = (0, 255, 0)  # Green BGR
        
        # Draw bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Draw label
        text = f"{label} {distance:.1f}m"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, color, 2)
        
        # Critical warnings
        if warning == 'CRITICAL':
            print(f"[{warning}] {label} at {distance:.1f}m (TTC: {ttc:.1f}s, Risk: {collision_prob:.0%})")
    
    # FPS counter
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display
    cv2.imshow("Camera + Proximity Sensor Test", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nProcessed {frame_count} frames")
