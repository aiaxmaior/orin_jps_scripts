#!/usr/bin/env python3
"""
Advanced Proximity Sensor for ADAS
Multi-method distance estimation including unknown obstacles
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum


class ObjectClass(Enum):
    CAR = 0
    BICYCLE = 1
    PERSON = 2
    ROAD_SIGN = 3
    UNKNOWN = 99


@dataclass
class ObjectDimensions:
    height: float
    width: float
    length: float


# Real-world object dimensions (meters)
OBJECT_DIMENSIONS = {
    ObjectClass.CAR: ObjectDimensions(height=1.5, width=1.8, length=4.5),
    ObjectClass.BICYCLE: ObjectDimensions(height=1.2, width=0.6, length=1.8),
    ObjectClass.PERSON: ObjectDimensions(height=1.7, width=0.5, length=0.3),
    ObjectClass.ROAD_SIGN: ObjectDimensions(height=0.8, width=0.8, length=0.05),
    ObjectClass.UNKNOWN: ObjectDimensions(height=1.0, width=1.0, length=1.0),
}


class AdvancedProximitySensor:
    
    def __init__(self, focal_length=700.0, image_width=640, image_height=360, 
                 camera_height=1.2, camera_tilt_angle=5.0):
        self._focal_length = focal_length
        self._image_width = image_width
        self._image_height = image_height
        self._camera_height = camera_height
        self._camera_tilt_deg = camera_tilt_angle
        self._camera_tilt_rad = np.radians(camera_tilt_angle)
        
        # Previous frame data for temporal tracking
        self._prev_bboxes = {}
        self._frame_count = 0
        
    def estimate_distance_known_object(self, bbox, object_class):
        """Distance calc using known object dimensions (pinhole model)"""
        x1, y1, x2, y2 = bbox
        bbox_height = y2 - y1
        bbox_width = x2 - x1
        
        if object_class not in OBJECT_DIMENSIONS:
            return None
            
        real_dims = OBJECT_DIMENSIONS[object_class]
        
        if bbox_height < 1:
            return None
            
        # Use height-based distance
        dist_height = (real_dims.height * self._focal_length) / bbox_height
        
        # Use width-based distance as secondary
        if bbox_width > 1:
            dist_width = (real_dims.width * self._focal_length) / bbox_width
            distance = (dist_height + dist_width) / 2.0
        else:
            distance = dist_height
            
        return {
            'distance': distance,
            'confidence': 0.8,
            'method': 'known_object'
        }
    
    def estimate_distance_ground_plane(self, bbox):
        """Ground plane geometry - works for ANY ground-level object"""
        x1, y1, x2, y2 = bbox
        bbox_bottom_y = y2
        
        # Distance from bottom of frame
        y_diff = self._image_height - bbox_bottom_y
        
        if y_diff < 1:
            return None
            
        # Ground plane calc (assumes object touches ground)
        distance = (self._camera_height * self._focal_length) / y_diff
        
        # Adjust for camera tilt
        distance *= np.cos(self._camera_tilt_rad)
        
        return {
            'distance': distance,
            'confidence': 0.7,
            'method': 'ground_plane'
        }
    
    def estimate_distance_bbox_area(self, bbox, previous_bbox=None):
        """Fallback method using bbox area change"""
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        
        if area < 1:
            return None
            
        # Assume avg real-world area of 2.0 m^2
        avg_real_area = 2.0
        distance = np.sqrt((avg_real_area * self._focal_length**2) / area)
        
        return {
            'distance': distance,
            'confidence': 0.4,
            'method': 'bbox_area'
        }
    
    def estimate_distance_multi_method(self, bbox, object_class=ObjectClass.UNKNOWN):
        """Multi-method fusion for robust distance estimation"""
        estimates = []
        
        # Method 1: Known object dimensions
        if object_class != ObjectClass.UNKNOWN:
            result = self.estimate_distance_known_object(bbox, object_class)
            if result:
                estimates.append(result)
        
        # Method 2: Ground plane geometry
        result_gp = self.estimate_distance_ground_plane(bbox)
        if result_gp:
            estimates.append(result_gp)
        
        # Method 3: Bbox area fallback
        result_area = self.estimate_distance_bbox_area(bbox)
        if result_area:
            estimates.append(result_area)
        
        if not estimates:
            return {'distance': -1.0, 'confidence': 0.0, 'method': 'none'}
        
        # Weighted average based on confidence
        total_weight = sum(e['confidence'] for e in estimates)
        weighted_dist = sum(e['distance'] * e['confidence'] for e in estimates) / total_weight
        
        max_conf_method = max(estimates, key=lambda x: x['confidence'])
        
        return {
            'distance': weighted_dist,
            'confidence': max_conf_method['confidence'],
            'method': max_conf_method['method'],
            'estimates': estimates,
            'obstacle_type': self._classify_obstacle(bbox)
        }
    
    def _classify_obstacle(self, bbox):
        """Classify obstacle type based on bbox geometry"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height if height > 0 else 1.0
        
        if aspect_ratio > 2.0:
            return 'wall/barrier'
        elif aspect_ratio < 0.5:
            return 'pole/sign'
        elif height < 50:
            return 'low_obstacle'
        else:
            return 'unknown_obstacle'
    
    def calculate_collision_probability(self, bbox, distance, ego_velocity):
        """Estimate collision risk based on distance and velocity"""
        if distance <= 0 or ego_velocity <= 0:
            return 0.0
            
        ttc = distance / ego_velocity  # Time to collision
        
        if ttc < 1.0:
            return 1.0
        elif ttc < 2.0:
            return 0.8
        elif ttc < 3.0:
            return 0.5
        else:
            return max(0.0, 1.0 - (ttc / 10.0))
    
    def get_safe_following_distance(self, ego_velocity):
        """2-second rule for safe following distance"""
        return max(3.0, ego_velocity * 2.0)


# ==============================================================================
# Test harness
# ==============================================================================
if __name__ == '__main__':
    print("="*70)
    print("Advanced Proximity Sensor Test")
    print("="*70)
    
    sensor = AdvancedProximitySensor(
        focal_length=700.0,
        image_width=640,
        image_height=360,
        camera_height=1.2,
        camera_tilt_angle=5.0
    )
    
    # Test cases
    test_cases = [
        ((250, 150, 400, 300), ObjectClass.CAR, "Known car"),
        ((100, 200, 200, 350), ObjectClass.UNKNOWN, "Unknown obstacle (wall?)"),
        ((300, 250, 400, 400), ObjectClass.UNKNOWN, "Unknown obstacle (near ground)"),
        ((450, 100, 500, 250), ObjectClass.UNKNOWN, "Pole/sign?"),
        ((50, 180, 350, 300), ObjectClass.UNKNOWN, "Wide obstacle (barrier?)"),
    ]
    
    for bbox, obj_class, description in test_cases:
        result = sensor.estimate_distance_multi_method(bbox, obj_class)
        distance = result['distance']
        confidence = result['confidence']
        method = result['method']
        obstacle_type = result['obstacle_type']
        
        ego_vel = 13.89  # m/s (~50 km/h)
        collision_prob = sensor.calculate_collision_probability(bbox, distance, ego_vel)
        
        print(f"\n{description}:")
        print(f"  Distance: {distance:.2f} m")
        print(f"  Confidence: {confidence:.2f}")
        print(f"  Method: {method}")
        print(f"  Obstacle type: {obstacle_type}")
        print(f"  Collision prob: {collision_prob:.2%}")
        
        if 'estimates' in result:
            for est in result['estimates']:
                print(f"    - {est['method']}: {est['distance']:.2f}m (conf: {est['confidence']:.2f})")
