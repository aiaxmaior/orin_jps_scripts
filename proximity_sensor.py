#!/usr/bin/env python3
# Proximity sensor for ADAS - monocular distance estimation

import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum


class ObjectClass(Enum):
    CAR = 0
    BICYCLE = 1
    PERSON = 2
    ROAD_SIGN = 3


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
}


class ProximitySensor:
    
    def __init__(self, focal_length, image_width=1280, image_height=720):
        self._focal_length = focal_length
        self._image_width = image_width
        self._image_height = image_height
        
        # Try loading calibration data
        try:
            data = np.load('camera_calibration.npz')
            self._camera_matrix = data['camera_matrix']
            self._dist_coeffs = data['dist_coeffs']
            self._focal_length = float(data['focal_length'])
            print(f"Loaded calibration: focal_length = {self._focal_length:.2f}")
        except FileNotFoundError:
            print("No calibration file, using default focal length")
            self._camera_matrix = None
            self._dist_coeffs = None

    def estimate_distance_from_height(self, bbox_height, object_class):
        """Distance = (real_height * focal_length) / pixel_height"""
        if bbox_height <= 0:
            return float('inf')

        dimensions = OBJECT_DIMENSIONS.get(object_class)
        if dimensions is None:
            return float('inf')

        real_height = dimensions.height
        distance = (real_height * self._focal_length) / bbox_height

        return distance

    def estimate_distance_from_width(self, bbox_width, object_class):
        if bbox_width <= 0:
            return float('inf')

        dimensions = OBJECT_DIMENSIONS.get(object_class)
        if dimensions is None:
            return float('inf')

        real_width = dimensions.width
        distance = (real_width * self._focal_length) / bbox_width

        return distance

    def estimate_distance(self, bbox, object_class):
        """Estimate distance using both height and width"""
        x1, y1, x2, y2 = bbox

        bbox_width = x2 - x1
        bbox_height = y2 - y1

        dist_h = self.estimate_distance_from_height(bbox_height, object_class)
        dist_w = self.estimate_distance_from_width(bbox_width, object_class)

        # Weighted avg (height more reliable)
        distance = (dist_h * 0.7 + dist_w * 0.3)

        return {
            'distance': distance,
            'distance_from_height': dist_h,
            'distance_from_width': dist_w,
            'bbox_height': bbox_height,
            'bbox_width': bbox_width,
        }

    def calculate_time_to_collision(self, distance, ego_velocity, object_velocity=0.0):
        """Time to collision calc"""
        relative_velocity = ego_velocity - object_velocity

        if relative_velocity <= 0:
            return float('inf')

        ttc = distance / relative_velocity
        return ttc

    def get_proximity_warning_level(self, distance, ttc):
        if distance < 3.0 or ttc < 1.5:
            return 'CRITICAL'
        elif distance < 10.0 or ttc < 3.0:
            return 'WARNING'
        elif distance < 20.0:
            return 'CAUTION'
        else:
            return 'SAFE'

    def is_in_collision_path(self, bbox, lane_center_x=None):
        """Check if object is in vehicle's path"""
        if lane_center_x is None:
            lane_center_x = self._image_width // 2

        x1, y1, x2, y2 = bbox
        bbox_center_x = (x1 + x2) // 2

        # Collision zone (±30% of image width)
        collision_zone_width = self._image_width * 0.3

        return abs(bbox_center_x - lane_center_x) < collision_zone_width


# ==============================================================================
# Test harness
# ==============================================================================
def test_proximity_sensor():
    FOCAL_LENGTH = 700.0
    
    sensor = ProximitySensor(focal_length=FOCAL_LENGTH)

    print("="*60)
    print("Proximity Sensor Test")
    print("="*60)

    test_cases = [
        ((500, 300, 700, 600), ObjectClass.CAR, "Car - medium distance"),
        ((400, 250, 800, 650), ObjectClass.CAR, "Car - close"),
        ((550, 400, 650, 500), ObjectClass.PERSON, "Person - far"),
        ((500, 350, 750, 550), ObjectClass.BICYCLE, "Bicycle"),
    ]

    for bbox, obj_class, description in test_cases:
        result = sensor.estimate_distance(bbox, obj_class)
        distance = result['distance']

        ttc = sensor.calculate_time_to_collision(
            distance=distance,
            ego_velocity=15.0,
            object_velocity=0.0
        )

        warning = sensor.get_proximity_warning_level(distance, ttc)
        in_path = sensor.is_in_collision_path(bbox)

        print(f"\n{description}:")
        print(f"  Distance: {distance:.2f} m")
        print(f"  TTC: {ttc:.2f} s")
        print(f"  Warning: {warning}")
        print(f"  In path: {in_path}")


if __name__ == "__main__":
    test_proximity_sensor()
