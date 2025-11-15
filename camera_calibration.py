#!/usr/bin/env python3
# Camera calibration using checkerboard pattern
# Extracts focal length for proximity calculations

import numpy as np
import cv2, glob, os

CHECKERBOARD = (6,9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def calibrate_camera(img_dir='calibration_images', pattern_size=CHECKERBOARD):
    """
    Calibrates camera using checkerboard images
    Returns: camera matrix, distortion coeffs, focal length
    """
    
    # 3D object points
    objp = np.zeros((1, pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    
    obj_points = []
    img_points = []
    
    images = glob.glob(os.path.join(img_dir, '*.jpg'))
    
    if not images:
        print(f"No images found in {img_dir}")
        return None
    
    h,w = 0,0
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h,w = gray.shape[:2]
        
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, 
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        if ret == True:
            obj_points.append(objp)
            
            # Refine corner positions
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            img_points.append(corners2)
            
            # Visualize
            cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
            cv2.imshow('Calibration', img)
            cv2.waitKey(100)
    
    cv2.destroyAllWindows()
    
    if len(obj_points) > 0:
        # Run calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, (w,h), None, None)
        
        focal_length = mtx[0,0]
        print(f"\nCalibration complete!")
        print(f"Focal length: {focal_length:.2f} pixels")
        print(f"Camera matrix:\n{mtx}")
        
        # Save results
        np.savez('camera_calibration.npz', 
                 camera_matrix=mtx,
                 dist_coeffs=dist,
                 focal_length=focal_length,
                 image_width=w,
                 image_height=h)
        
        return mtx, dist, focal_length
    else:
        print("Failed to find any valid calibration patterns")
        return None


if __name__ == '__main__':
    result = calibrate_camera()
    if result:
        print("\nCalibration saved to camera_calibration.npz")
    else:
        print("\nCalibration failed - ensure calibration images are available")
