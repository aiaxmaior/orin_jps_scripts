#!/usr/bin/env python3
# DeepStream probe with proximity metadata (requires pyds)
# NOTE: pyds not available, use C version instead

import sys

print("="*70)
print("ERROR: Python DeepStream bindings (pyds) not available")
print("="*70)
print("\nThis module requires pyds for metadata access.")
print("\nUse the C proximity apps instead:")
print("  ./deepstream_proximity          # DashCamNet + proximity")
print("  ./deepstream_proximity_yolo     # YOLOv8n-seg + proximity")
print("\nThese provide:")
print("  - Distance calculation for all detected objects")
print("  - Color-coded bounding boxes (Red/Orange/Yellow/Green)")
print("  - Distance labels overlaid on video")
print("  - Console warnings for critical objects")
print("="*70)

sys.exit(1)
