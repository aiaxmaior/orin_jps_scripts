#!/usr/bin/env python3
# DeepStream Python pipeline with proximity (requires pyds)
# NOTE: pyds not available, use C version instead

import sys

print("="*70)
print("ERROR: Python DeepStream bindings (pyds) not available")
print("="*70)
print("\nUse the C version instead:")
print("  ./deepstream_proximity          # For DashCamNet")
print("  ./deepstream_proximity_yolo     # For YOLOv8n-seg")
print("\nOr use deepstream-app directly:")
print("  deepstream-app -c deepstream_single_csi.txt")
print("="*70)

sys.exit(1)
