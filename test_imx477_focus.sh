#!/bin/bash
# Test IMX477 with different focus positions
# The IMX477 autofocus can be controlled via nvarguscamerasrc

echo "Testing IMX477 camera with autofocus..."
echo "Press Ctrl+C to stop"

# Test with autofocus enabled
gst-launch-1.0 nvarguscamerasrc sensor-id=1 ! \
  "video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1" ! \
  nvvidconv ! "video/x-raw, format=BGRx" ! \
  videoconvert ! "video/x-raw, format=BGR" ! \
  xvimagesink sync=0
