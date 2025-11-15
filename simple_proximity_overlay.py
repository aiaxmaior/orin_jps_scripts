#!/usr/bin/env python3
# Simple proximity overlay using GStreamer appsink

import cv2
import numpy as np
from advanced_proximity_sensor import AdvancedProximitySensor, ObjectClass
import gi
gi.require_version('Gst', '1.0')
from gi import GObject
from gi.repository import Gst, GLib
import sys

Gst.init(None)

sensor = AdvancedProximitySensor(
    focal_length=700.0,
    image_width=640,
    image_height=360,
    camera_height=1.2,
    camera_tilt_angle=5.0
)

DASHCAMNET_CLASSES = {
    0: ('car', ObjectClass.CAR),
    1: ('bicycle', ObjectClass.BICYCLE),
    2: ('person', ObjectClass.PERSON),
    3: ('road_sign', ObjectClass.ROAD_SIGN)
}

def new_sample(sink, data):
    sample = sink.emit('pull-sample')
    if not sample:
        return Gst.FlowReturn.OK
    
    buf = sample.get_buffer()
    caps = sample.get_caps()
    
    height = caps.get_structure(0).get_value('height')
    width = caps.get_structure(0).get_value('width')
    
    success, map_info = buf.map(Gst.MapFlags.READ)
    if not success:
        return Gst.FlowReturn.OK
    
    # Convert to numpy
    frame_data = np.ndarray(
        shape=(height, width, 4),
        dtype=np.uint8,
        buffer=map_info.data
    )
    
    frame = cv2.cvtColor(frame_data, cv2.COLOR_RGBA2BGR)
    
    # TODO: Extract bboxes from DeepStream metadata
    # Need pyds for this
    
    buf.unmap(map_info)
    return Gst.FlowReturn.OK

# Pipeline
pipeline_str = f"""
    nvarguscamerasrc sensor-id=0 ! 
    video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 !
    nvvidconv ! video/x-raw, width=640, height=360 !
    appsink name=sink emit-signals=True sync=False
"""

pipeline = Gst.parse_launch(pipeline_str)
sink = pipeline.get_by_name('sink')
sink.connect('new-sample', new_sample, None)

pipeline.set_state(Gst.State.PLAYING)

try:
    loop = GLib.MainLoop()
    loop.run()
except KeyboardInterrupt:
    pass

pipeline.set_state(Gst.State.NULL)
