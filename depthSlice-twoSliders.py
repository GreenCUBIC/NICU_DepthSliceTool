# -*- coding: utf-8 -*-
"""
Created on Mon May 30 01:08:41 2021

@author: Zein Hajj-Ali - zeinhajjali@sce.carleton.ca
"""

import pyrealsense2 as rs
import numpy as np
import cv2

windowName = "depthSlide-slider"
slider1Name = "Slice depth (increments of 0.001)"
slider2Name = "Slice start (increments of 0.001)"

# Create pipeline
pipeline = rs.pipeline()

# Create a config object
config = rs.config()

# Tell config that we will use a recorded device from file to be used by the pipeline through playback.
rs.config.enable_device_from_file(config, "Z:\Patient_11\Video_Data\Patient11.bag")

# Change this parameters according to the recorded bag file resolution
config.enable_stream(rs.stream.depth)
config.enable_stream(rs.stream.color)

# Start streaming from file
pipeline.start(config)

align = rs.align(rs.stream.color)

slider1Arg = 0
slider2Arg = 0
slice1At = 0
slice2At = 0

def updateFrame(arg):
    # Get value of slider
    value1 = cv2.getTrackbarPos(slider1Name, windowName)
    value2 = cv2.getTrackbarPos(slider2Name, windowName)
    
    sliceDepth1 = value1/1000
    sliceDepth2 = value2/1000
    
    # Render image in opencv window
    cv2.imshow(windowName, np_color_frame)
    
    return value1, sliceDepth1, value2, sliceDepth2
    
# Create opencv window to render image in
cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar(slider1Name, windowName, 15, 500, updateFrame)
cv2.createTrackbar(slider2Name, windowName, 0, 1000, updateFrame)

# Streaming loop
while True:
    
    # Get frameset of depth
    frames = pipeline.wait_for_frames()
    
    aligned_frames = align.process(frames)

    # Get depth and color frames
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue
    
    scaling_factor = depth_frame.get_units()
        
    np_depth_frame = np.asanyarray(depth_frame.get_data())
    np_color_frame = np.asanyarray(color_frame.get_data())
    np_color_frame = np_color_frame[...,::-1]
    
    np_depth_frame_scaled = np_depth_frame * scaling_factor
    
    sliceEnd = slice1At + slice2At
    np_depth_frame_bool1 = (np_depth_frame_scaled < sliceEnd) * 1
    np_depth_frame_bool2 = (np_depth_frame_scaled > slice2At) * 1
    np_depth_frame_bool = np.bitwise_and(np_depth_frame_bool1, np_depth_frame_bool2)
    
    np_color_frame_masked = np_color_frame
    
    for i in range(0, 3):
        np_color_frame_masked[:, :, i] = np_color_frame_masked[:, :, i] * np_depth_frame_bool
    
    slider1Arg, slice1At, slider2Arg, slice2At = updateFrame(0)
    
    
    key = cv2.waitKey(1)
    # if pressed ESCAPE exit program
    if key == 27:
        cv2.destroyAllWindows()
        break
    
