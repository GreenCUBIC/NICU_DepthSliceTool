# -*- coding: utf-8 -*-
"""
Created on Tue May 18 15:17:35 2021

@author: Zein Hajj-Ali - zeinhajjali@sce.carleton.ca
"""

import pyrealsense2 as rs
import numpy as np
import cv2

windowName = "depthSlice"

# Create pipeline
pipeline = rs.pipeline()

# Create a config object
config = rs.config()

# Tell config that we will use a recorded device from file to be used by the pipeline through playback.
config.enable_device_from_file("Z:\Patient_28\Video_Data\Pt_28_1.bag")

# Change this parameters according to the recorded bag file resolution
config.enable_stream(rs.stream.depth)
config.enable_stream(rs.stream.color)

# Start streaming from file
pipeline_prof = pipeline.start(config)

playback = pipeline_prof.get_device().as_playback()

playback.set_real_time(False)

align = rs.align(rs.stream.color)

# Create opencv window to render image in
cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)

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
    
    np_depth_frame_bool = (np_depth_frame_scaled < 0.36) * 1.0
    
    np_color_frame_masked = np_color_frame
    
    for i in range(0, 3):
        np_color_frame_masked[:, :, i] = np_color_frame_masked[:, :, i] * np_depth_frame_bool
    
    print("new frame")
    print(np_color_frame_masked[100,100])

    # Render image in opencv window
    cv2.imshow(windowName, np_color_frame)
    
    device = pipeline_prof.get_device()
    
    key = cv2.waitKey(1)
    # if pressed ESCAPE exit program
    if key == 27:
        cv2.destroyAllWindows()
        break
