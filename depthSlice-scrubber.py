# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 20:01:36 2021

@author: Zein Hajj-Ali - zeinhajjali@sce.carleton.ca
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import datetime
import copy

windowName = "depthSlide-slider"
slider1Name = "Slice depth (increments of 0.001)"
slider2Name = "Slice start (increments of 0.001)"
switchName = "0: Play \n1: Pause"
seekName = "Timestamp"

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
pipeline_prof = pipeline.start(config)

device = pipeline_prof.get_device()

playback = device.as_playback()

duration = playback.get_duration()

align = rs.align(rs.stream.color)

slider1Arg = 0
slider2Arg = 0
slice1At = 0
slice2At = 0
savedFrame = None
isPaused = False

def updateFrame(arg):
    # Get value of slider
    value1 = cv2.getTrackbarPos(slider1Name, windowName)
    value2 = cv2.getTrackbarPos(slider2Name, windowName)
    
    sliceDepth1 = value1/1000
    sliceDepth2 = value2/1000
    
    return value1, sliceDepth1, value2, sliceDepth2

def playPause(arg):
    playPauseFlag = cv2.getTrackbarPos(switchName, windowName)
    global isPaused
    
    if playPauseFlag == 1:
        print('just paused')
        isPaused = True
        playback.pause()
        
    else:
        isPaused = False
        playback.resume()
        
    
# Create opencv window to render image in
cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
# cv2.createTrackbar(seekName, windowName, 0, )
cv2.createTrackbar(slider1Name, windowName, 15, 500, updateFrame)
cv2.createTrackbar(slider2Name, windowName, 0, 1000, updateFrame)
cv2.createTrackbar(switchName, windowName, 0, 1, playPause)

frames = pipeline.wait_for_frames()
# Streaming loop
while True:
    
    print(isPaused)

    if not isPaused:
        frames = pipeline.wait_for_frames()
                  
    aligned_frames = align.process(frames)

    # Get depth and color frames
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue
    
    scaling_factor = depth_frame.get_units()
        
    np_depth_frame = np.asanyarray(depth_frame.get_data())
    if not isPaused:
        np_color_frame = np.asanyarray(color_frame.get_data())
        savedFrame = copy.deepcopy(np_color_frame)
    else:
        np_color_frame = copy.deepcopy(savedFrame)
    np_color_frame = np_color_frame[...,::-1]
    
    np_depth_frame_scaled = np_depth_frame * scaling_factor
        
    np_depth_frame_scaled_copy = np_depth_frame_scaled.view()
    sliceEnd = slice1At + slice2At
    np_depth_frame_bool1 = (np_depth_frame_scaled_copy < sliceEnd) * 1
    np_depth_frame_bool2 = (np_depth_frame_scaled_copy > slice2At) * 1
    np_depth_frame_bool = np.bitwise_and(np_depth_frame_bool1, np_depth_frame_bool2)
    
    np_color_frame_masked = np_color_frame.view()
    
    for i in range(0, 3):
        np_color_frame_masked[:, :, i] = np_color_frame_masked[:, :, i] * np_depth_frame_bool
    
    slider1Arg, slice1At, slider2Arg, slice2At = updateFrame(0)
    
    # Render image in opencv window
    cv2.imshow(windowName, np_color_frame_masked)
    
    key = cv2.waitKey(1)
    # if pressed ESCAPE exit program
    if key == 27:
        cv2.destroyAllWindows()
        break
    
