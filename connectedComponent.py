# -*- coding: utf-8 -*-
"""
Created on Wed May 26 18:23:34 2021

@author: Zein Hajj-Ali - zeinhajjali@sce.carleton.ca
"""

# CONNECTED COMPONENT ANALYSIS FROM DEPTH 0.30 TO 0.50

import pyrealsense2 as rs
import numpy as np
import cv2

windowName = "depthSlice"

# Create pipeline
pipeline = rs.pipeline()

# Create a config object
config = rs.config()

# Tell config that we will use a recorded device from file to be used by the pipeline through playback.
rs.config.enable_device_from_file(config, "Z:\Patient_28\Video_Data\Pt_28_1.bag")

# Change this parameters according to the recorded bag file resolution
config.enable_stream(rs.stream.depth)
config.enable_stream(rs.stream.color)

# Start streaming from file
pipeline.start(config)

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
    
    np_depth_frame_bool = (np_depth_frame_scaled < 0.309) * 1.0
    
    np_color_frame_masked = np_color_frame
    
    for i in range(0, 3):
        np_color_frame_masked[:, :, i] = np_color_frame_masked[:, :, i] * np_depth_frame_bool
    
    
    print("new frame")
    print(np_color_frame_masked[100,100])
    
    # np_gray_frame_masked = cv2.cvtColor(np_color_frame_masked, cv2.COLOR_BGR2GRAY)
    # gray_inverted = cv2.bitwise_not(np_gray_frame_masked)
    
    np_depth_frame_masked = np_depth_frame_scaled
    
    np_depth_frame_masked = np_depth_frame_masked * np_depth_frame_bool

    masked_uint8 = np.asarray(np_depth_frame_masked, dtype="uint8")
    numLabels, labels = cv2.connectedComponents(masked_uint8)
    
    depth_uint8 = np_depth_frame_bool.astype('uint8')
    depth_uint8[depth_uint8 > 0] = 255
    inv_depth_bool = cv2.bitwise_not(depth_uint8)
    
    blobDetectorParams = cv2.SimpleBlobDetector_Params()
    blobDetectorParams.filterByArea = True
    blobDetectorParams.minArea = 150
    blobDetectorParams.filterByCircularity = True
    blobDetectorParams.minCircularity = 0
    # blobDetectorParams.filterByInertia
    # blobDetectorParams.minInertiaRatio = 0.2
    
    blobDetector = cv2.SimpleBlobDetector_create(blobDetectorParams)
    blobKeypoints = blobDetector.detect(inv_depth_bool)
    
    frame_blobs = cv2.drawKeypoints(inv_depth_bool, blobKeypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Render image in opencv window
    cv2.imshow(windowName, frame_blobs)
    
    
    key = cv2.waitKey(1)
    # if pressed ESCAPE exit program
    if key == 27:
        cv2.destroyAllWindows()
        break
