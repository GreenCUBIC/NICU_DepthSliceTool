# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 12:17:55 2021

@author: Zein Hajj-Ali - zeinhajjali@sce.carleton.ca
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import datetime
import time
import copy
import sys
import math
# from collections import deque
import os
import csv
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import libdst

DSENABLE = "DEPTH_SELECT_ENABLE"
PTENABLE = "PERSPECTIVE_TRANSFORM_ENABLE"
RGBENABLE = "RGB_OVERLAY_ENABLE"
DEBUG_FLAG = False
PTERROR_REPORT = True

windowName = "DepthSlice Tool"
slider1Name = "Slice depth (increments of 0.001)"
slider2Name = "Slice start (increments of 0.001)"
switchName = "0: Play\n1: Pause"

root = Tk()
root.withdraw()
root.overrideredirect(True)
root.geometry('0x0+0+0')
root.deiconify()
root.lift()
root.focus_force()
filename = askopenfilename(filetypes=[("Bag files", ".bag")], parent=root)
root.destroy()
if not filename:
    sys.exit("No file selected")

# Set up streaming pipeline
align = rs.align(rs.stream.depth)
colorizer = rs.colorizer()
pc = rs.pointcloud()
hole_filling = rs.hole_filling_filter()

config = rs.config()
rs.config.enable_device_from_file(config, filename)
config.enable_stream(rs.stream.depth)
config.enable_stream(rs.stream.color)

pipeline = rs.pipeline()
profile = pipeline.start(config)
device = profile.get_device()
playback = device.as_playback()
playback.seek(datetime.timedelta(seconds=32))
duration = playback.get_duration()
stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
intrinsics = stream.get_intrinsics()

# Global vars
slider1Arg = 0
slider2Arg = 0
slice1At = 0
slice2At = 0
scaling_factor = 0
savedFrame = None
isPaused = False
depthSelectEnabled = False
perspectiveSelectEnabled = False
rgbOverlayEnabled = False
rotationMatrix = None
fulcrumPixel_idx = None
depthPoints = []
perspectivePoints = []
avgTorsoDepth = []
np_depth_frame_prev = None
np_depth_frame_prev_prev = None
PTError = None
PTAngle = None
PTAxis = None
fulcrumPixel_idx = None

def buttonHandler(*args):
    global depthSelectEnabled, perspectiveSelectEnabled, perspectivePoints, rgbOverlayEnabled
    if args[1] == DSENABLE:
        perspectiveSelectEnabled = False
        depthSelectEnabled = not depthSelectEnabled
    elif args[1] == PTENABLE:
        depthSelectEnabled = False
        perspectivePoints = []
        perspectiveSelectEnabled = not perspectiveSelectEnabled
    elif args[1] == RGBENABLE:
        rgbOverlayEnabled = not rgbOverlayEnabled

def mouseEvent(action, x, y, flags, *userdata):
    global depthPoints, depthSelectEnabled, perspectivePoints, perspectiveSelectEnabled
    if action == cv2.EVENT_LBUTTONDBLCLK:
        if depthSelectEnabled:
            depthPoints.append((x, y))
        # Perspective transform
        elif perspectiveSelectEnabled:
            if len(perspectivePoints) < 4:
                perspectivePoints.append((x, y))
            else:
                perspectivePoints = []
                perspectivePoints.append((x, y))

# Handle frame updates when depth sliders are changed
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
        isPaused = True
        playback.pause()
        
    else:
        isPaused = False
        playback.resume()
        
    
# Create opencv window with trackbars, tool buttons, and set the mouse action handler
cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar(slider1Name, windowName, 1500 if DEBUG_FLAG else 15, 1000, updateFrame)
cv2.createTrackbar(slider2Name, windowName, 0, 1500, updateFrame)
cv2.createTrackbar(switchName, windowName, 0, 1, playPause)
cv2.setMouseCallback(windowName, mouseEvent)
cv2.createButton("RGB Overlay (Only on original video)", buttonHandler, RGBENABLE, cv2.QT_PUSH_BUTTON|cv2.QT_NEW_BUTTONBAR)
cv2.createButton("Toggle Depth Selector", buttonHandler, DSENABLE, cv2.QT_PUSH_BUTTON|cv2.QT_NEW_BUTTONBAR)
cv2.createButton("Perspective Transformation", buttonHandler, PTENABLE, cv2.QT_PUSH_BUTTON|cv2.QT_NEW_BUTTONBAR)

frames = pipeline.wait_for_frames()

# Streaming loop
while True:
    
    if not isPaused:
        frames = pipeline.wait_for_frames()
                  
    aligned_frames = align.process(frames)

    # Get depth and color frames
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue
        
    scaling_factor = depth_frame.get_units()
    
    # Handle pausing without losing information from the paused frame
    if not isPaused:
        np_color_frame = np.asanyarray(color_frame.get_data())
        savedFrame = copy.deepcopy(np_color_frame)
    else:
        np_color_frame = copy.deepcopy(savedFrame)
    np_color_frame = np_color_frame[...,::-1]
    np_depth_frame = np.asanyarray(depth_frame.get_data())
        
    if len(perspectivePoints) == 4:
        if(DEBUG_FLAG):
            start_time = time.time()
        
        np_depth_frame, contours, contours_filteredArea, contours_filteredCircularity, headSphere, maxHeadSlice, torsoSphere, rotationMatrix, fulcrumPixel_idx, errs = libdst.PTwithCrossSection(intrinsics, np_depth_frame, perspectivePoints, scaling_factor, pc, rotationMatrix, fulcrumPixel_idx, isPaused, np_depth_frame_prev, np_depth_frame_prev_prev, PTError, PTAngle, PTAxis, DEBUG_FLAG)
        PTError, PTAngle, PTAxis = errs
        
        # np_depth_frame_prev = np_depth_frame.copy()
        
        if(DEBUG_FLAG):
            print("--- {}s seconds ---".format((time.time() - start_time)))
    else:
        np_depth_frame = np.asanyarray(depth_frame.get_data())
        
    np_depth_frame_scaled = np_depth_frame * scaling_factor
            
    np_depth_color_frame = cv2.applyColorMap(cv2.convertScaleAbs(np_depth_frame, alpha=0.03), cv2.COLORMAP_TURBO)
        
    # Make boolean mask for a depth slice
    # np_depth_frame_scaled_copy = np_depth_frame_scaled.copy()
    sliceEnd = slice1At + slice2At
    np_depth_frame_bool1 = (np_depth_frame_scaled < sliceEnd) * 1
    np_depth_frame_bool2 = (np_depth_frame_scaled > slice2At) * 1
    np_depth_frame_bool = np.bitwise_and(np_depth_frame_bool1, np_depth_frame_bool2)
    
    np_depth_color_frame_masked = np_depth_color_frame.copy()
    np_color_frame_masked = np_color_frame.copy()
    
    # Slice the color frame using the boolean mask
    for i in range(0, 3):
        np_depth_color_frame_masked[:, :, i] = np_depth_color_frame_masked[:, :, i] * np_depth_frame_bool
    
    slider1Arg, slice1At, slider2Arg, slice2At = updateFrame(0)
    finalDepthImage = np_depth_color_frame_masked
    libdst.displayDepthPoints(np_depth_frame_scaled, finalDepthImage, depthPoints, DEBUG_FLAG)
    
    if len(perspectivePoints) == 4:
        
        for cons in contours_filteredArea:
            finalDepthImage = cv2.drawContours(finalDepthImage, cons, -1, (255,0,255), 2)
        
        # if maxHeadSlice is not None:
        #     for i in range(maxHeadSlice):
        #         finalDepthImage = cv2.drawContours(finalDepthImage, contours_filteredArea[i], -1, (0,0,255), 2)
            
        # for cons in contours_filteredCircularity:
        #     finalDepthImage = cv2.drawContours(finalDepthImage, cons, -1, (0,0,255), 2)
            
        # Display final headsphere contours
        if headSphere is not None:
            finalDepthImage = cv2.drawContours(finalDepthImage, headSphere, -1, (255, 0, 0), 2)
            
        if torsoSphere is not None:
            finalDepthImage = cv2.drawContours(finalDepthImage, torsoSphere, -1, (0,0,255), 2)
            # finalDepthImage = cv2.drawContours(finalDepthImage, [torsoSphere[-1]], -1, (0,0,255), -1)
            
            roi = np.ones(np_depth_frame.shape)
            roi = cv2.drawContours(roi, [torsoSphere[-1]], -1, 0, -1)
            
            np_ma_torsoROI = np.ma.masked_array(np_depth_frame, mask=roi)
            if (DEBUG_FLAG):
                print("torsoROI Mean: {}, Time: {}".format(np_ma_torsoROI.mean(), aligned_frames.get_timestamp()))
            if not isPaused:
                avgTorsoDepth.append([np_ma_torsoROI.mean(), aligned_frames.get_timestamp()])
            
        # for cons in contours_filteredRectangularity:
        #     finalDepthImage = cv2.drawContours(finalDepthImage, cons, -1, (255,0,0), 1)
    
    output_image = finalDepthImage
    
    if rgbOverlayEnabled and len(perspectivePoints) != 4:
        for i in range(0, 3):
            np_color_frame_masked[:, :, i] = np_color_frame_masked[:, :, i] * np_depth_frame_bool
            
        finalColorImage = np_color_frame_masked
        libdst.displayDepthPoints(np_depth_frame_scaled, finalColorImage, depthPoints, DEBUG_FLAG)
        output_image = finalColorImage
    
    # Render image in opencv window
    cv2.imshow(windowName, output_image)

    # If user presses ESCAPE or clicks the close button, end    
    key = cv2.waitKey(1)
    if (key == 27) or (cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) != 1):
        if len(avgTorsoDepth) > 0:
            avgTorsoDepth_filename = os.path.splitext(filename)[0] + "_TorsoROIDepth.csv"
            # with open(avgTorsoDepth_filename, 'w') as f:
            #     csvWriter = csv.writer(f)
            #     csvWriter.writerow(["Mean Depth", "Timestamp"])
            #     csvWriter.writerows(avgTorsoDepth)
                
        if PTError is not None:
            PTError_filename = os.path.splitext(filename)[0] + "_PerspectiveTransformError.csv"
            with open(PTError_filename, 'w') as f:
                csvWriter = csv.writer(f)
                csvWriter.writerow(["Angle (rad)", "Axis", "Absolute Error (%)"])
                csvWriter.writerow([PTAngle, PTAxis, PTError])
        cv2.destroyAllWindows()
        break
    

