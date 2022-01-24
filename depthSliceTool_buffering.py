# -*- coding: utf-8 -*-
#%%
"""
Created on Wed Sep 29 12:36:34 2021

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
END_OF_BUFFER = False

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
isPaused = True
depthSelectEnabled = False
perspectiveSelectEnabled = False
rgbOverlayEnabled = False
rotationMatrix = None
fulcrumPixel_idx = None
depthPoints = []
perspectivePoints = []
avgTorsoDepth = [[0, 0]]
np_depth_frame_prev = None
np_depth_frame_prev_prev = None
PTError = None
PTAngle = None
PTAxis = None
AAtest = None
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
cv2.createTrackbar(switchName, windowName, 1, 1, playPause)
cv2.setMouseCallback(windowName, mouseEvent)
cv2.createButton("RGB Overlay (Only on original video)", buttonHandler, RGBENABLE, cv2.QT_PUSH_BUTTON|cv2.QT_NEW_BUTTONBAR)
cv2.createButton("Toggle Depth Selector", buttonHandler, DSENABLE, cv2.QT_PUSH_BUTTON|cv2.QT_NEW_BUTTONBAR)
cv2.createButton("Perspective Transformation", buttonHandler, PTENABLE, cv2.QT_PUSH_BUTTON|cv2.QT_NEW_BUTTONBAR)

def bufferVideo(nFrames):
    global AAtest
    print("Buffering {} frames".format(nFrames))
    
    depth_frames = []
    color_frames = []
    timestamps = []
    epochtime = []
    for i in range(nFrames):
        frame = pipeline.wait_for_frames()
        aligned_frame = align.process(frame)
        depth_frame = aligned_frame.get_depth_frame()
        color_frame = aligned_frame.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        
        np_depth_frame = np.asanyarray(depth_frame.get_data())
        np_color_frame = np.asanyarray(color_frame.get_data())
        depth_frames.append(np_depth_frame.copy())
        color_frames.append(np_color_frame.copy())
        epochtime.append(aligned_frame.get_frame_metadata(rs.frame_metadata_value.time_of_arrival) // 100)
        frameTime = aligned_frame.get_frame_metadata(rs.frame_metadata_value.time_of_arrival) // 1000
        systemTime = datetime.datetime.fromtimestamp(frameTime)
        timestamps.append(systemTime)

        # AAtest = aligned_frame.get_frame_metadata(rs.frame_metadata_value.time_of_arrival)
        
    return depth_frames, color_frames, timestamps, depth_frame.get_units(), epochtime

depth_frames, color_frames, timestamps, scaling_factor, epochtime = bufferVideo(18500)
    
savedTimestamp = None
# Streaming loop
frameCounter = 0
while frameCounter < len(depth_frames):
    print(frameCounter)
    np_depth_frame = depth_frames[frameCounter]
    np_color_frame = color_frames[frameCounter]
    
    # Handle pausing without losing information from the paused frame
    if not isPaused:
        frameCounter = frameCounter+1
        
    if frameCounter == len(depth_frames):
        frameCounter = 0
        
    # ONLY FOR TESTING
    if frameCounter == len(depth_frames)-1:
        END_OF_BUFFER = True

    # if not isPaused:
    #     # ONLY FOR TESTING
    #     if savedTimestamp == timestamps[frameCounter]:
    #         continue
    #     else:
    #         savedTimestamp = timestamps[frameCounter]

        
    np_color_frame = np_color_frame[...,::-1]
        
    if len(perspectivePoints) == 4:
        if(DEBUG_FLAG):
            start_time = time.time()
        
        np_depth_frame, contours, contours_filteredArea, contours_filteredCircularity, headSphere, maxHeadSlice, torsoSphere, rotationMatrix, fulcrumPixel_idx, errs = libdst.perspectiveTransformHandler(intrinsics, np_depth_frame, perspectivePoints, scaling_factor, pc, rotationMatrix, fulcrumPixel_idx, isPaused, np_depth_frame_prev, np_depth_frame_prev_prev, PTError, PTAngle, PTAxis, DEBUG_FLAG)
        PTError, PTAngle, PTAxis = errs
        # np_depth_frame = perspectiveTransformHandler(intrinsics, depth_frame, perspectivePoints)
        
        # np_depth_frame_prev = np_depth_frame.copy()
        
        if(DEBUG_FLAG):
            print("--- {}s seconds ---".format((time.time() - start_time)))
        
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
            # print("Head contours:")
            for c in headSphere:
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                area = cv2.contourArea(c)
                perimeter = cv2.arcLength(c, True)
                circularity = 4*math.pi*(area/(perimeter*perimeter))
                equi_diameter = np.sqrt(4*area/np.pi)
            #     print("Center: ({}, {}); Area: {}; Equivalent diameter: {}; Circularity: {}".format(cX, cY, area, equi_diameter, circularity))
            # print("")
           
        if torsoSphere is not None:
            finalDepthImage = cv2.drawContours(finalDepthImage, torsoSphere, -1, (0,0,255), 2)
            # finalDepthImage = cv2.drawContours(finalDepthImage, [torsoSphere[-1]], -1, (0,0,255), -1)
            # print("Torso contours:")
            for c in torsoSphere:
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                area = cv2.contourArea(c)
                perimeter = cv2.arcLength(c, True)
                circularity = 4*math.pi*(area/(perimeter*perimeter))
                equi_diameter = np.sqrt(4*area/np.pi)
            #     print("Center: ({}, {}); Area: {}; Equivalent diameter: {}; Circularity: {}".format(cX, cY, area, equi_diameter, circularity))
            # print("")
            
            roi = np.ones(np_depth_frame.shape)
            roi = cv2.drawContours(roi, [torsoSphere[-1]], -1, 0, -1)
            
            np_ma_torsoROI = np.ma.masked_array(np_depth_frame, mask=roi)
            if (DEBUG_FLAG):
                print("torsoROI Mean: {}, Time: {}".format(np_ma_torsoROI.mean(), timestamps[frameCounter]))

            # # Keep this or the one after, not both
            # if not isPaused and timestamps[frameCounter] != avgTorsoDepth[-1][0]:
            #     avgTorsoDepth.append([timestamps[frameCounter], np_ma_torsoROI.mean()])

            if not isPaused:
                avgTorsoDepth.append([timestamps[frameCounter], np_ma_torsoROI.mean()])
        
        # Anthropomorphic checks
        # TURNED OFF TO SAVE TIME AND RECORD AVERAGE TORSO DEPTH
        # if torsoSphere is not None and headSphere is not None:

        #     MHead = cv2.moments(headSphere[-1])
        #     cXHead = int(MHead["m10"] / MHead["m00"])
        #     cYHead = int(MHead["m01"] / MHead["m00"])

        #     MTorso = cv2.moments(torsoSphere[-1])
        #     cXTorso = int(MTorso["m10"] / MTorso["m00"])
        #     cYTorso = int(MTorso["m01"] / MTorso["m00"])
        #     centerDistance = np.sqrt(((cXTorso - cXHead) ** 2) + ((cYTorso - cYHead) ** 2))
        #     # print("Distance between centers of the largest contours: {}".format(centerDistance))

        #     slope = (cYTorso - cYHead) / (cXTorso - cXHead)

        #     line = lambda x : cYHead + (slope * (x - cXHead))

        #     def testContourLine(contour, tolerance=3, line=line):
        #         distances, ptsOnLine = [], []
        #         for c in contour:
        #             # if (abs(c[0][1] - line(c[0][0])) <= tolerance):
        #             #     ptsOnLine.append((c[0][0], c[0][1]))
        #             distances.append(abs(c[0][1] - line(c[0][0])))
        #             ptsOnLine.append((c[0][0], c[0][1]))

        #         return distances, ptsOnLine

        #     headDists, headPts = testContourLine(headSphere[-1])
        #     torsoDists, torsoPts = testContourLine(torsoSphere[-1])
        #     # print(headDists[0])
        #     # print(torsoDists[0])

        #     headDists, headPts = (list(t) for t in zip(*sorted(zip(headDists, headPts))))
        #     torsoDists, torsoPts = (list(t) for t in zip(*sorted(zip(torsoDists, torsoPts))))
        #     headConPt = None
        #     torsoConPt = None
        #     for pt in headPts:
        #         if ((pt[0] >= min(cXHead, cXTorso) and pt[0] <= max(cXHead, cXTorso))) and ((pt[1] >= min(cYHead, cYTorso) and pt[1] <= max(cYHead, cYTorso))):
        #             headConPt = pt
        #             break

        #     for pt in torsoPts:
        #         if ((pt[0] >= min(cXHead, cXTorso) and pt[0] <= max(cXHead, cXTorso))) and ((pt[1] >= min(cYHead, cYTorso) and pt[1] <= max(cYHead, cYTorso))):
        #             torsoConPt = pt
        #             break

        #     # print(headDists[0])
        #     # print(torsoDists[0])
        #     # print(headConPt)
        #     # print(torsoConPt)

        #     finalDepthImage = cv2.line(finalDepthImage, headConPt, torsoConPt, (0, 0, 0), thickness=2)
        #     neckDistance = math.hypot(headConPt[0]-torsoConPt[0], headConPt[1]-torsoConPt[1])
        #     # print("Neck distance: {}".format(neckDistance))
        #     headEllipse = cv2.fitEllipse(headSphere[-1])
        #     # print(headEllipse)
        #     torsoEllipse = cv2.fitEllipse(torsoSphere[-1])
        #     # print(torsoEllipse[-1])
        #     # print(math.degrees(math.atan(slope)) + 90)

        #     # Evaluate fitted ellipse (compare to the detected ROIs)
        #     torsoROI_detected_arr = np.zeros(finalDepthImage.shape[:2])
        #     torsoROI_detected_arr = cv2.drawContours(torsoROI_detected_arr, np.array([torsoSphere[-1]]), 0, 1, -1)
        #     torsoROI_detected_arr = torsoROI_detected_arr != 0

        #     torsoROI_ellipse_arr = np.zeros(finalDepthImage.shape[:2])
        #     torsoROI_ellipse_arr = cv2.ellipse(torsoROI_ellipse_arr, torsoEllipse, 1, -1)
        #     torsoROI_ellipse_arr = torsoROI_ellipse_arr != 0

        #     dice = lambda ellipse, detected : np.sum(ellipse[detected==1])*2.0 / (np.sum(ellipse) + np.sum(detected))
        #     dice_torso_ellipse = dice(torsoROI_ellipse_arr, torsoROI_detected_arr)
        #     # print("Dice-Sorenson of torso (taking detected as Actual and ellipse as Predicted) is: {}".format(dice_torso_ellipse))

        #     # finalDepthImage = cv2.ellipse(finalDepthImage, headEllipse, (0, 0, 0), 2)
        #     finalDepthImage = cv2.ellipse(finalDepthImage, torsoEllipse, (255, 255, 255), 2)

        
        #     # print(headSphere[-1][n][0])

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
    if (key == 27) or (cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) != 1) or END_OF_BUFFER:
        if len(avgTorsoDepth) > 0:
            avgTorsoDepth_filename = os.path.splitext(filename)[0] + "_TorsoROIDepth.csv"
            with open(avgTorsoDepth_filename, 'w') as f:
                csvWriter = csv.writer(f)
                csvWriter.writerow(["Timestamp", "Mean Depth"])
                csvWriter.writerows(avgTorsoDepth)
                
        if PTError is not None:
            PTError_filename = os.path.splitext(filename)[0] + "_PerspectiveTransformError.csv"
            with open(PTError_filename, 'w') as f:
                csvWriter = csv.writer(f)
                csvWriter.writerow(["Angle (rad)", "Axis", "Absolute Error (%)"])
                csvWriter.writerow([PTAngle, PTAxis, PTError])
        cv2.destroyAllWindows()
        break
        
# %%
