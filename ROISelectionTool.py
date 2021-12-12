# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 14:00:48 2021

@author: Zein Hajj-Ali - zeinhajjali@sce.carleton.ca
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import datetime
import time
import sys
import math
import os
import random
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from PIL import Image
import libdst

PTENABLE = "PERSPECTIVE_TRANSFORM_ENABLE"
STAGEONE_FLAG = "STAGE_ONE"
STAGETWO_FLAG = "STAGE_TWO"
STAGETHREE_FLAG = "STAGE_THREE"
DEBUG_FLAG = False
PTERROR_REPORT = True
SELECT_MIDDLE_FRAME = True

windowName = "ROI Selection/Comparision Tool"

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
scaling_factor = 0
savedFrame = None
perspectiveSelectEnabled = False
currStage = STAGEONE_FLAG
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
drawingHeadFinished = False
stageOne_headROI = None
stageOne_headPts = []
drawingTorsoFinished = False
stageTwo_torsoROI = None
stageTwo_torsoPts = []

def mouseEvent(action, x, y, flags, *userdata):
    global perspectivePoints, perspectiveSelectEnabled, stageOne_headPts, drawingHeadFinished, stageTwo_torsoPts, drawingTorsoFinished, currStage
    if currStage == STAGEONE_FLAG:
        if action == cv2.EVENT_RBUTTONDOWN:
            if len(stageOne_headPts) < 3:
                print("You need a minimum of three points")
                return
            if drawingHeadFinished:
                currStage = STAGETWO_FLAG
                print("Confirmed polygon")
                print("Moving to stage 2")
            else:
                print("Closed polygon") 
                print("Right-click again to confirm")
                print("Left-click to start over")
                drawingHeadFinished = True
        elif action == cv2.EVENT_LBUTTONDOWN:
            if not drawingHeadFinished:
                print("Point selected at ({}, {})".format(x, y))
                stageOne_headPts.append((x, y))
            else:
                print("Resetting points")
                stageOne_headPts = []
                drawingHeadFinished = False
    elif currStage == STAGETWO_FLAG:
        if action == cv2.EVENT_RBUTTONDOWN:
            if len(stageTwo_torsoPts) < 3:
                print("You need a minimum of three points")
                return
            if drawingTorsoFinished:
                currStage = STAGETHREE_FLAG
                print("Confirmed polygon")
                print("Moving to stage 3")
            else:
                print("Closed polygon") 
                print("Right-click again to confirm")
                print("Left-click to start over")
                drawingTorsoFinished = True
        elif action == cv2.EVENT_LBUTTONDOWN:
            if not drawingTorsoFinished:
                print("Point selected at ({}, {})".format(x, y))
                stageTwo_torsoPts.append((x, y))
            else:
                print("Resetting points")
                stageTwo_torsoPts = []
                drawingTorsoFinished = False
    else:
        if action == cv2.EVENT_LBUTTONDBLCLK:
            # Perspective transform
            if perspectiveSelectEnabled:
                if len(perspectivePoints) < 4:
                    perspectivePoints.append((x, y))
                else:
                    perspectivePoints = []
                    perspectivePoints.append((x, y))
                
# Create opencv window with trackbars, tool buttons, and set the mouse action handler
cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE|cv2.WINDOW_GUI_NORMAL)
cv2.setMouseCallback(windowName, mouseEvent)

def bufferVideo(nFrames):
    print("Buffering {} frames".format(nFrames))
    
    depth_frames = []
    color_frames = []
    timestamps = []
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
        timestamps.append(aligned_frame.get_timestamp())
        
    return depth_frames, color_frames, timestamps, depth_frame.get_units()

depth_frames, color_frames, timestamps, scaling_factor = bufferVideo(90)
    
# Streaming loop
frameCounter = int(len(depth_frames)/2) if SELECT_MIDDLE_FRAME else random.randrange(0, len(depth_frames))
print(frameCounter)
savedMaskToggle = False
savedDepthImage = False

pathToResults = os.path.splitext(filename)[0] + "_results/"
try:
    os.mkdir(pathToResults)
except OSError:
    pass

np_color_frame = color_frames[frameCounter]

np_depth_frame = depth_frames[frameCounter]
im = Image.fromarray(np_color_frame)
im.save(pathToResults + "color_frame.jpg")
np_color_frame = np_color_frame[...,::-1]
finalDepthImage_output = None

while True:
    
    # Stage 1: Manual head ROI selection
    if currStage == STAGEONE_FLAG:
        output_image = np_color_frame.copy()
        
        len_headPts = len(stageOne_headPts)
        # Show selected polygon points and connections
        if len_headPts > 0:
            output_image = cv2.circle(output_image, stageOne_headPts[0], radius=0, color=(255, 0, 0), thickness=-1)
        if len_headPts > 1:
            for i in range(1, len_headPts):
                output_image = cv2.line(output_image, stageOne_headPts[i], stageOne_headPts[i-1], (255, 0, 0), thickness=1)
        if drawingHeadFinished:
            output_image = cv2.line(output_image, stageOne_headPts[-1], stageOne_headPts[0], (255, 0, 0), thickness=1)
            
            # Show completed polygon (with alpha channel for some transparancy)
            overlay = output_image.copy()
            overlay = cv2.drawContours(overlay, np.array([stageOne_headPts]), 0, (255, 0, 0), -1)
            alpha = 0.2
            output_image = cv2.addWeighted(overlay, alpha, output_image, 1-alpha, 0)
            
            # Save area of completed polygon in a numpy masked array for comparison with automatic method
            stageOne_headROI = np.zeros(np_color_frame.shape[:2])
            stageOne_headROI = cv2.drawContours(stageOne_headROI, np.array([stageOne_headPts]), 0, 255, -1)
           
    # Stage 2: Manual torso ROI selection
    elif currStage == STAGETWO_FLAG:
        if not savedMaskToggle:
            im = Image.fromarray(stageOne_headROI)
            im = im.convert("L")
            im.save(pathToResults + "headROI_manual.jpg")
            savedMaskToggle = True
            
        len_torsoPts = len(stageTwo_torsoPts)
        # Show selected polygon points and connections
        if len_torsoPts > 0:
            output_image = cv2.circle(output_image, stageTwo_torsoPts[0], radius=0, color=(0, 0, 255), thickness=-1)
        if len_torsoPts > 1:
            for i in range(1, len_torsoPts):
                output_image = cv2.line(output_image, stageTwo_torsoPts[i], stageTwo_torsoPts[i-1], (0, 0, 255), thickness=1)
        if drawingTorsoFinished:
            output_image = cv2.line(output_image, stageTwo_torsoPts[-1], stageTwo_torsoPts[0], (0, 0, 255), thickness=1)
            
            # Show completed polygon (with alpha channel for some transparancy)
            overlay = output_image.copy()
            overlay = cv2.drawContours(overlay, np.array([stageTwo_torsoPts]), 0, (0, 0, 255), -1)
            alpha = 0.2
            output_image = cv2.addWeighted(overlay, alpha, output_image, 1-alpha, 0)
            
            # Save area of completed polygon in a numpy masked array for comparison with automatic method
            stageTwo_torsoROI = np.zeros(np_color_frame.shape[:2])
            stageTwo_torsoROI = cv2.drawContours(stageTwo_torsoROI, np.array([stageTwo_torsoPts]), 0, 255, -1)
        
    # Stage 3: Automatic head and torso ROI selection (as in depthSliceTool)
    else:
        if finalDepthImage_output is None:
            if savedMaskToggle:
                im = Image.fromarray(stageTwo_torsoROI)
                im = im.convert("L")
                im.save(pathToResults + "torsoROI_manual.jpg")
                savedMaskToggle = False
            
            np_depth_frame_orig = np_depth_frame.copy()
            if len(perspectivePoints) < 4:
                perspectiveSelectEnabled = True
            if len(perspectivePoints) == 4:
                if(DEBUG_FLAG):
                    start_time = time.time()
                
                np_depth_frame, contours, contours_filteredArea, contours_filteredCircularity, headSphere, maxHeadSlice, torsoSphere, rotationMatrix, fulcrumPixel_idx, errs  = libdst.perspectiveTransformHandler(intrinsics, np_depth_frame, perspectivePoints, scaling_factor, pc, rotationMatrix, fulcrumPixel_idx, True, np_depth_frame_prev, np_depth_frame_prev_prev, PTError, PTAngle, PTAxis, DEBUG_FLAG)
                PTError, PTAngle, PTAxis = errs

                print(errs)
                print(rotationMatrix)

                inv_rotMat = np.linalg.inv(rotationMatrix)
                            
                if(DEBUG_FLAG):
                    print("--- {}s seconds ---".format((time.time() - start_time)))
                
            np_depth_frame_scaled = np_depth_frame * scaling_factor
            np_depth_frame_orig_scaled = np_depth_frame_orig * scaling_factor
                    
            np_depth_color_frame = cv2.applyColorMap(cv2.convertScaleAbs(np_depth_frame, alpha=0.03), cv2.COLORMAP_TURBO)
            np_depth_color_frame_orig = cv2.applyColorMap(cv2.convertScaleAbs(np_depth_frame_orig, alpha=0.03), cv2.COLORMAP_TURBO)
            
            finalDepthImage_PT = np_depth_color_frame
            finalDepthImage = np_depth_color_frame_orig
            finalColorImage = np_color_frame.copy()
            
            if not savedDepthImage:
                im = Image.fromarray(finalDepthImage)
                im.save(pathToResults + "depth_frame.jpg")
                savedDepthImage = True
            
            if len(perspectivePoints) == 4:
                    
                # Display final headsphere contours
                if headSphere is not None:
                    finalDepthImage_PT = cv2.drawContours(finalDepthImage_PT, headSphere, -1, (255, 0, 0), 2)
                    
                    # Get points of head contour after PT
                    headContour_pts = []
                    for px in headSphere[-1]:
                        depth = np_depth_frame[px[0][1],px[0][0]]
                        point = rs.rs2_deproject_pixel_to_point(intrinsics, (px[0][0], px[0][1]), depth)
                        headContour_pts.append(point)
                    
                    # Apply inverse rotation matrix to PT head contour points to get points at original angle
                    np_headContour_pts = np.asanyarray(headContour_pts)
                    np_headContour_pts_transformed = inv_rotMat.dot(np_headContour_pts.T).T
                    
                    # Project original angle head contour points back to pixels
                    headContour_pixels = []
                    for pt in np_headContour_pts_transformed:
                        pixel = rs.rs2_project_point_to_pixel(intrinsics, pt)
                        headContour_pixels.append(pixel)
                        
                    headContour_pixels = np.asanyarray(headContour_pixels)
                    headContour_pixels = np.absolute(headContour_pixels)
                    headContour_pixels = headContour_pixels.astype(int)
                    
                    finalDepthImage = np_depth_color_frame_orig
                    
                    for i in range(1, len(headContour_pixels)):
                        finalDepthImage = cv2.line(finalDepthImage, headContour_pixels[i], headContour_pixels[i-1], (255, 0, 0), thickness=1)
                        
                    finalDepthImage = cv2.line(finalDepthImage, headContour_pixels[-1], headContour_pixels[0], (255, 0, 0), thickness=1)
                    overlay = finalDepthImage.copy()
                    overlay = cv2.drawContours(overlay, np.array([headContour_pixels]), 0, (255, 0, 0), -1)
                    alpha = 0.2
                    finalDepthImage = cv2.addWeighted(overlay, alpha, finalDepthImage, 1-alpha, 0)
                        
                    stageThree_headROI = np.zeros(finalDepthImage.shape[:2])
                    stageThree_headROI = cv2.drawContours(stageThree_headROI, np.array([headContour_pixels]), 0, 255, -1)
                    
                    if not savedMaskToggle:
                        im = Image.fromarray(stageThree_headROI)
                        im = im.convert("L")
                        im.save(pathToResults + "headROI_auto.jpg")
                        savedMaskToggle = True
                    
                # Display final torsoSphere contours
                if torsoSphere is not None:
                    finalDepthImage_PT = cv2.drawContours(finalDepthImage_PT, torsoSphere, -1, (0, 0, 255), 2)
                    
                    # Get points of head contour after PT
                    torsoContour_pts = []
                    for px in torsoSphere[-1]:
                        depth = np_depth_frame[px[0][1],px[0][0]]
                        point = rs.rs2_deproject_pixel_to_point(intrinsics, (px[0][0], px[0][1]), depth)
                        torsoContour_pts.append(point)
                    
                    # Apply inverse rotation matrix to PT head contour points to get points at original angle
                    np_torsoContour_pts = np.asanyarray(torsoContour_pts)
                    np_torsoContour_pts_transformed = inv_rotMat.dot(np_torsoContour_pts.T).T
                    
                    # Project original angle head contour points back to pixels
                    torsoContour_pixels = []
                    for pt in np_torsoContour_pts_transformed:
                        pixel = rs.rs2_project_point_to_pixel(intrinsics, pt)
                        torsoContour_pixels.append(pixel)
                        
                    torsoContour_pixels = np.asanyarray(torsoContour_pixels)
                    torsoContour_pixels = np.absolute(torsoContour_pixels)
                    torsoContour_pixels = torsoContour_pixels.astype(int)
                    
                    
                    for i in range(1, len(torsoContour_pixels)):
                        finalDepthImage = cv2.line(finalDepthImage, torsoContour_pixels[i], torsoContour_pixels[i-1], (0, 0, 255), thickness=1)
                    
                    finalDepthImage = cv2.line(finalDepthImage, torsoContour_pixels[-1], torsoContour_pixels[0], (0, 0, 255), thickness=1)
                    overlay = finalDepthImage.copy()
                    overlay = cv2.drawContours(overlay, np.array([torsoContour_pixels]), 0, (0, 0, 255), -1)
                    alpha = 0.2
                    finalDepthImage = cv2.addWeighted(overlay, alpha, finalDepthImage, 1-alpha, 0)
                        
                    stageThree_torsoROI = np.zeros(finalDepthImage.shape[:2])
                    stageThree_torsoROI = cv2.drawContours(stageThree_torsoROI, np.array([torsoContour_pixels]), 0, 255, -1)
                    
                    if savedMaskToggle:
                        im = Image.fromarray(stageThree_torsoROI)
                        im = im.convert("L")
                        im.save(pathToResults + "torsoROI_auto.jpg")
                        savedMaskToggle = False
                        finalDepthImage_output = finalDepthImage.copy()
        
            output_image = finalDepthImage
            
        else:
            output_image = finalDepthImage_output
        
    # Render image in opencv window
    cv2.imshow(windowName, output_image)
    
    # If user presses ESCAPE or clicks the close button, end    
    key = cv2.waitKey(1)
    if (key == 27) or (cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) != 1):
        
        
        
        cv2.destroyAllWindows()
        break
        