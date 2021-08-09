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
from tkinter import Tk
from tkinter.filedialog import askopenfilename

DSENABLE = "DEPTH_SELECT_ENABLE"
PTENABLE = "PERSPECTIVE_TRANSFORM_ENABLE"

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
depthPoints = []
perspectivePoints = []

def buttonHandler(*args):
    global depthSelectEnabled, perspectiveSelectEnabled, perspectivePoints
    if args[1] == DSENABLE:
        perspectiveSelectEnabled = False
        depthSelectEnabled = not depthSelectEnabled
    elif args[1] == PTENABLE:
        depthSelectEnabled = False
        perspectivePoints = []
        perspectiveSelectEnabled = not perspectiveSelectEnabled

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
                
def displayDepthPoints(depth_frame_scaled, output):
    global depthPoints
    # Display depth of selected points on frame
    for point in depthPoints:
        depth = depth_frame_scaled[point[1], point[0]].astype(str)
        cv2.putText(output,
                    depth,
                    point, 
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.3,
                    color=(255,255,255),
                    )
        print("x: " + str(point[0]) + " y: " + str(point[1]) + " Depth: " + depth)
        
def npShift(arr, numX, numY, fill_value=0):
    result = np.empty_like(arr)
    if numX > 0:
        result[:numX,:] = fill_value
        result[numX:,:] = arr[:-numX,:]
    elif numX < 0:
        result[numX:,:] = fill_value
        result[:numX,:] = arr[-numX:,:]
    else:
        result[:] = arr
        
    if numY > 0:
        result[:,:numY] = fill_value
        result[:,numY:] = arr[:,:-numY]
    elif numX < 0:
        result[:,numY:] = fill_value
        result[:,:numY] = arr[:,-numY:]
    else:
        result[:] = result
        
    return result

def calculateRotationMatrix(points):
    rMatrices = []
    tpDiffs = []
    for pointIndex in range(len(points)):
        vAB = np.subtract(points[(pointIndex + 1) % 4], points[pointIndex])
        vAC = np.subtract(points[(pointIndex + 3) % 4], points[pointIndex])
        normalVector = np.cross(vAB, vAC)
        normalVector = normalVector / np.linalg.norm(normalVector)
        # print(normalVector)
        newNormal = np.array([0, 0, -1])
        rAxis = np.cross(normalVector, newNormal)
        rAxis = rAxis / np.linalg.norm(rAxis)
        rAngle = np.arccos(np.dot(normalVector, newNormal))
        # print(rAxis)
        # print(rAngle)
        rAxisCMatrix = np.array([[0, -rAxis[2], rAxis[1]],
                                 [rAxis[2], 0, -rAxis[0]],
                                 [-rAxis[1], rAxis[0], 0]])
        rotationMatrix = (np.cos(rAngle)*np.identity(3)) + ((np.sin(rAngle)*rAxisCMatrix) +((1-np.cos(rAngle))*(np.outer(rAxis, rAxis))))
        # print(rotationMatrix)
        rMatrices.append(rotationMatrix)
        
        testPoints = rotationMatrix.dot(np.asanyarray(points).T).T
        testPointDiff = testPoints[(pointIndex + 2 ) % 4, 2] - testPoints[pointIndex, 2]
        tpDiffs.append(abs(testPointDiff))
        
    temp = min(tpDiffs)
    minIdx = [i for i, j in enumerate(tpDiffs) if j == temp]
    # print(minIdx)
    return rMatrices[minIdx[0] if isinstance(minIdx, list) else minIdx]
        
def perspectiveTransformHandler(intrinsics, depth_frame, perspectivePoints):
    global pc
    points = []
    camera_intrinsic_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                                        [0, intrinsics.fy, intrinsics.ppy],
                                        [0, 0, 1]])
    camera_rotation_matrix = np.identity(3)
    camera_translation_matrix = np.array([0.0, 0.0, 0.0])
    distortion_coeffs = np.asanyarray(intrinsics.coeffs)
    np_depth_frame = np.asanyarray(depth_frame.get_data())
    
    for pixel in perspectivePoints:
        depth = np_depth_frame[pixel[1],pixel[0]]
        point = rs.rs2_deproject_pixel_to_point(intrinsics, pixel, depth)
        points.append(point)
        
    # print(perspectivePoints[3])
    # print(np_depth_frame[perspectivePoints[3][1], perspectivePoints[3][0]])
    
    rotationMatrix = calculateRotationMatrix(points)
    
    # Create translation matrix to center frame after rotation (DOESN'T SEEM TO WORK RIGHT NOW)
    # middlePixel = (int(intrinsics.width/2), int(intrinsics.height/2))
    # middleDepth = np_depth_frame[middlePixel[1], middlePixel[0]]
    # middlePoint = rs.rs2_deproject_pixel_to_point(intrinsics, middlePixel, middleDepth)
    # middlePointRotated = rotationMatrix.dot(np.asanyarray(middlePoint).T).T
    # middlePixelRotated = rs.rs2_project_point_to_pixel(intrinsics, middlePointRotated)
    # middlePointRotatedTranslated = rs.rs2_deproject_pixel_to_point(intrinsics, middlePixel, middlePointRotated[2])
    # translationVector = np.array([middlePointRotatedTranslated[0]-middlePointRotated[0],
    #                               middlePointRotatedTranslated[1]-middlePointRotated[1],
    #                               0])
    # translationVector = translationVector*scaling_factor
    # # print(middlePoint)
    # # print(middlePointRotated)
    # # print(middlePointRotatedTranslated)
    
    pcPoints = pc.calculate(depth_frame)
    np_verts = np.asanyarray(pcPoints.get_vertices(dims=2))
    
    np_verts_transformed = rotationMatrix.dot(np_verts.T).T
    np_verts_transformed = np_verts_transformed[~np.all(np_verts_transformed == 0, axis=1)]
    np_verts_transformed = np_verts_transformed / scaling_factor
    # np_verts_transformed = np_verts_transformed - translationVector
    
    # project back to 2D image with depth as data (WORKING BUT SLOW)   
    np_transformed_depth_frame = np.zeros([1080,1920])
    for vert in np_verts_transformed:
        pixel = rs.rs2_project_point_to_pixel(intrinsics, vert)
        if (pixel[0] < 960 and pixel[1] < 540 and pixel[0] >= -960 and pixel[1] >= -540):
            np_transformed_depth_frame[int(pixel[1] + 540),int(pixel[0] + 960)] = vert[2]
    
            
    # project back to 2D image using openCV function instead (WORKING BUT SLOW)
    # np_transformed_depth_pixels, _ = cv2.projectPoints(np_verts_transformed,
    #                                                 camera_rotation_matrix,
    #                                                 camera_translation_matrix,
    #                                                 camera_intrinsic_matrix,
    #                                                 distortion_coeffs)    
    # np_transformed_depth_frame = np.zeros([1080, 1920])
    # for pixel, vert in zip(np_transformed_depth_pixels, np_verts_transformed):
    #     if (pixel[0,0] < 1920 and pixel[0,1] < 1080):
    #         np_transformed_depth_frame[int(pixel[0,1]), int(pixel[0,0])] = vert[2] / scaling_factor
            
    # Remove rows and columns of all zeros
    np_transformed_depth_frame = np_transformed_depth_frame[~np.all(np_transformed_depth_frame == 0, axis=1)]
    np_transformed_depth_frame = np_transformed_depth_frame[:, ~np.all(np_transformed_depth_frame == 0, axis=0)]
    
    # np_final_frame = np_transformed_depth_frame
    
    # OpenCV dilation
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    np_dilated_depth_frame = cv2.dilate(np_transformed_depth_frame, dilation_kernel)
    np_final_frame = np_dilated_depth_frame
    
    
    return np_final_frame


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
cv2.createTrackbar(slider1Name, windowName, 15, 1000, updateFrame)
cv2.createTrackbar(slider2Name, windowName, 0, 1500, updateFrame)
cv2.createTrackbar(switchName, windowName, 0, 1, playPause)
cv2.setMouseCallback(windowName, mouseEvent)
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
        
    if len(perspectivePoints) == 4:
        # pc.map_to(depth_frame)
        start_time = time.time()
        np_depth_frame = perspectiveTransformHandler(intrinsics, depth_frame, perspectivePoints)
        print("--- %s seconds ---" % (time.time() - start_time))
    else:
        np_depth_frame = np.asanyarray(depth_frame.get_data())
        
    np_depth_frame_scaled = np_depth_frame * scaling_factor
        
    # depth_color_frame = colorizer.colorize(depth_frame)
    # np_depth_color_frame = np.asanyarray(depth_color_frame.get_data())
    np_depth_color_frame = cv2.applyColorMap(cv2.convertScaleAbs(np_depth_frame, alpha=0.03), cv2.COLORMAP_TURBO)
        
    # Make boolean mask for a depth slice
    # np_depth_frame_scaled_copy = np_depth_frame_scaled.copy()
    sliceEnd = slice1At + slice2At
    np_depth_frame_bool1 = (np_depth_frame_scaled < sliceEnd) * 1
    np_depth_frame_bool2 = (np_depth_frame_scaled > slice2At) * 1
    np_depth_frame_bool = np.bitwise_and(np_depth_frame_bool1, np_depth_frame_bool2)
    
    np_color_frame_masked = np_depth_color_frame.copy()
    
    # Slice the color frame using the boolean mask
    for i in range(0, 3):
        np_color_frame_masked[:, :, i] = np_color_frame_masked[:, :, i] * np_depth_frame_bool
    
    slider1Arg, slice1At, slider2Arg, slice2At = updateFrame(0)
    
    finalImage = np_color_frame_masked.copy()
    # finalImage = np_color_frame.copy()
    
    displayDepthPoints(np_depth_frame_scaled, finalImage)
    
    output_image = finalImage
    
    # Render image in opencv window
    cv2.imshow(windowName, output_image)

    # If user presses ESCAPE or clicks the close button, end    
    key = cv2.waitKey(1)
    if (key == 27) or (cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) != 1):
        cv2.destroyAllWindows()
        break
    

