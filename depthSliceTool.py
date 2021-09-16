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

DSENABLE = "DEPTH_SELECT_ENABLE"
PTENABLE = "PERSPECTIVE_TRANSFORM_ENABLE"
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
rotationMatrix = None
fulcrumPixel_idx = None
depthPoints = []
perspectivePoints = []
avgTorsoDepth = []
PTError = None
PTAngle = None
PTAxis = None

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
        if (DEBUG_FLAG): 
            print("x: {}, y: {}, Depth: {}".format(point[0], point[1], depth))
        
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
    global PTError, PTAngle, PTAxis
    
    rMatrices = []
    rAngles = []
    rAxes = []
    tpDiffs = []
    tpComparision = []
    for pointIndex in range(len(points)):
        vAB = np.subtract(points[(pointIndex + 1) % 4], points[pointIndex])
        vAC = np.subtract(points[(pointIndex + 3) % 4], points[pointIndex])
        normalVector = np.cross(vAB, vAC)
        normalVector = normalVector / np.linalg.norm(normalVector)
        newNormal = np.array([0, 0, -1])
        rAxis = np.cross(normalVector, newNormal)
        rAxis = rAxis / np.linalg.norm(rAxis)
        rAngle = np.arccos(np.dot(normalVector, newNormal))
        rAxisCMatrix = np.array([[0, -rAxis[2], rAxis[1]],
                                 [rAxis[2], 0, -rAxis[0]],
                                 [-rAxis[1], rAxis[0], 0]])
        rotationMatrix = (np.cos(rAngle)*np.identity(3)) + ((np.sin(rAngle)*rAxisCMatrix) +((1-np.cos(rAngle))*(np.outer(rAxis, rAxis))))
        rMatrices.append(rotationMatrix)
        
        if (DEBUG_FLAG):
            print("Normal Vector: {}".format(normalVector))
            print("rAxis: {}".format(rAxis))
            print("rAngle: {}".format(rAngle))
            print("Rotation Matrix: {}".format(rotationMatrix))
        
        rAngles.append(rAngle)
        rAxes.append(np.array2string(rAxis))
        
        testPoints = rotationMatrix.dot(np.asanyarray(points).T).T
        testPointDiff = testPoints[(pointIndex + 2 ) % 4, 2] - testPoints[pointIndex, 2]
        tpComparision.append(testPoints[pointIndex, 2])
        tpDiffs.append(abs(testPointDiff))
        
    temp = min(tpDiffs)
    minIdx = [i for i, j in enumerate(tpDiffs) if j == temp]
    minIdx = minIdx[0] if isinstance(minIdx, list) else minIdx
    
    if (PTERROR_REPORT):
        PTError = (tpDiffs[minIdx] / tpComparision[minIdx]) * 100
        PTAngle = rAngles[minIdx]
        PTAxis = rAxes[minIdx]
    
    if (DEBUG_FLAG):
        print("Chosen rotation point: {}".format(minIdx))
        
    return rMatrices[minIdx], ((minIdx + 2) % 4)
        
def perspectiveTransformHandler(intrinsics, depth_frame, perspectivePoints, np_depth_frame_prev):
    global pc, rotationMatrix, fulcrumPixel_idx, isPaused
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
    
    if rotationMatrix is None:
        rotationMatrix, fulcrumPixel_idx = calculateRotationMatrix(points)
    
    cropPointsX = list(map(lambda p: p[0], perspectivePoints))
    cropPointsY = list(map(lambda p: p[1], perspectivePoints))
    minCropX = min(cropPointsX)
    maxCropX = max(cropPointsX) + 1
    minCropY = min(cropPointsY)
    maxCropY = max(cropPointsY) + 1
    
    if (DEBUG_FLAG):
        print(perspectivePoints)
        
    pPoints = []
    for point in perspectivePoints:
        pX = point[0] - minCropX
        pY = point[1] - minCropY
        pPoints.append((pX, pY))
        
    if (DEBUG_FLAG):
        print(pPoints)
    
    np_depth_frame = np_depth_frame[minCropY:maxCropY, minCropX:maxCropX]
    fulcrumPoint = rs.rs2_deproject_pixel_to_point(intrinsics, pPoints[fulcrumPixel_idx], np_depth_frame[pPoints[fulcrumPixel_idx][1], pPoints[fulcrumPixel_idx][0]])
    fulcrumPointRotated = rotationMatrix.dot(np.asanyarray(fulcrumPoint).T).T
    fulcrumPixelDepth = fulcrumPointRotated[2] * scaling_factor
    
    verts = []
    for ix, iy in np.ndindex(np_depth_frame.shape):
        depth = np_depth_frame[ix, iy]
        point = rs.rs2_deproject_pixel_to_point(intrinsics, [iy, ix], depth)
        verts.append(point)
    
    np_verts = np.asanyarray(verts)
    # pcPoints = pc.calculate(depth_frame)
    # np_verts = np.asanyarray(pcPoints.get_vertices(dims=2))
    
    np_verts_transformed = rotationMatrix.dot(np_verts.T).T
    np_verts_transformed = np_verts_transformed[~np.all(np_verts_transformed == 0, axis=1)]
    np_verts_transformed = np_verts_transformed
    
    # project back to 2D image with depth as data (WORKING BUT SLOW)   
    np_transformed_depth_frame = np.zeros([1080,1920])
    for vert in np_verts_transformed:
        pixel = rs.rs2_project_point_to_pixel(intrinsics, vert)
        if (pixel[0] < 960 and pixel[1] < 540 and pixel[0] >= -960 and pixel[1] >= -540):
            np_transformed_depth_frame[int(pixel[1] + 540),int(pixel[0]) + 960] = vert[2]
            
    # Remove rows and columns of all zeros
    np_transformed_depth_frame = np_transformed_depth_frame[~np.all(np_transformed_depth_frame == 0, axis=1)]
    np_transformed_depth_frame = np_transformed_depth_frame[:, ~np.all(np_transformed_depth_frame == 0, axis=0)]
        
    # OpenCV dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    np_dilated_depth_frame = cv2.dilate(np_transformed_depth_frame, kernel)
    np_final_frame = np_dilated_depth_frame
    np_eroded_depth_frame = cv2.erode(np_dilated_depth_frame, kernel)
    np_final_frame = np_eroded_depth_frame
    
    # Black out pixels that have not changed since the last frame
    # NOT WORKING YET
    
    # if not isPaused and np_depth_frame_prev is not None and np_final_frame.shape == np_depth_frame_prev.shape:
    #     staticPixels = np.invert((np_final_frame / 2).astype(int) == (np_depth_frame_prev / 2).astype(int)) * 1.0
    #     print(staticPixels)
    #     print(staticPixels.shape)
    #     np_final_frame = np_final_frame * staticPixels
    
    
    # fulcrumPoint = rotationMatrix.dot(np.asanyarray(points[fulcrumPixel_idx]).T).T
    # fulcrumPixelDepth = fulcrumPoint[2] * scaling_factor
    contours, contours_filteredArea, contours_filteredCircularity, headSphere, allHeadSpheres, maxHeadSlice, torsoSphere = None, None, None, None, None, None, None
    if np.any(np_final_frame):
        contours, contours_filteredArea, contours_filteredCircularity, headSphere, allHeadSpheres, maxHeadSlice, torsoSphere = crossSections(np_final_frame, fulcrumPixelDepth)
    
    return np_final_frame, contours, contours_filteredArea, contours_filteredCircularity, headSphere, maxHeadSlice, torsoSphere
    # return np_final_frame

def crossSections(np_depth_frame, fulcrumPixelDepth):
    global scaling_factor
    
    np_depth_frame = np_depth_frame * scaling_factor
    minDepth = np.min(np_depth_frame[np_depth_frame != 0])
    bedDepth = fulcrumPixelDepth
    sliceInc = (bedDepth - minDepth) / 20
    
    if (DEBUG_FLAG):
        print("minDepth: {}".format(minDepth))
        print("bedDepth: {}".format(bedDepth))
    
    np_depth_frame[np_depth_frame == 0] = bedDepth + 1
    
    sliceDepth = minDepth
    cross_section_frames = []
    for i in range(19):
        np_depth_frame_mask = (np_depth_frame <= sliceDepth) * 1.0
        cross_section_frames.append(np_depth_frame_mask)        
        sliceDepth = sliceDepth + sliceInc
        
    # Find contours for each slice and filter to different criteria
    allContours = []
    allContours_area = []
    allContours_circularity = []
    for np_cs_frame in cross_section_frames:
        np_cs_frame = np_cs_frame.astype(np.uint8)
        contours, hierarchy = cv2.findContours(np_cs_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        contours_filteredArea = []
        for con in contours:
            area = cv2.contourArea(con)
            if 100 < area:
                contours_filteredArea.append(con)
            
        contours_filteredCircularity = []
        for con in contours_filteredArea:
            perimeter = cv2.arcLength(con, True)
            area = cv2.contourArea(con)
            if perimeter == 0:
                break
            circularity = 4*math.pi*(area/(perimeter*perimeter))
            if 0.50 < circularity < 1.50:
                contours_filteredCircularity.append(con)
                
        allContours.append(contours)
        allContours_area.append(contours_filteredArea)
        allContours_circularity.append(contours_filteredCircularity)
                
        if (DEBUG_FLAG):
            print("Contours: {}".format(len(contours)))
            print("Contours (after area filter): {}".format(len(contours_filteredArea)))
            print("Contours (after circle filter): {}".format(len(contours_filteredCircularity)))
        
    # Find head sphere contours
    headSpheres = []
    checkedContours = []
    checkedIds = []
    maxSlice_headSpheres = []
    
    def buildSphere(child, i, sphereList, contourPool, maxSlice=None):
        sphereList.append(child)
        checkedContours.append(child)
        M = cv2.moments(child)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        
        for parent in contourPool[i]:
            if not (id(parent) in checkedIds):
                checkedContours.append(checkedContours)
                ids = map(id, checkedContours)
            if (cv2.pointPolygonTest(parent, (cX, cY), True) >= 0):
                if i+1 < (len(contourPool) if maxSlice is None else maxSlice):
                    # print("maxSlice is: {}".format(maxSlice))
                    # print("i is: {}".format(i))
                    sphereList, _ = buildSphere(parent, i+1, sphereList, contourPool, maxSlice)
                break
        
        if len(sphereList) > 1:
            # print("At the end of buildSphere, i is: {}".format(i))
            return sphereList, i
        else:
            return None, None
    
    for i in range(len(allContours_circularity)-1):
        for child in allContours_circularity[i]:
            if not (id(child) in checkedIds):
                sphere, maxHeadSlice = buildSphere(child, i+1, [], allContours_circularity)
                if sphere is not None:
                    headSpheres.append(sphere)
                    maxSlice_headSpheres.append(maxHeadSlice)
                
    if (DEBUG_FLAG):
        print("Number of headSpheres: {}".format(len(headSpheres)))
    
    headSphere = None
    maxHeadSlice = None
    if len(headSpheres) > 0:
        headSphereCircularityErrs = []
        for sphere in headSpheres:
            perimeter = cv2.arcLength(sphere[-1], True)
            area = cv2.contourArea(sphere[-1])
            circularity = 4*math.pi*(area/(perimeter*perimeter))
            headSphereCircularityErrs.append(abs(1-circularity))
        
        chosenHeadSphere_idx = headSphereCircularityErrs.index(min(headSphereCircularityErrs))
        headSphere = headSpheres[chosenHeadSphere_idx]
        maxHeadSlice = maxSlice_headSpheres[chosenHeadSphere_idx]
    
    # Find torso cuboid contours
    torsoSpheres = []
    checkedContours = []
    checkedIds = []
    
    if maxHeadSlice is not None:
        for i in range(maxHeadSlice):
            for child in allContours_area[i]:
                if not (id(child) in checkedIds):
                    sphere, _ = buildSphere(child, i+1, [], allContours_area, maxHeadSlice)
                    if sphere is not None:
                        torsoSpheres.append(sphere)
                    
    torsoSphere = None
    if len(torsoSpheres) > 0:
        torsoSphereAreas = []
        for sphere in torsoSpheres:
            area = cv2.contourArea(sphere[-1])
            torsoSphereAreas.append(area)
            
        chosenTorsoSphere_idx = torsoSphereAreas.index(max(torsoSphereAreas))
        torsoSphere = torsoSpheres[chosenTorsoSphere_idx]
    
    return allContours, allContours_area, allContours_circularity, headSphere, headSpheres, maxHeadSlice, torsoSphere

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
cv2.createButton("Toggle Depth Selector", buttonHandler, DSENABLE, cv2.QT_PUSH_BUTTON|cv2.QT_NEW_BUTTONBAR)
cv2.createButton("Perspective Transformation", buttonHandler, PTENABLE, cv2.QT_PUSH_BUTTON|cv2.QT_NEW_BUTTONBAR)

frames = pipeline.wait_for_frames()

np_depth_frame_prev = None

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
        if(DEBUG_FLAG):
            start_time = time.time()
        
        np_depth_frame, contours, contours_filteredArea, contours_filteredCircularity, headSphere, maxHeadSlice, torsoSphere = perspectiveTransformHandler(intrinsics, depth_frame, perspectivePoints, np_depth_frame_prev)
        # np_depth_frame = perspectiveTransformHandler(intrinsics, depth_frame, perspectivePoints)
        
        np_depth_frame_prev = np_depth_frame.copy()
        
        if(DEBUG_FLAG):
            print("--- {}s seconds ---".format((time.time() - start_time)))
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
    
    if len(perspectivePoints) == 4:
        
        for cons in contours_filteredArea:
            finalImage = cv2.drawContours(finalImage, cons, -1, (255,0,255), 2)
        
        # if maxHeadSlice is not None:
        #     for i in range(maxHeadSlice):
        #         finalImage = cv2.drawContours(finalImage, contours_filteredArea[i], -1, (0,0,255), 2)
            
        # for cons in contours_filteredCircularity:
        #     finalImage = cv2.drawContours(finalImage, cons, -1, (0,0,255), 2)
            
        # Display final headsphere contours
        if headSphere is not None:
            finalImage = cv2.drawContours(finalImage, headSphere, -1, (255, 0, 0), 2)
            
        if torsoSphere is not None:
            finalImage = cv2.drawContours(finalImage, torsoSphere, -1, (0,0,255), 2)
            # finalImage = cv2.drawContours(finalImage, [torsoSphere[-1]], -1, (0,0,255), -1)
            
            roi = np.ones(np_depth_frame.shape)
            roi = cv2.drawContours(roi, [torsoSphere[-1]], -1, 0, -1)
            
            np_ma_torsoROI = np.ma.masked_array(np_depth_frame, mask=roi)
            if (DEBUG_FLAG):
                print("torsoROI Mean: {}, Time: {}".format(np_ma_torsoROI.mean(), aligned_frames.get_timestamp()))
            if not isPaused:
                avgTorsoDepth.append([np_ma_torsoROI.mean(), aligned_frames.get_timestamp()])
            
        # for cons in contours_filteredRectangularity:
        #     finalImage = cv2.drawContours(finalImage, cons, -1, (255,0,0), 1)
    
    output_image = finalImage
    
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
    

