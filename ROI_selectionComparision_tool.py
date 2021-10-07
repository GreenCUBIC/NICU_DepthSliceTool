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
import copy
import sys
import math
import os
import csv
import random
from tkinter import Tk
from tkinter.filedialog import askopenfilename

PTENABLE = "PERSPECTIVE_TRANSFORM_ENABLE"
STAGEONE_FLAG = "STAGE_ONE"
STAGETWO_FLAG = "STAGE_TWO"
STAGETHREE_FLAG = "STAGE_THREE"
DEBUG_FLAG = False
PTERROR_REPORT = True

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
        
def perspectiveTransformHandler(intrinsics, np_depth_frame, perspectivePoints):
    global pc, rotationMatrix, fulcrumPixel_idx, isPaused, np_depth_frame_prev, np_depth_frame_prev_prev
    points = []
    
    for pixel in perspectivePoints:
        depth = np_depth_frame[pixel[1],pixel[0]]
        point = rs.rs2_deproject_pixel_to_point(intrinsics, pixel, depth)
        points.append(point)
    
    if rotationMatrix is None:
        rotationMatrix, fulcrumPixel_idx = calculateRotationMatrix(points)
        
    if (DEBUG_FLAG):
        print(perspectivePoints)
        
    pPoints = []
    for point in perspectivePoints:
        pX = point[0]
        pY = point[1]
        
        pPoints.append((pX, pY))
        
    if (DEBUG_FLAG):
        print(pPoints)
    
    fulcrumPoint = rs.rs2_deproject_pixel_to_point(intrinsics, pPoints[fulcrumPixel_idx], np_depth_frame[pPoints[fulcrumPixel_idx][1], pPoints[fulcrumPixel_idx][0]])
    fulcrumPointRotated = rotationMatrix.dot(np.asanyarray(fulcrumPoint).T).T
    fulcrumPixelDepth = fulcrumPointRotated[2] * scaling_factor
    
    verts = []
    for iy, ix in np.ndindex(np_depth_frame.shape):
        depth = np_depth_frame[iy, ix]
        point = rs.rs2_deproject_pixel_to_point(intrinsics, [ix, iy], depth)
        verts.append(point)
    
    np_verts = np.asanyarray(verts)    
    np_verts_transformed = rotationMatrix.dot(np_verts.T).T
    np_verts_transformed = np_verts_transformed[~np.all(np_verts_transformed == 0, axis=1)]
    np_verts_transformed = np_verts_transformed
    
    # project back to 2D image with depth as data (WORKING BUT SLOW)   
    np_transformed_depth_frame = np.zeros([1080,1920])
    for vert in np_verts_transformed:
        pixel = rs.rs2_project_point_to_pixel(intrinsics, vert)
        if (pixel[0] < 960 and pixel[1] < 540 and pixel[0] >= -960 and pixel[1] >= -540):
            np_transformed_depth_frame[int(pixel[1] + 0),int(pixel[0]) + 0] = vert[2]
            
    # Remove rows and columns of all zeros
    np_final_frame = np_transformed_depth_frame
    
    
        
    # OpenCV dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    np_dilated_depth_frame = cv2.dilate(np_final_frame, kernel)
    np_final_frame = np_dilated_depth_frame
    np_eroded_depth_frame = cv2.erode(np_dilated_depth_frame, kernel)
    np_final_frame = np_eroded_depth_frame
    
    
    # fulcrumPoint = rotationMatrix.dot(np.asanyarray(points[fulcrumPixel_idx]).T).T
    # fulcrumPixelDepth = fulcrumPoint[2] * scaling_factor
    contours, contours_filteredArea, contours_filteredCircularity, headSphere, allHeadSpheres, maxHeadSlice, torsoSphere = None, None, None, None, None, None, None
    if np.any(np_final_frame):
        contours, contours_filteredArea, contours_filteredCircularity, headSphere, allHeadSpheres, maxHeadSlice, torsoSphere = crossSections(np_final_frame, fulcrumPixelDepth)
    
    return np_final_frame, contours, contours_filteredArea, contours_filteredCircularity, headSphere, maxHeadSlice, torsoSphere, rotationMatrix
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
                    sphereList, _ = buildSphere(parent, i+1, sphereList, contourPool, maxSlice)
                break
        
        if len(sphereList) > 1:
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
frameCounter = random.randrange(0, len(depth_frames))

while True:
    
    # Stage 1: Manual head ROI selection
    if currStage == STAGEONE_FLAG:
        np_color_frame = color_frames[frameCounter]
        np_color_frame = np_color_frame[...,::-1]
        stageOne_headROI = np.ones(np_color_frame.shape)
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
            stageOne_headROI = cv2.drawContours(stageOne_headROI, np.array([stageOne_headPts]), 0, 0, -1)
            np_color_frame_masked_headROI = np.ma.masked_array(np_color_frame, stageOne_headROI)
           
    # Stage 2: Manual torso ROI selection
    elif currStage == STAGETWO_FLAG:
        stageTwo_torsoROI = np.ones(np_color_frame.shape)
        
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
            alpha = 0.4
            output_image = cv2.addWeighted(overlay, alpha, output_image, 1-alpha, 0)
            
            # Save area of completed polygon in a numpy masked array for comparison with automatic method
            stageTwo_torsoROI = cv2.drawContours(stageTwo_torsoROI, np.array([stageTwo_torsoPts]), 0, 0, -1)
            np_color_frame_masked_torsoROI = np.ma.masked_array(np_color_frame, stageTwo_torsoROI)
        
    # Stage 3: Automatic head and torso ROI selection (as in depthSliceTool)
    else:
        np_depth_frame = depth_frames[frameCounter]
        np_depth_frame_orig = np_depth_frame.copy()
        if len(perspectivePoints) < 4:
            perspectiveSelectEnabled = True
        if len(perspectivePoints) == 4:
            if(DEBUG_FLAG):
                start_time = time.time()
            
            np_depth_frame, contours, contours_filteredArea, contours_filteredCircularity, headSphere, maxHeadSlice, torsoSphere, rotationMatrix = perspectiveTransformHandler(intrinsics, np_depth_frame, perspectivePoints)
            inv_rotMat = np.linalg.inv(rotationMatrix)
                        
            if(DEBUG_FLAG):
                print("--- {}s seconds ---".format((time.time() - start_time)))
            
        np_depth_frame_scaled = np_depth_frame * scaling_factor
        np_depth_frame_orig_scaled = np_depth_frame_orig * scaling_factor
                
        np_depth_color_frame = cv2.applyColorMap(cv2.convertScaleAbs(np_depth_frame, alpha=0.03), cv2.COLORMAP_TURBO)
        np_depth_color_frame_orig = cv2.applyColorMap(cv2.convertScaleAbs(np_depth_frame_orig, alpha=0.03), cv2.COLORMAP_TURBO)
        
        finalDepthImage_PT = np_depth_color_frame
        finalDepthImage = np_depth_color_frame_orig
        
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
                
    
        output_image = finalDepthImage
        
        #DEBUGGING
        # output_image = finalDepthImage_PT
        # cv2.imshow("test2", finalDepthImage_PT)
        
    # Render image in opencv window
    cv2.imshow(windowName, output_image)
    
    # If user presses ESCAPE or clicks the close button, end    
    key = cv2.waitKey(1)
    if (key == 27) or (cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) != 1):
        # if len(avgTorsoDepth) > 0:
        #     avgTorsoDepth_filename = os.path.splitext(filename)[0] + "_TorsoROIDepth.csv"
        #     with open(avgTorsoDepth_filename, 'w') as f:
        #         csvWriter = csv.writer(f)
        #         csvWriter.writerow(["Mean Depth", "Timestamp"])
        #         csvWriter.writerows(avgTorsoDepth)
                
        # if PTError is not None:
        #     PTError_filename = os.path.splitext(filename)[0] + "_PerspectiveTransformError.csv"
        #     with open(PTError_filename, 'w') as f:
        #         csvWriter = csv.writer(f)
        #         csvWriter.writerow(["Angle (rad)", "Axis", "Absolute Error (%)"])
        #         csvWriter.writerow([PTAngle, PTAxis, PTError])
        cv2.destroyAllWindows()
        break
        