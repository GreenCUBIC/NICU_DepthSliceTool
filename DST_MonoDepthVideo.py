# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 14:06:39 2022

@author: Zein Hajj-Ali - zeinhajjali@sce.carleton.ca
"""

class intrinsics_noRS2:
    def __init__(self, model, pp, f, coeffs):
        self.model = model
        self.ppx = pp[0]
        self.ppy = pp[1]
        self.fx = f[0]
        self.fy = f[1]
        self.coeffs = coeffs

import numpy as np
import cv2
import datetime
import time
import sys
import math
import os
import csv
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import libdst

# Flags
DSENABLE = "DEPTH_SELECT_ENABLE"
PTENABLE = "PERSPECTIVE_TRANSFORM_ENABLE"
RGBENABLE = "RGB_OVERLAY_ENABLE"
DEBUG_FLAG = False

# OpenCV component name strings
windowName = "DepthSlice Tool"
slider1Name = "Slice depth (increments of 0.001)"
slider2Name = "Slice start (increments of 0.001)"
sliderSeek = "Seek (seconds):"
switchName = "0: Play\n1: Pause"

# Global vars
saveFolder = None
slider1Arg = 0
slider2Arg = 0
slice1At = 0
slice2At = 0
seekTime = None
seekFrame = 0
scaling_factor = 0
k = []
distortionModel = ""
d = []
init_timestamp = None
init_epochTime = None
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
def updateSlicers(arg):
    # Get value of slider
    value1 = cv2.getTrackbarPos(slider1Name, windowName)
    value2 = cv2.getTrackbarPos(slider2Name, windowName)
    
    sliceDepth1 = value1/1000
    sliceDepth2 = value2/1000
    
    return sliceDepth1, sliceDepth2

def playPause(arg):
    playPauseFlag = cv2.getTrackbarPos(switchName, windowName)
    global isPaused
    
    if playPauseFlag == 1:
        isPaused = True
        
    else:
        isPaused = False

# def onSeek(arg, framerate, set=True, videoCap=None):
#     if not set:
#         value = cv2.getTrackbarPos(sliderSeek, windowName)
#         seekIndex = value * framerate
#         return seekIndex

#     else:
#         seekIndex = arg * framerate
#         videoCap.set(cv2.CAP_PROP_POS_FRAMES, seekIndex)

def play(videoFile, intrinsics):
    rotationMatrix, fulcrumPixel_idx = None, None
    cap = cv2.VideoCapture(videoFile)
    # totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    framerate= int(cap.get(cv2.CAP_PROP_FPS))
    # totalSecs = totalFrames / framerate
    # cv2.createTrackbar(sliderSeek, windowName, 0, int(totalSecs), lambda seekFunc: onSeek(seekFunc, framerate, videoCap=cap))
    # print("Frames: {}; Framerate: {}; Seconds: {}".format(totalFrames, framerate, totalSecs))
    frameCounter = 0

    # TESTING
    # ONLY USE WHEN GETTING DATA ON SPECIFIC FRAMES
    # STARTING_FRAME = 13500
    # cap.set(cv2.CAP_PROP_POS_FRAMES, STARTING_FRAME)
    # frameCounter = STARTING_FRAME



    def readFrame():
        ret, img = cap.read()
        np_blueChannel = img[:, :, 0].astype('uint16')
        np_greenChannel = img[:, :, 1].astype('uint16')
        np_highBits = np_greenChannel << 8
        np_depth16 = np_highBits | np_blueChannel
        return ret, np_depth16
    
    ret, np_depth16 = readFrame()
    prevTime = 0
    while ret:
        
        timeElapsed = time.time() - prevTime
        if not isPaused and timeElapsed >= 1.0/framerate:
            ret, np_depth16 = readFrame()
            frameCounter += 1

        if ret:
            np_depth_frame_orig = np_depth16.copy()

            if len(perspectivePoints) == 4:
                tic = time.perf_counter()
                np_depth_frame, contours, contours_filteredArea, contours_filteredCircularity, headSphere, maxHeadSlice, torsoSphere, rotationMatrix, fulcrumPixel_idx, errs = libdst.PTwithCrossSection(intrinsics, np_depth_frame_orig.copy(), perspectivePoints, scaling_factor, None, rotationMatrix, fulcrumPixel_idx, isPaused, np_depth_frame_prev, np_depth_frame_prev_prev, PTError, PTAngle, PTAxis, DEBUG_FLAG, rs2_functions=False)
                toc = time.perf_counter()
                # print(f"PT in {toc - tic:0.4f} seconds")
                # Without Numba or Cupy (iteration-based method), around 2-2.3 secs
                # Without Numba or Cupy (matrix operations), around 37 secs
                # With Numba (iteration-based method), around 7-8 secs
                # With Numba (matrix operations), around 9-11 secs
                # With cupy (matrix operations), around 40 secs

            else:
                np_depth_frame = np_depth_frame_orig

            np_depth_color_frame = cv2.applyColorMap(cv2.convertScaleAbs(np_depth_frame, alpha=0.03), cv2.COLORMAP_TURBO)

            slice1At, slice2At = updateSlicers(0)
            np_depth_frame_scaled = np_depth_frame * scaling_factor
            sliceEnd = slice1At + slice2At
            np_depth_frame_bool1 = (np_depth_frame_scaled < sliceEnd) * 1
            np_depth_frame_bool2 = (np_depth_frame_scaled > slice2At) * 1
            np_depth_frame_bool = np.bitwise_and(np_depth_frame_bool1, np_depth_frame_bool2)
            
            # np_depth_frame_sliced = np_depth_frame_scaled * np_depth_frame_bool

            np_depth_color_frame_masked = np_depth_color_frame.copy()
            # Slice the color frame using the boolean mask
            for i in range(0, 3):
                np_depth_color_frame_masked[:, :, i] = np_depth_color_frame_masked[:, :, i] * np_depth_frame_bool

            output_image = libdst.displayDepthPoints(np_depth_frame_scaled, np_depth_color_frame_masked, depthPoints, DEBUG_FLAG)

            if len(perspectivePoints) == 4:
                finalDepthImage = output_image.copy()
        
                # for cons in contours_filteredArea:
                #     finalDepthImage = cv2.drawContours(finalDepthImage, cons, -1, (255,0,255), 2)
                
                # if maxHeadSlice is not None:
                #     for i in range(maxHeadSlice):
                #         finalDepthImage = cv2.drawContours(finalDepthImage, contours_filteredArea[i], -1, (0,0,255), 2)
                    
                # for cons in contours_filteredCircularity:
                #     finalDepthImage = cv2.drawContours(finalDepthImage, cons, -1, (0,255,255), 2)
                    
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
                        print("torsoROI Mean: {}, Time: {}".format(np_ma_torsoROI.mean(), frameCounter/framerate))


                    # # Keep this or the one after, not both
                    # if not isPaused and timestamps[frameCounter] != avgTorsoDepth[-1][0]:
                    #     avgTorsoDepth.append([timestamps[frameCounter], np_ma_torsoROI.mean()])

                    if not isPaused:
                        epoch_frameTime = init_epochTime + (frameCounter/framerate)
                        string_frameTime = datetime.datetime.fromtimestamp(epoch_frameTime).strftime('%Y-%m-%d %H:%M:%S.%f')
                        avgTorsoDepth.append([string_frameTime, np_ma_torsoROI.mean()])
                
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
    
            # output_image = np_depth_frame_sliced
            # print(np_depth_frame_sliced.max())
            cv2.imshow(windowName, output_image)

            # Exit conditions
            key = cv2.waitKey(1)
            if (key == 27) or (cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) != 1) or (frameCounter >= 55000):
                if len(avgTorsoDepth) > 1:
                    avgTorsoDepth_filename = saveFolder + "//" + os.path.splitext(os.path.basename(videoFile))[0] + "_PT_ROI_AvgDepth.csv"
                    with open(avgTorsoDepth_filename, 'w') as f:
                        csvWriter = csv.writer(f)
                        csvWriter.writerow(["Timestamp", "Mean Depth"])
                        csvWriter.writerows(avgTorsoDepth)
                        
                # if PTError is not None:
                #     PTError_filename = os.path.splitext(videoFile)[0] + "_PerspectiveTransformError.csv"
                #     with open(PTError_filename, 'w') as f:
                #         csvWriter = csv.writer(f)
                #         csvWriter.writerow(["Angle (rad)", "Axis", "Absolute Error (%)"])
                #         csvWriter.writerow([PTAngle, PTAxis, PTError])
                cv2.destroyAllWindows()
                break

def main():
    global scaling_factor, k, distortionModel, d, init_timestamp, init_epochTime, saveFolder

    root = Tk()
    root.withdraw()
    root.overrideredirect(True)
    root.geometry('0x0+0+0')
    root.deiconify()
    root.lift()
    root.focus_force()
    # If you want to pick from the NAS, add the positonal argument (initialdir=r"\\134.117.64.31\\Main Storage")
    videoFile = askopenfilename(filetypes=[("Depth video encoded as RGB files", ".mj2")], parent=root, initialdir=r"\\134.117.64.31\\Main Storage")
    intrinsicsFile = askopenfilename(filetypes=[("Camera intrinsics information in text file", ".txt")], parent=root, initialdir=r"\\134.117.64.31\\Main Storage")
    timestampFile = askopenfilename(filetypes=[("Video starting timestamp in text file", ".txt")], parent=root, initialdir=r"\\134.117.64.31\\Main Storage")
    saveFolder = "C:\\Users\\zeinhajjali-admin\\Documents\\depthSliceTool\\bagmerge\\AvgDepths"
    root.destroy()
    if not videoFile:
        sys.exit("No video file selected")
    elif not intrinsicsFile:
        sys.exit("No intrinsics file selected")
    
    if timestampFile:
        timestampFile = open(timestampFile, 'r')
        print('Reading timestamp')
        timeString = timestampFile.readline().split('=')[1].rstrip()
        init_timestamp = datetime.datetime.strptime(timeString, '%Y-%m-%d %H:%M:%S.%f')
        init_epochTime = datetime.datetime.timestamp(init_timestamp)

    # Store intrinsics in vars
    intrinsicsFile = open(intrinsicsFile, 'r')
    print('Reading intrinsics...')
    scaling_factor = float(intrinsicsFile.readline().split('=')[1])
    k = [float(x) for x in intrinsicsFile.readline().split('=')[1].split(' ')]
    distortionModel = intrinsicsFile.readline().split('=')[1]
    d = [float(x) for x in intrinsicsFile.readline().split('=')[1].split(' ')]

    # Store intrinsics in the form: [Distortion Model, (ppx, ppy), (fx, fy), coeffs]
    intrinsics = intrinsics_noRS2(distortionModel, (k[2], k[5]), (k[0], k[4]), d)

    # Create opencv window with trackbars, tool buttons, and set the mouse action handler
    cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar(slider1Name, windowName, 1500 if DEBUG_FLAG else 15, 1000, updateSlicers)
    cv2.createTrackbar(slider2Name, windowName, 0, 1500, updateSlicers)
    cv2.createTrackbar(switchName, windowName, 1, 1, playPause)
    cv2.setMouseCallback(windowName, mouseEvent)
    # cv2.createButton("RGB Overlay (Only on original video)", buttonHandler, RGBENABLE, cv2.QT_PUSH_BUTTON|cv2.QT_NEW_BUTTONBAR)
    cv2.createButton("Toggle Depth Selector", buttonHandler, DSENABLE, cv2.QT_PUSH_BUTTON|cv2.QT_NEW_BUTTONBAR)
    cv2.createButton("Perspective Transformation", buttonHandler, PTENABLE, cv2.QT_PUSH_BUTTON|cv2.QT_NEW_BUTTONBAR)
    play(videoFile, intrinsics)

if __name__=="__main__":
    main()
