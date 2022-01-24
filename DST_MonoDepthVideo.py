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
from tkinter import Frame, Tk
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
switchName = "0: Play\n1: Pause"

# Global vars
slider1Arg = 0
slider2Arg = 0
slice1At = 0
slice2At = 0
scaling_factor = 0
k = []
distortionModel = ""
d = []
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

def play(videoFile, intrinsics):
    rotationMatrix, fulcrumPixel_idx = None, None
    # cap = cv2.VideoCapture(filename, cv2.CAP_FFMPEG)
    cap = cv2.VideoCapture(videoFile)
    # print(cap.getBackendName())
    tmp = int(cap.get(cv2.CAP_PROP_FOURCC))
    # print(chr(tmp&0xff) + chr((tmp>>8)&0xff) + chr((tmp>>16)&0xff) + chr((tmp>>24)&0xff))

    def readFrame():
        ret, img = cap.read()
        np_blueChannel = img[:, :, 0].astype('uint16')
        np_greenChannel = img[:, :, 1].astype('uint16')
        np_highBits = np_greenChannel << 8
        np_depth16 = np_highBits | np_blueChannel
        return ret, np_depth16
    
    ret, np_depth16 = readFrame()

    while ret:
        if not isPaused:
            ret, np_depth16 = readFrame()

        if ret:
            np_depth_frame_orig = np_depth16.copy()

            if len(perspectivePoints) == 4:
                tic = time.perf_counter()
                np_depth_frame, contours, contours_filteredArea, contours_filteredCircularity, headSphere, maxHeadSlice, torsoSphere, rotationMatrix, fulcrumPixel_idx, errs = libdst.perspectiveTransformHandler(intrinsics, np_depth_frame_orig.copy(), perspectivePoints, scaling_factor, None, rotationMatrix, fulcrumPixel_idx, isPaused, np_depth_frame_prev, np_depth_frame_prev_prev, PTError, PTAngle, PTAxis, DEBUG_FLAG, rs2_functions=False)
                toc = time.perf_counter()
                print(f"PT in {toc - tic:0.4f} seconds")
                # Without Numba or Cupy (iteration-based method), around 2-2.3 secs
                # Without Numba or Cupy (matrix operations), around 37 secs
                # With Numba (iteration-based method), around 7-8 secs
                # With Numba (matrix operations), around 9-11 secs
                # With cupy (matrix operations), around 40 secs

            else:
                np_depth_frame = np_depth_frame_orig

            slice1At, slice2At = updateSlicers(0)
            np_depth_frame_scaled = np_depth_frame * scaling_factor
            sliceEnd = slice1At + slice2At
            np_depth_frame_bool1 = (np_depth_frame_scaled < sliceEnd) * 1
            np_depth_frame_bool2 = (np_depth_frame_scaled > slice2At) * 1
            np_depth_frame_bool = np.bitwise_and(np_depth_frame_bool1, np_depth_frame_bool2)
            
            np_depth_frame_sliced = np_depth_frame_scaled * np_depth_frame_bool

            output_image = libdst.displayDepthPoints(np_depth_frame_scaled, np_depth_frame_sliced, depthPoints, DEBUG_FLAG)
            
            # output_image = np_depth_frame_sliced
            # print(np_depth_frame_sliced.max())
            cv2.imshow(windowName, output_image)

            # Exit conditions
            key = cv2.waitKey(1)
            if (key == 27) or (cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) != 1):
                cv2.destroyAllWindows()
                break

def main():
    global scaling_factor, k, distortionModel, d

    root = Tk()
    root.withdraw()
    root.overrideredirect(True)
    root.geometry('0x0+0+0')
    root.deiconify()
    root.lift()
    root.focus_force()
    videoFile = askopenfilename(filetypes=[("Depth video encoded as RGB files", ".mj2")], parent=root)
    root.destroy()
    if not videoFile:
        sys.exit("No video file selected")

    root = Tk()
    root.withdraw()
    root.overrideredirect(True)
    root.geometry('0x0+0+0')
    root.deiconify()
    root.lift()
    root.focus_force()
    intrinsicsFile = askopenfilename(filetypes=[("Camera intrinsics information in text file", ".txt")], parent=root)
    root.destroy()
    if not intrinsicsFile:
        sys.exit("No intrinsics file selected")

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
