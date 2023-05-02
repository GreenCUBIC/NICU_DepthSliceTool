import os
import sys
import time
from tkinter import Tk
from tkinter.filedialog import askdirectory

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Im
import pyrealsense2 as rs

OUTFOLDER_PATH = 'C:/Users/Zalamaan/Documents/Repos/depthSliceTool/DummyCapture/out/'
RGBFOLDER_PATH = OUTFOLDER_PATH + 'RGB/'
DEPTHFOLDER_PATH = OUTFOLDER_PATH + 'Depth/'
WINDOWNAME = 'RS'

hole_filling = rs.hole_filling_filter(mode=1)

pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
depth_sensor = device.query_sensors()[0]
depth_sensor.set_option(rs.option.visual_preset, 2)

# print(depth_sensor.get_option_range(rs.option.visual_preset))

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

for i in range(120):
    countdownTime = 3
    print(i+1)
    while countdownTime:
        frames = pipeline.wait_for_frames()
        # print(countdownTime)
        time.sleep(1)
        countdownTime -= 1



    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    # depth_frame = hole_filling.process(depth_frame)
    color_frame = frames.get_color_frame()
    # if not depth_frame or not color_frame:
    #     continue

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    images = np.hstack((color_image, depth_colormap))

    im = Im.fromarray(color_image)
    im.save(RGBFOLDER_PATH + str(i) + '.png')
    im = Im.fromarray(depth_image)
    im.save(DEPTHFOLDER_PATH + str(i) + '.png')

    cv2.namedWindow(WINDOWNAME, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(WINDOWNAME, images)

    while True:
        key = cv2.waitKey(1)
        if (key == 27) or (cv2.getWindowProperty(WINDOWNAME, cv2.WND_PROP_VISIBLE) != 1):
            cv2.destroyAllWindows()
            break

pipeline.stop()

    