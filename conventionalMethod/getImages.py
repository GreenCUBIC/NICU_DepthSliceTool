import os
import time
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as im

ROOT_FOLDER_PATH = "C:/Users/Zalamaan/Documents/Repos/NICU_Data/DepthFrameFullPrec_prePT/"

allPts = [1, 2, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 17, *range(21, 35), *range(90, 95)]
bedPoint = (0, 0)
numSlices = 15


def testImshow(img):
    while True:
        cv2.imshow("test", img)

        # If user presses ESCAPE or clicks the close button, end    
        key = cv2.waitKey(1)
        if (key == 27) or (cv2.getWindowProperty("test", cv2.WND_PROP_VISIBLE) != 1):
            cv2.destroyAllWindows()
            break

def selectBedPoint(action, x, y, flags, *userdata):
    global bedPoint
    if action == cv2.EVENT_LBUTTONDBLCLK:
            bedPoint = (x, y)

def setBedPoint(img):
    global bedPoint
    img_8bit = cv2.convertScaleAbs(img, alpha=(255.0/65535.0))
    depth_frame = img_8bit
    depth_frame = depth_frame.astype('float64')
    depth_frame *= 255.0/depth_frame.max()
    depth_frame = depth_frame.astype('uint8')
    depth_frame = cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET)

    windowName = "selectBedPoint"
    cv2.namedWindow(windowName)
    cv2.setMouseCallback(windowName, selectBedPoint)

    while True:
        output_frame = depth_frame.copy()
        if bedPoint != (0, 0):
            output_frame = cv2.circle(output_frame, bedPoint, 3, (0, 0, 0), 2)
        cv2.imshow(windowName, output_frame)

        # If user presses ESCAPE or clicks the close button, end    
        key = cv2.waitKey(1)
        if (key == 27) or (cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) != 1):
            cv2.destroyAllWindows()
            break
    
    bedPointDepth = img[bedPoint[1], bedPoint[0]]
    print(bedPointDepth)
    return bedPointDepth

def getSlices(img, numSlices, lowestPoint=None):
    depth_frame = img.copy()
    # depth_frame = depth_frame.astype('float64')
    # depth_frame *= 65535.0/depth_frame.max()
    # depth_frame = depth_frame.astype('uint16')
    sliceDepth = depth_frame.min()
    sliceInc = (lowestPoint - sliceDepth) / numSlices
    allSlices = []
    for i in range(numSlices):
        depth_frame_mask = ((depth_frame <= sliceDepth) == (depth_frame != 0)) * 1.0
        # if sliceDepth <= lowestPoint:
        allSlices.append(depth_frame_mask)
        sliceDepth = sliceDepth + sliceInc

    return allSlices

def getContours(slice):
    slice_frame = slice.astype(np.uint8)
    contours, hierarchy = cv2.findContours(slice_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_filteredArea = []
    for con in contours:
        area = cv2.contourArea(con)
        if area >= 10000:
            contours_filteredArea.append(con)

    return contours_filteredArea

def con_is_candidate(contour, img):
    img_height = img.shape[0]
    img_width = img.shape[1]

    contour = contour.reshape(contour.shape[0], contour.shape[2])
    if (contour[:, 0] == (img_width - 1)).any() or (contour[:, 0] == 0).any() or (contour[:, 1] == (img_height - 1)).any() or (contour[:, 1] == 0).any():
        # Get max distance from center of image for probability
        imgX = int(img_width/2)
        imgY = int(img_height/2)
        imgCenter = np.array((imgX, imgY))
        maxDist = np.linalg.norm(imgCenter)

        # Get contour center and calc distance from image center
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        contourCenter = np.asarray((cX, cY))
        dist = np.linalg.norm(contourCenter - imgCenter)

        prob = (maxDist - dist) / maxDist
        return prob
    else:
        return 0

def getImgIDProb(img, numSlices, bedPointDepth):
    slices = getSlices(img, numSlices, bedPointDepth)
    chosenSlice = slices[-1]
    cons = getContours(chosenSlice)
    # print(len(cons))

    maxProb = 0
    for con in cons:
        # showCons = cv2.convertScaleAbs(img, alpha=(255.0/65535.0))
        # showCons = cv2.applyColorMap(showCons, cv2.COLORMAP_JET)
        # showCons = cv2.drawContours(showCons, [con], -1, color=(255, 0, 255), thickness=-1)
        # testImshow(showCons)
        currProb = con_is_candidate(con, img)
        # print(currProb)
        if currProb > maxProb:
            maxProb = currProb

    return maxProb

def findUniqueFileNames(rootFolderPath, patientId):
    nurseFiles_path = rootFolderPath + 'p' + str(patientId) + '/nurse/*_0.png'
    nurseFiles = glob(nurseFiles_path)
    nooneFiles_path = rootFolderPath + 'p' + str(patientId) + '/noone/*_0.png'
    nooneFiles = glob(nooneFiles_path)

    uniqueFiles = list(set(nooneFiles).symmetric_difference(set(nurseFiles)))

    uniqueFileNames = []
    for file in uniqueFiles:
        uniqueFileNames.append(os.path.basename(file).split('.')[0][:-2])

    return uniqueFileNames

# load image
# testImg = "C:/Users/Zalamaan/Documents/Repos/NICU_Data/DepthFrameFullPrec_prePT/p90/nurse/p90_55.png"
testImg = "Patient_14_part2_part_2_D.jpg"
testImg = "C:/Users/Zalamaan/Documents/Repos/depthSliceTool/conventionalMethod/Pt_28_1_part_602_D.jpg"

depth_frame = cv2.imread(testImg, -1)
testImshow(depth_frame)

# normalize and get 8-bit
# img_8bit = cv2.convertScaleAbs(depth_frame, alpha=(255.0/65535.0))
# depth_frame = img_8bit
depth_frame = depth_frame.astype('float64')
depth_frame *= 255.0/depth_frame.max()
depth_frame = depth_frame.astype('uint8')
# depth_frame0 = cv2.applyColorMap(depth_frame[:, :, 0], cv2.COLORMAP_JET)
# testImshow(depth_frame[:, :, 0])
# testImshow(depth_frame[:, :, 1])
# testImshow(depth_frame[:, :, 2])


# save it
# cv2.imwrite("P8_HHA_0.png", depth_frame[:, :, 0])
# cv2.imwrite("P8_HHA_1.png", depth_frame[:, :, 1])
# cv2.imwrite("P8_HHA_2.png", depth_frame[:, :, 2])

depth_frame = cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET)
testImshow(depth_frame)

cv2.imwrite("Pt_28_1_part_602_D_colormapped.png", depth_frame)

