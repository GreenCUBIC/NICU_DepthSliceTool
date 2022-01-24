import pyrealsense2 as rs
import numpy as np
import cv2
import math
from numba import jit
import cupy as cp

def displayDepthPoints(depth_frame_scaled, output, depthPoints, DEBUG_FLAG = False):
    '''
    Displays selected depth points on the current frame.

    Parameters:
        depth_frame_scaled (NxM numpy array of doubles): Numpy array containing depth values for each pixel.
        output (NxM numpy array of uint8): Numpy array containing color frame to show depth values on.
        depthPoints (List of points (x, y)): List of points of which to show depth values on color frame.
        DEBUG_FLAG (boolean): Flag to show/hide debug prints. Defaults to False (Don't show debugging information).

    Returns:
        output (NxM numpy array of uint8): Numpy array containing color frame with depth values shown.
    '''

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

    return output
  
def npShift(arr, numX, numY, fill_value=0):
    '''
    DO NOT USE.

    '''

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

def calculateRotationMatrix(points, DEBUG_FLAG = False):
    '''
    Calculates rotation matrix to based on given points to make frame face head-on.

    Parameters:
        points (list of 3D points): List of points selected during perspective transformation process.
        DEBUG_FLAG (boolean): Flag to show/hide debug prints. Defaults to False (Don't show debugging information).

    Returns:
        rotationMatrix (3x3 Numpy array): Numpy array containing calculated rotation matrix.
        fulcrumPixel_idx (int): Index of middle point used for calculating rotation matrix.
        [PTError, PTAngle, PTAxis] (List of floats): Array containing information on the calculation of the rotation matrix (error, angle, and axis).
    '''
    
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
    
    PTError = (tpDiffs[minIdx] / tpComparision[minIdx]) * 100
    PTAngle = rAngles[minIdx]
    PTAxis = rAxes[minIdx]
    
    if (DEBUG_FLAG):
        print("Chosen rotation point: {}".format(minIdx))
        
    return rMatrices[minIdx], ((minIdx + 2) % 4), [PTError, PTAngle, PTAxis]

@jit
def deprojectAll(np_frame, ppxy, fxy, coeffs, model):
    # Algorithm adapted from rs2_deproject_pixel_to_point in librealsense
    # (https://github.com/IntelRealSense/librealsense/blob/e9f05c55f88f6876633bd59fd1cb3848da64b699/src/cuda/rscuda_utils.cuh)
    # Slower than just a for loop going through each of the pixels for some reason?

    np_points = np.zeros((1, 3), dtype='int16')
    for ix, iy in np.ndindex(np_frame.shape):
        np_appended = np.array([[iy, ix, np_frame[ix, iy]]], dtype='int16')
        np_points = np.vstack([np_points, np_appended])

    np_points = np.delete(np_points, (0), axis=0)

    x = (np_points[:, 0] - ppxy[0]) / fxy[0]
    y = (np_points[:, 1] - ppxy[1]) / fxy[1]

    if model == '"Inverse Brown Conrady"':
        r2 = (x * x) + (y * y)
        f = 1 + (coeffs[0] * r2) + (coeffs[1] * r2*r2) + (coeffs[4] * r2*r2*r2)
        ux = (x * f) + (2 * coeffs[2] * x * y) + (coeffs[3] * (r2 + 2 * x*x))
        uy = (y * f) + (2 * coeffs[3] * x * y) + (coeffs[2] * (r2 + 2 * y*y))
        x = ux
        y = uy

    points = np.zeros(np_points.shape)
    points[:, 0] = np_points[:, 2] * x
    points[:, 1] = np_points[:, 2] * y
    points[:, 2] = np_points[:, 2]

    np_points = cp.asnumpy(points)

    return np_points

def cp_deprojectPixelToPoint(np_frame, ppxy, fxy, coeffs, model):
    # Algorithm adapted from rs2_deproject_pixel_to_point in librealsense
    # (https://github.com/IntelRealSense/librealsense/blob/e9f05c55f88f6876633bd59fd1cb3848da64b699/src/cuda/rscuda_utils.cuh)
    # This seems to be slower even when using gpu for matrix operations (maybe because of overhead moving the data to gpu?)

    mempool = cp.get_default_memory_pool()
    mempool.set_limit(size=1024**3)
    # print(mempool.get_limit())
    pinned_mempool = cp.get_default_pinned_memory_pool()

    cp_frame = cp.array(np_frame)
    ppxy = cp.array(ppxy)
    fxy = cp.array(fxy)
    coeffs = cp.array(coeffs)
    cp_points = cp.array([0, 0, 0], dtype='int16')
    # print(mempool.used_bytes())
    # print(mempool.total_bytes())
    for ix, iy in cp.ndindex(cp_frame.shape):
        cp_appended = cp.array([iy, ix, cp_frame[ix, iy].get()], dtype='int16')
        cp_points = cp.vstack([cp_points, cp_appended])
        # print("{}, {}, {}".format(ix, iy, cp_frame[ix, iy]))
        
        if mempool.total_bytes() > 1000**3:
            mempool.free_all_blocks()

    # print("Used bytes: {}; Total bytes: {}; Free bytes: {}".format(mempool.used_bytes(), mempool.total_bytes(), mempool.free_bytes()))
    # print("Free blocks: {}".format(mempool.n_free_blocks()))
    # cp_points = cp.delete(cp_points, (0), axis=0)

    x = (cp_points[:, 0] - ppxy[0]) / fxy[0]
    y = (cp_points[:, 1] - ppxy[1]) / fxy[1]
    print("Shapes of x and y: {}, {}".format(x.shape, y.shape))

    if model == '"Inverse Brown Conrady"':
        r2 = (x * x) + (y * y)
        f = 1 + (coeffs[0] * r2) + (coeffs[1] * r2*r2) + (coeffs[4] * r2*r2*r2)
        ux = (x * f) + (2 * coeffs[2] * x * y) + (coeffs[3] * (r2 + 2 * x*x))
        uy = (y * f) + (2 * coeffs[3] * x * y) + (coeffs[2] * (r2 + 2 * y*y))
        x = ux
        y = uy

    # print(cp_points.shape)
    # print(cp_points[:, 2].shape)
    points = cp.zeros(cp_points.shape)
    # print(points.shape)
    points[:, 0] = cp_points[:, 2] * x
    points[:, 1] = cp_points[:, 2] * y
    points[:, 2] = cp_points[:, 2]

    np_points = cp.asnumpy(points)

    del points, cp_points
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()

    return np_points

def deprojectPixelToPoint(pixel, depth, ppxy, fxy, coeffs, model):
    # Algorithm adapted from rs2_deproject_pixel_to_point in librealsense
    # (https://github.com/IntelRealSense/librealsense/blob/e9f05c55f88f6876633bd59fd1cb3848da64b699/src/cuda/rscuda_utils.cuh)

    point = [0, 0, 0]

    x = (pixel[0] - ppxy[0]) / fxy[0]
    y = (pixel[1] - ppxy[1]) / fxy[1]

    if model == '"Inverse Brown Conrady"':
        r2 = (x * x) + (y * y)
        f = 1 + (coeffs[0] * r2) + (coeffs[1] * r2*r2) + (coeffs[4] * r2*r2*r2)
        ux = (x * f) + (2 * coeffs[2] * x * y) + (coeffs[3] * (r2 + 2 * x*x))
        uy = (y * f) + (2 * coeffs[3] * x * y) + (coeffs[2] * (r2 + 2 * y*y))
        x = ux
        y = uy

    point[0] = depth * x
    point[1] = depth * y
    point[2] = depth

    return point

def projectPointToPixel(point, ppxy, fxy, coeffs, model):
    # Algorithm adapted from rs2_project_point_to_pixel in librealsense
    # (https://github.com/IntelRealSense/librealsense/blob/e9f05c55f88f6876633bd59fd1cb3848da64b699/src/cuda/rscuda_utils.cuh)

    pixel = [0, 0]

    x = point[0] / point[2]
    y = point[1] / point[2]

    if model == '"Modified Brown Conrady"':
        r2 = x * x + y * y
        f = 1 + coeffs[0] * r2 + coeffs[1] * r2*r2 + coeffs[4] * r2*r2*r2
        x *= f
        y *= f
        dx = x + 2 * coeffs[2] * x*y + coeffs[3] * (r2 + 2 * x*x)
        dy = y + 2 * coeffs[3] * x*y + coeffs[2] * (r2 + 2 * y*y)
        x = dx
        y = dy

    elif model == '"F-Theta"':
        r = math.sqrt(x*x + y * y)
        rd = (1.0 / coeffs[0] * math.atan(2 * r* math.tan(coeffs[0] / 2.0)))
        x *= rd / r
        y *= rd / r

    pixel[0] = x * fxy[0] + ppxy[0]
    pixel[1] = y * fxy[1] + ppxy[1]

    return pixel
        
def perspectiveTransformHandler(intrinsics, np_depth_frame, perspectivePoints, scaling_factor, pc, rotationMatrix, fulcrumPixel_idx, isPaused, np_depth_frame_prev, np_depth_frame_prev_prev, PTError, PTAngle, PTAxis, DEBUG_FLAG = False, rs2_functions = True):
    '''
    Transforms depth frame perspective to face camera head-on.

    Parameters:
        intrinsics (list): List of camera intrinsics returned by pyrealsense for the specific camera used.
        np_depth_frame (NxM Numpy array of floats): Numpy array containing depth values of each pixel.
        perspectivePoints (list of points): List of manually selected points to be used to calculate rotation matrix and crop image appropratly.
        scaling_factor (float): Scaling factor to convert arbitrary depth unit to meters.
        pc (proprietary list of points): Pointcloud representation of the depth frame from the pyrealsense library.
        rotationMatrix (3x3 Numpy matrix OR None): Numpy array containing calculated rotation matrix.
        fulcrumPixel_idx (int OR None): Index of middle point used for calculating rotation matrix.
        isPaused (boolean): Flag showing if the playback is paused on a single frame.
        np_depth_frame_prev (NxM Numpy array of floats): Numpy array containing depth values of each pixel for the previous frame.
        np_depth_frame_prev_prev (NxM Numpy array of floats): Numpy array containing depth values of each pixel for the frame before the last.
        DEBUG_FLAG (boolean): Flag to show/hide debug prints. Defaults to False (Don't show debugging information).

    Returns:
        np_final_frame (KxL Numpy array of floats): Numpy array conatining depth values of each pixel after perspective transformation.
        contours (list of lists of points): List of lists, where each child list contains points that make up contours.
        contours_filteredArea (list of lists of points): Similar to 'contours' with some lists removed according to area criteria.
        contours_filteredCircularity (list of lists of points): Similar to 'contours_filteredArea' with some lists removed according to shape criteria.
        headSphere (list of lists of points): All contours of selected head region ordered from smallest to largest.
        maxHeadSlice (int): Index of top-most depthslice containing a headSphere contour.
        torsoSphere (list of lists of points): All contours of selected torso region ordered from smallest to largest.
        [PTError, PTAngle, PTAxis] (List of floats): Array containing information on the calculation of the rotation matrix (error, angle, and axis).
    '''

    points = []
    # camera_intrinsic_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
    #                                     [0, intrinsics.fy, intrinsics.ppy],
    #                                     [0, 0, 1]])
    # camera_rotation_matrix = np.identity(3)
    # camera_translation_matrix = np.array([0.0, 0.0, 0.0])
    # distortion_coeffs = np.asanyarray(intrinsics.coeffs)
    
    for pixel in perspectivePoints:
        depth = np_depth_frame[pixel[1],pixel[0]]
        if not rs2_functions:
            point = deprojectPixelToPoint(pixel, depth, (intrinsics.ppx, intrinsics.ppy), (intrinsics.fx, intrinsics.fy), intrinsics.coeffs, intrinsics.model)
        else:
            point = rs.rs2_deproject_pixel_to_point(intrinsics, pixel, depth)

        points.append(point)
    
    if rotationMatrix is None:
        rotationMatrix, fulcrumPixel_idx, err = calculateRotationMatrix(points, DEBUG_FLAG)
        PTError, PTAngle, PTAxis = err
    
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
    
    # Black out pixels that have not changed since the last frame
    # NOT WORKING YET
    
    # np_depth_frame_cp = np_depth_frame.copy()
    # if not isPaused and np_depth_frame_prev is not None and np_depth_frame_prev_prev is not None and np_depth_frame.shape == np_depth_frame_prev.shape:
    #     diffPixels = cv2.absdiff(np_depth_frame, np_depth_frame_prev)
    #     diffPixels_prev = cv2.absdiff(np_depth_frame, np_depth_frame_prev_prev)
    #     staticPixels = (diffPixels > 5)
    #     staticPixels_prev = (diffPixels > 5)
        
    #     staticOverPrev = (np.logical_or(staticPixels, staticPixels_prev)) * 1.0
    #     print(staticOverPrev)
    #     print(np_depth_frame)
    #     print(staticOverPrev.shape)
    #     np_depth_frame = np_depth_frame * staticOverPrev
    
    # if np_depth_frame_prev is not None:
    #     np_depth_frame_prev_prev = np_depth_frame_prev.copy()
    # np_depth_frame_prev = np_depth_frame_cp
        
    if not rs2_functions:
        fulcrumPoint = deprojectPixelToPoint(pPoints[fulcrumPixel_idx], np_depth_frame[pPoints[fulcrumPixel_idx][1], pPoints[fulcrumPixel_idx][0]], (intrinsics.ppx, intrinsics.ppy), (intrinsics.fx, intrinsics.fy), intrinsics.coeffs, intrinsics.model)
    else:
        fulcrumPoint = rs.rs2_deproject_pixel_to_point(intrinsics, pPoints[fulcrumPixel_idx], np_depth_frame[pPoints[fulcrumPixel_idx][1], pPoints[fulcrumPixel_idx][0]])
    fulcrumPointRotated = rotationMatrix.dot(np.asanyarray(fulcrumPoint).T).T
    fulcrumPixelDepth = fulcrumPointRotated[2] * scaling_factor
    
    # TESTING
    # Trying deprojection of whole frame at once
    # Seems to slow it down even more somehow (even when using GPU)
    
    # verts = []
    # if not rs2_functions:
    #     # verts = cp_deprojectPixelToPoint(np_depth_frame, (intrinsics.ppx, intrinsics.ppy), (intrinsics.fx, intrinsics.fy), intrinsics.coeffs, intrinsics.model)
    #     verts = deprojectAll(np_depth_frame, (intrinsics.ppx, intrinsics.ppy), (intrinsics.fx, intrinsics.fy), intrinsics.coeffs, intrinsics.model)
    # else:
    #     for ix, iy in np.ndindex(np_depth_frame.shape):
    #         depth = np_depth_frame[ix, iy]
            
    #         point = rs.rs2_deproject_pixel_to_point(intrinsics, [iy, ix], depth)
    #         verts.append(point)

    verts = []
    if rs2_functions:
        for ix, iy in np.ndindex(np_depth_frame.shape):
            depth = np_depth_frame[ix, iy]
            point = rs.rs2_deproject_pixel_to_point(intrinsics, [iy, ix], depth)
            verts.append(point)
    else:
        for ix, iy in np.ndindex(np_depth_frame.shape):
            depth = np_depth_frame[ix, iy]
            point = deprojectPixelToPoint([iy, ix], depth, (intrinsics.ppx, intrinsics.ppy), (intrinsics.fx, intrinsics.fy), intrinsics.coeffs, intrinsics.model)
            verts.append(point)
    
    np_verts = np.asanyarray(verts)
    print(np_verts.shape)
    print(np_depth_frame.shape)
    # pcPoints = pc.calculate(depth_frame)
    # np_verts = np.asanyarray(pcPoints.get_vertices(dims=2))
    np_verts_transformed = rotationMatrix.dot(np_verts.T).T
    np_verts_transformed = np_verts_transformed[~np.all(np_verts_transformed == 0, axis=1)]
    np_verts_transformed = np_verts_transformed
    
    # project back to 2D image with depth as data (WORKING BUT SLOW)   
    np_transformed_depth_frame = np.zeros([1080,1920])
    for vert in np_verts_transformed:
        if not rs2_functions:
            pixel = projectPointToPixel(vert, (intrinsics.ppx, intrinsics.ppy), (intrinsics.fx, intrinsics.fy), intrinsics.coeffs, intrinsics.model)
        else:
            pixel = rs.rs2_project_point_to_pixel(intrinsics, vert)

        if (pixel[0] < 960 and pixel[1] < 540 and pixel[0] >= -960 and pixel[1] >= -540):
            np_transformed_depth_frame[int(pixel[1] + 540),int(pixel[0]) + 960] = vert[2]
            
    # Remove rows and columns of all zeros
    np_transformed_depth_frame = np_transformed_depth_frame[~np.all(np_transformed_depth_frame == 0, axis=1)]
    np_transformed_depth_frame = np_transformed_depth_frame[:, ~np.all(np_transformed_depth_frame == 0, axis=0)]
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
        contours, contours_filteredArea, contours_filteredCircularity, headSphere, allHeadSpheres, maxHeadSlice, torsoSphere = crossSections(np_final_frame, fulcrumPixelDepth, scaling_factor, DEBUG_FLAG)
    
    return np_final_frame, contours, contours_filteredArea, contours_filteredCircularity, headSphere, maxHeadSlice, torsoSphere, rotationMatrix, fulcrumPixel_idx, [PTError, PTAngle, PTAxis]
    # return np_final_frame

def crossSections(np_depth_frame, fulcrumPixelDepth, scaling_factor, DEBUG_FLAG = False):
    '''
    Transforms depth frame perspective to face camera head-on.

    Parameters:
        np_depth_frame (NxM Numpy array of floats): Numpy array containing depth values of each pixel.
        fulcrumPixelDepth (float): Depth of the pixel used as the fulcrum of the rotation matrix calculation.
        scaling_factor (float): Scaling factor to convert arbitrary depth unit to meters.
        DEBUG_FLAG (boolean): Flag to show/hide debug prints. Defaults to False (Don't show debugging information).

    Returns:
    allContours, allContours_area, allContours_circularity, headSphere, headSpheres, maxHeadSlice, torsoSphere

        np_final_frame (KxL Numpy array of floats): Numpy array conatining depth values of each pixel after perspective transformation.
        allContours (list of lists of points): List of lists, where each child list contains points that make up contours.
        allContours_area (list of lists of points): Similar to 'contours' with some lists removed according to area criteria.
        allContours_circularity (list of lists of points): Similar to 'contours_filteredArea' with some lists removed according to shape criteria.
        headSphere (list of lists of points): All contours of selected head region ordered from smallest to largest.
        headSpheres (list of lists of lists of points): List of all possible headSpheres.
        maxHeadSlice (int): Index of top-most depthslice containing a headSphere contour.
        torsoSphere (list of lists of points): All contours of selected torso region ordered from smallest to largest.
    '''

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