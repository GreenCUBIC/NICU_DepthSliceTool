# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 23:20:03 2021

source /opt/ros/noetic/setup.bash
python3 realsense_rosbag_merge.py MEMEA_files/p13/p6Merged.bag /mnt/z/ 6 Patient6 30 42

@author: Zein Hajj-Ali - zeinhajjali@sce.carleton.ca
"""

import os
import argparse
import glob

def main():
    
    parser = argparse.ArgumentParser(description='Merge one or more realsense bag files with the possibilities of filtering topics.')

    parser.add_argument('outputbag',
                        help='output bag file with topics merged')
    parser.add_argument('rootFolder', help='root patient folder')
    parser.add_argument('patientNum', help='Patient number')
    parser.add_argument('initialFilename', help='initial filename (no extension')
    parser.add_argument('startIdx', help='starting bag file')
    parser.add_argument('endIdx', help='ending bag file')
    # parser.add_argument('inputbag', nargs='+',
                        # help='input bag files')

    args = parser.parse_args()
    
    outputbag = args.outputbag
    rootFolder = args.rootFolder
    patientNum = args.patientNum
    initialFile = args.initialFilename
    startIdx = int(args.startIdx)
    endIdx = int(args.endIdx)

    firstFile = rootFolder + 'Patient_' + str(patientNum) + '/Video_Data/' + initialFile + '.bag'

    inputbags = [firstFile]
    # for i in range(startIdx, endIdx):
    otherbags = []
    if startIdx < 10:
        tempbags = glob.glob(rootFolder + 'Patient_' + str(patientNum) + '/Video_Data/' + initialFile + '_part_[0-9].bag')
        otherbags = otherbags + tempbags
        if endIdx >= 10:
            tempbags = glob.glob(rootFolder + 'Patient_' + str(patientNum) + '/Video_Data/' + initialFile + '_part_[0-9][0-9].bag')
            otherbags = otherbags + tempbags
            if endIdx >= 100:
                tempbags = glob.glob(rootFolder + 'Patient_' + str(patientNum) + '/Video_Data/' + initialFile + '_part_[0-9][0-9][0-9].bag')
                otherbags = otherbags + tempbags
    elif (startIdx >= 10 and startIdx < 100):
        tempbags = glob.glob(rootFolder + 'Patient_' + str(patientNum) + '/Video_Data/' + initialFile + '_part_[0-9][0-9].bag')
        otherbags = otherbags + tempbags
        if endIdx >= 100:
            tempbags = glob.glob(rootFolder + 'Patient_' + str(patientNum) + '/Video_Data/' + initialFile + '_part_[0-9][0-9][0-9].bag')
            otherbags = otherbags + tempbags
    elif (startIdx >= 100):
        tempbags = glob.glob(rootFolder + 'Patient_' + str(patientNum) + '/Video_Data/' + initialFile + '_part_[0-9][0-9][0-9].bag')
        otherbags = otherbags + tempbags

    substr = []
    for i in range(int(startIdx), int(endIdx)):
        sub = '_part_' + str(i) + '.bag'
        substr.append(sub)

    filteredbags = [str for str in otherbags if any(sub in str for sub in substr)]

    inputbags = inputbags + filteredbags

    print('output: {}'.format(outputbag))
    # print(otherbags)
    # print(substr)
    # print(filteredbags)
    print('inputBags: {}'.format(' '.join(inputbags)))
    
    # inputbags.append(bagString)
    # inputbags = args.inputbag
    # inputbags = ["/mnt/z/Patient_6/Video_Data"]
    
    # headerfile = "headerfile.bag"
    
    datafile = '~/datafile_p' + str(patientNum) + '.bag'
    
    if os.path.exists(datafile):
        os.remove(datafile)
        
    if os.path.exists(outputbag):
        os.remove(outputbag)

    tempbags = []
    for bag in inputbags:
        bag = '"' + bag + '"'
        tempbags.append(bag)

    inputbags = tempbags

    # Create file with headers
    # os.system("python3 rosbag_merge.py " + headerfile + " " + inputbags[0] + " " + inputbags[1] + " -t '/device_0/sensor_0/Depth_0/image/data /device_0/sensor_0/Depth_0/image/metadata /device_0/sensor_0/Infrared_1/image/data /device_0/sensor_0/Infrared_1/image/metadata /device_0/sensor_1/Color_0/image/data /device_0/sensor_1/Color_0/image/metadata' -v -i")

    # Create file with data
    os.system("python3 rosbag_merge.py " + datafile + " " + " ".join(inputbags[1:]) + " -t '/device_0/sensor_0/Depth_0/image/data /device_0/sensor_0/Depth_0/image/metadata /device_0/sensor_0/Infrared_1/image/data /device_0/sensor_0/Infrared_1/image/metadata /device_0/sensor_1/Color_0/image/data /device_0/sensor_1/Color_0/image/metadata' -v")
    
    # Merge them
    os.system("python3 rosbag_merge.py " + outputbag + " " + inputbags[0] + " " + datafile + " -v")
    
    if os.path.exists(datafile):
        os.remove(datafile)

if __name__ == "__main__":
    main()