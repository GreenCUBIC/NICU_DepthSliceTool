# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 23:20:03 2021

python3 realsense_rosbag_merge.py p26/p26Merged.bag p26/p26.bag p26/p26_1.bag p26/p26_2.bag p26/p26_3.bag p26/p26_4.bag p26/p26_5.bag p26/p26_6.bag p26/p26_7.bag p26/p26_8.bag p26/p26_9.bag p26/p26_10.bag p26/p26_11.bag p26/p26_12.bag p26/p26_13.bag p26/p26_14.bag p26/p26_15.bag p26/p26_16.bag p26/p26_17.bag p26/p26_18.bag p26/p26_19.bag p26/p26_20.bag

@author: Zein Hajj-Ali - zeinhajjali@sce.carleton.ca
"""

import os
import argparse

def main():
    
    parser = argparse.ArgumentParser(description='Merge one or more realsense bag files with the possibilities of filtering topics.')

    parser.add_argument('outputbag',
                        help='output bag file with topics merged')
    parser.add_argument('inputbag', nargs='+',
                        help='input bag files')

    args = parser.parse_args()
    
    outputbag = args.outputbag
    inputbags = args.inputbag
    
    # headerfile = "headerfile.bag"
    
    datafile = "datafile.bag"
    
    if os.path.exists(datafile):
        os.remove(datafile)
        
    if os.path.exists(outputbag):
        os.remove(outputbag)

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