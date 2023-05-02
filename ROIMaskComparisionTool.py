# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 14:18:47 2021

@author: Zein Hajj-Ali - zeinhajjali@sce.carleton.ca
"""

from tkinter import Tk
from tkinter.filedialog import askdirectory
from PIL import Image
import numpy as np
import sys

headROI_auto_filename = "headROI_auto.jpg"
headROI_manual_filename = "headROI_manual.jpg"
torsoROI_auto_filename = "torsoROI_auto.jpg"
torsoROI_manual_filename = "torsoROI_manual.jpg"
headROI_auto_path = None
headROI_manual_path = None
torsoROI_auto_path = None
torsoROI_manual_path = None

def selectDir():
    global headROI_auto_path, headROI_manual_path, torsoROI_auto_path, torsoROI_manual_path
    root = Tk()
    root.withdraw()
    root.overrideredirect(True)
    root.geometry('0x0+0+0')
    root.deiconify()
    root.lift()
    root.focus_force()
    folder_path = askdirectory()
    if not folder_path:
        sys.exit("No folder selected")
    root.destroy()

    print("Folder: \n{}\n".format(folder_path))
    
    headROI_auto_path = folder_path + "/" + headROI_auto_filename
    headROI_manual_path = folder_path + "/" + headROI_manual_filename
    torsoROI_auto_path = folder_path + "/" + torsoROI_auto_filename
    torsoROI_manual_path = folder_path + "/" + torsoROI_manual_filename

def main(): 
    # Select directory and open mask images
    selectDir()
    try:
        headROI_auto = Image.open(headROI_auto_path)
        headROI_manual = Image.open(headROI_manual_path)
        torsoROI_auto = Image.open(torsoROI_auto_path)
        torsoROI_manual = Image.open(torsoROI_manual_path)
    except FileNotFoundError as notFoundError:
        print(notFoundError.filename + " does not exist")
    else:
        np_headROI_auto = np.array(headROI_auto)
        np_headROI_manual = np.array(headROI_manual)
        np_torsoROI_auto = np.array(torsoROI_auto)
        np_torsoROI_manual = np.array(torsoROI_manual)
        
        # Convert int arrays to boolean masks
        np_headROI_auto = np_headROI_auto != 0
        np_headROI_manual = np_headROI_manual != 0
        np_torsoROI_auto = np_torsoROI_auto != 0
        np_torsoROI_manual = np_torsoROI_manual != 0
        
        dice = lambda auto, manual : np.sum(auto[manual==1])*2.0 / (np.sum(auto) + np.sum(manual))
        
        dice_head = dice(np_headROI_auto, np_headROI_manual)
        dice_torso = dice(np_torsoROI_auto, np_torsoROI_manual)
        # print(dice_head)
        print(f'Dice Torso: {dice_torso}')
        print(f'Dice Head: {dice_head}')
        
        # falsePos = lambda auto, manual : np.sum(auto[manual==0]) / np.sum(auto)
        
        # falsePos_head = falsePos(np_headROI_auto, np_headROI_manual)
        # falsePos_torso = falsePos(np_torsoROI_auto, np_torsoROI_manual)
        # print(falsePos_head)
        # print(falsePos_torso)

        jaccard = lambda d : (d / (2 - d))

        jaccard_torso = jaccard(dice_torso)
        jaccard_head = jaccard(dice_head)
        print(f'Jaccard Torso: {jaccard_torso}')
        print(f'Jaccard Head: {jaccard_head}')

        print(np.sum(np_torsoROI_auto))
        print(np.sum(np_torsoROI_manual))
        print(np.sum(np_headROI_auto))
        print(np.sum(np_headROI_manual))

        # accuracy = lambda auto, manual : np.sum(auto[manual==1]) / (np.sum(np.ones(manual.shape)))

        # accuracy_torso = accuracy(np_torsoROI_auto, np_torsoROI_manual)
        # print("Acc: \n{}\n".format(accuracy_torso))
        
if __name__ == "__main__":
    while True:
        main()