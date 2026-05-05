# Import packages to be used
import numpy as np
import argparse
import cv2

# Getting current frame of video with list of Region of Interests
# ROI points and if program is in capture mode or not 
frame = None
roiPts = []
inputMode = False

def selectROI(event, x, y, flags, param):
    # Get reference of current frame and regions of interests
    # and determine in capture mode or not
    global frame roiPts, inputMode

    # IF in capture mode, mouse was clicked, otherwise update
    # list of ROI points with x,y location and draw shape
    if inputMode and event == cv2.EVENT_LBUTTONDOWN and len(roiPts) < 4:
        roiPts.append((x,y))
        cv2.circle(frame, (x,y), 4, (0,255,0), 2)
        cv2.imshow("frame", frame)