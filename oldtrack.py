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
    global frame, roiPts, inputMode

    # IF in capture mode, mouse was clicked, otherwise update
    # list of ROI points with x,y location and draw shape
    if inputMode and event == cv2.EVENT_LBUTTONDOWN and len(roiPts) < 4:
        roiPts.append((x,y))
        cv2.circle(frame, (x,y), 4, (0,255,0), 2)
        cv2.imshow("frame", frame)

def main():
    # Creating argument parse and parsing the arguments 
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help = "path to the (optional) video file")
    args = vars(ap.parse_args())

    # Grab reference to current frame, and region of interests 
    # points and if in capture mode or not 
    global frame, roiPts, inputMode

    # If video path not supplies, grab reference to camera
    if not args.get("video", False):
        camera = cv2.VideoCapture(0)
    # Load video 
    else:
        camera = cv2.VideoCapture(args["video"])

    # Mouse callback -clicks
    cv2.namedWindow("frame")
    cv2.setMouseCallback("frame", selectROI)

    # Termination critieria for cam shift -> max of 10 iterations
    # or movement by 1 pixel 
    # And bounding box for ROI
    termination = (cv2.TERM_CRITERIA_EPS | cv2.TermCriteria_COUNT, 10, 1)
    roiBox = None   
    roiHist = None

    # Looping over frames
    while True:
        # Grab current frame
        # camera.read grabs 2 values, (T/F if frame reading was success, and the frame itself)
        (grabbed, frame) = camera.read()

        # Check if at end of the video since no frame could be grabbed 
        if not grabbed:
            break

        # Checking if we had the regions of Interests
        if roiBox is not None:
            # Convert frame to HSV color space = Going from the Red, Green, Blue color space
            # to Hue, Saturation, Value 
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            backProj = cv2.calcBackProject([hsv], [0], roiHist, [0, 180], 1)

        # Cam shift returns 2 values, ([estimated position, size, and orientation of object], and 
        # new estimated position of ROI)
        # Cam shift to back projection, converting points to the bounded shape and draw it
        # Arguments: backProj - Output of the Histogram projection
        #            roiBox - Bounding shape containing object we want to track
        #            termination - End of vid/ capture 
            (r, roiBox) = cv2.CamShift(backProj, roiBox, termination)

            # Testing testing
            pts = cv2.boxPoints(r)
            pts = np.int0(pts)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

        # Show frame and record if button is pressed 
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # Handle if 'i' key is pressed, then goes to ROI
        # Capture mode 
        if key == ord("i") and len(roiPts) < 4: 
            # Let user know we're in capture mode and clone frame
            inputMode = True
            orig = frame.copy()
            
            # Freezes frame 
            # Loops until 4 regions of interests are selected and 
            # then another button is pressed to exit ROI selection after 4 regions
            while len(roiPts) < 4:
                cv2.imshow("frame", frame)
                cv2.waitKey(1)

            # Determine top-left and bottom-right points
            roiPts = np.array(roiPts)
            s = roiPts.sum(axis = 1)
            tl = roiPts[np.argmin(s)]
            br = roiPts[np.argmax(s)]

            # Grabbing ROI for the bounding shape and convert to HSV space
            roi = orig[tl[1]:br[1], tl[0]:br[0]]
            
            # roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

            # Compute HSV histogram for ROI to store in bounding shape
            # only using Hue component 
            # Parameters of calcHist = ([image=roi], [channels], [histSize/bin size], [ranges of bin])
            if roi.size > 0: 
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                roiHist = cv2.calcHist([roi], [0], None, [16], [0, 180])
                roiHist = cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
                roiBox = (tl[0], tl[1], br[0] - tl[0], br[1] - tl[1])

            inputMode = False

            # If user presses 'q' it stops loop
        elif key == ord("q"):
            break

        # Cleanup camera and close any open windows
        camera.release()
        cv2.destroyAllWindows()

        if __name__ == "__main__":
            main()