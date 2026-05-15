import cv2
import os
import argparse
import numpy as np
from ultralytics import YOLO

# Global variables for the mouse click callback
clicked_point = None
selection_made = False

def mouse_click(event, x, y, flags, param):
    global clicked_point, selection_made
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        selection_made = True

def main():
    global clicked_point, selection_made
    
    # 1. Handle Command Line Arguments (Professional Touch)
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="Path to video file")
    args = vars(ap.parse_args())

    # 2. Setup AI Brain
    print("[INFO] Loading YOLOv8 model...")
    detector = YOLO('yolov8n.pt') 
    
    # 3. Choose Source (Video File vs. Webcam)
    source = args.get("video")
    if source and os.path.exists(source):
        print(f"[INFO] Source: Video File ({source})")
        camera = cv2.VideoCapture(source)
    else:
        print("[INFO] Source: Webcam")
        camera = cv2.VideoCapture(0)

    cv2.namedWindow("Sports Tracker")
    cv2.setMouseCallback("Sports Tracker", mouse_click)

    # 4. State Machine Variables
    state = "DETECT_AND_SELECT"
    tracker = None

    while True:
        # If we are selecting, we stay on the same frame
        if state != "DETECT_AND_SELECT":
            ret, frame = camera.read()
            if not ret: break
        else:
            # We only read one frame to show the selection screen
            ret, frame = camera.read()
            if not ret: break

        display_frame = frame.copy()

        # STATE: AI DETECTION & USER SELECTION
        if state == "DETECT_AND_SELECT":
            results = detector(frame)
            detected_boxes = []

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = detector.names[int(box.cls[0])]
                    detected_boxes.append((x1, y1, x2, y2, label))
                    
                    # Visual cues for the user
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(display_frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2.putText(display_frame, "CLICK an object. 'q' to quit.", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Sports Tracker", display_frame)

            # INNER LOOP: Wait here until user clicks an object
            selection_made = False
            while not selection_made:
                if cv2.waitKey(1) & 0xFF == ord('q'): return
                
                if selection_made:
                    cx, cy = clicked_point
                    for (x1, y1, x2, y2, label) in detected_boxes:
                        if x1 < cx < x2 and y1 < cy < y2:
                            tracker = cv2.TrackerCSRT_create()
                            tracker.init(frame, (x1, y1, x2-x1, y2-y1))
                            state = "TRACKING"
                            break
                    selection_made = False # Reset for next use
                    if state == "TRACKING": break

        # STATE: HIGH-SPEED TRACKING
        elif state == "TRACKING":
            success, box = tracker.update(frame)
            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            else:
                cv2.putText(display_frame, "LOST! Press 'r' to Reset", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Sports Tracker", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'): state = "DETECT_AND_SELECT"
            elif key == ord('q'): break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()