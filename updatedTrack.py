import cv2
import os
import argparse
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
    
    # 1. Handle Command Line Arguments
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
        print("[INFO] Source: Live Webcam Feed")
        camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("[ERROR] Could not open video source.")
        return

    cv2.namedWindow("Sports Tracker")
    cv2.setMouseCallback("Sports Tracker", mouse_click)

    # 4. State Machine Variables
    state = "DETECT_AND_SELECT"
    tracker = None

    while True:
        # We ALWAYS read a new frame every iteration to keep the webcam live
        ret, frame = camera.read()
        if not ret: 
            print("[INFO] Video stream ended or failed to read frame.")
            break

        display_frame = frame.copy()

        # ==========================================
        # STATE 1: LIVE DETECTION & SELECTION
        # ==========================================
        if state == "DETECT_AND_SELECT":
            # Run YOLO on the current frame
            results = detector(frame, verbose=False)
            detected_boxes = []

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = detector.names[int(box.cls[0])]
                    detected_boxes.append((x1, y1, x2, y2, label))
                    
                    # Draw live tracking choices
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(display_frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2.putText(display_frame, "Live Feed: CLICK an object to track. 'q' to quit.", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Check if a mouse click occurred during this frame step
            if selection_made:
                cx, cy = clicked_point
                for (x1, y1, x2, y2, label) in detected_boxes:
                    # Verify if click point lies inside a bounding box
                    if x1 < cx < x2 and y1 < cy < y2:
                        print(f"[INFO] Initializing tracker for: {label}")
                        tracker = cv2.TrackerCSRT_create()
                        # Hand off coordinates to CSRT: (x, y, width, height)
                        tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
                        state = "TRACKING"
                        break
                
                selection_made = False  # Reset flag whether box hit or missed

        # ==========================================
        # STATE 2: HIGH-SPEED TRACKING
        # ==========================================
        elif state == "TRACKING":
            success, box = tracker.update(frame)
            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(display_frame, "Tracking... Press 'r' to Reset", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "LOST! Press 'r' to Reset", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 5. Render Output Frame
        cv2.imshow("Sports Tracker", display_frame)
        
        # 6. Global Key Intercepts
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'): 
            print("[INFO] Resetting state machine to detection mode.")
            state = "DETECT_AND_SELECT"
        elif key == ord('q'): 
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()