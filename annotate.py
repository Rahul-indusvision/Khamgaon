import cv2
import time
from ultralytics import YOLO

model = YOLO(r"C:\New folder\Desktop\final testing\best.pt")
cap = cv2.VideoCapture(r"C:\New folder\Desktop\final testing\rare jam crop.mp4")

# Define ROI box
top_left = (450, 195)
bottom_right = (670, 500)
box_color = (0, 255, 0)
box_thickness = 2

# top_left = (220, 125)
# bottom_right = (340, 250)
# box_color = (0, 255, 0)
# box_thickness = 2

# Soap jamming variables
soap_start_time = None
soap_reset_start = None
soap_jamming = False

# Bar jamming variables
bar_jamming = False
bar_jam_reset_start = None

cv2.namedWindow("YOLO + Jamming Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO + Jamming Detection", 1000, 600)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    soap_detected = False
    bar_count = 0

    results = model(frame, conf=0.6, verbose=False)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        color = (255, 0, 0)

        cv2.circle(frame, (cx, cy), 5, color, -1)
        cv2.putText(frame, f"{label}", (cx + 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if label.lower() == "soap":
            soap_detected = True
        elif label.lower() == "bars":
            bar_count += 1

    # --- Soap Jamming Logic with reset timer ---
    if soap_detected:
        if soap_start_time is None:
            soap_start_time = current_time
        elif current_time - soap_start_time >= 4:
            soap_jamming = True
        soap_reset_start = None  # reset "no-soap" timer
    else:
        if soap_jamming:
            if soap_reset_start is None:
                soap_reset_start = current_time
            elif current_time - soap_reset_start >= 4:
                soap_jamming = False
                soap_start_time = None
                soap_reset_start = None
        else:
            soap_start_time = None
            soap_reset_start = None

    # --- Bar Jamming Logic ---
    if bar_count >= 3:
        bar_jamming = True
        bar_jam_reset_start = None
    else:
        if bar_jamming:
            if bar_jam_reset_start is None:
                bar_jam_reset_start = current_time
            elif current_time - bar_jam_reset_start >= 6:
                bar_jamming = False
                bar_jam_reset_start = None

    # --- Status Display ---
    if soap_jamming or bar_jamming:
        status_text = "JAMMING"
        status_color = (0, 0, 255)
    else:
        status_text = "NORMAL"
        status_color = (0, 255, 0)

    cv2.rectangle(frame, top_left, bottom_right, box_color, box_thickness)
    cv2.putText(frame, status_text, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 3)

    cv2.imshow("Jamming Detection", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
