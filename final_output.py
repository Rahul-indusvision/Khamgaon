import cv2
import numpy as np

video_path = r"C:\New folder\Desktop\New folder (5)\final_croped.mp4"
cap = cv2.VideoCapture(video_path)

# top_left = (220, 95)
# bottom_right = (340, 250)
# box_color = (0, 255, 0)
# box_thickness = 2

top_left = (443, 195)       # (x1, y1)
bottom_right = (701, 492)   # (x2, y2)
box_color = (0, 255, 0)    
box_thickness = 2    


min_area = 650
aspect_ratio_range = (0.8, 2.2)

first_detection_frame = None
jamming_triggered = False
fps = cap.get(cv2.CAP_PROP_FPS)

Time = []

cv2.namedWindow("Foreground Only with Box and Contours", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Foreground Only with Box and Contours", 1000, 600)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, fg_mask = cv2.threshold(gray, 145, 255, cv2.THRESH_BINARY)

    x1, y1 = top_left
    x2, y2 = bottom_right
    roi_mask = fg_mask[y1:y2, x1:x2]

    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contour_count = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if 1200 > area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            if aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]:
                valid_contour_count += 1
                cv2.rectangle(frame, (x + x1, y + y1), (x + x1 + w, y + y1 + h), (0, 0, 255), 2)

    if valid_contour_count >= 2:
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if first_detection_frame is None:
            first_detection_frame = current_frame
        else:
            frame_diff = current_frame - first_detection_frame
            time_diff_sec = frame_diff / fps 
            if time_diff_sec >= 8:
                jamming_triggered = True
            else:
                first_detection_frame = current_frame

    cv2.rectangle(frame, top_left, bottom_right, box_color, box_thickness)

    if jamming_triggered:
        text = "JAMMING"
        color = (0, 0, 255)
        if not Time and first_detection_frame is not None:
            jamming_time = (int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - first_detection_frame) / fps
            Time.append(jamming_time)
        if Time:
            jamming_time_text = f"Time: {Time[0]:.2f}s"
            cv2.putText(frame, jamming_time_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    else:
        text = "NORMAL"
        color = (0, 255, 0)

    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    cv2.imshow("Foreground Only with Box and Contours", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
