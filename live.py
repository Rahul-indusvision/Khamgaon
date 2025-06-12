import cv2
import threading
from pycomm3 import LogixDriver
from ultralytics import YOLO

# ------------------- PLC Setup -------------------
plc_client = None
plc_client_on = False
plc_lock = threading.Lock()

def initialize_plc(plc_ip):
    global plc_client, plc_client_on
    with plc_lock:
        try:
            plc_client = LogixDriver(plc_ip)
            plc_client.open()
            plc_client_on = True
            print("Connected to PLC IP:", plc_ip)
        except Exception as e:
            print(e)
            print("Failed to connect to PLC")

def reset_plc(plc_ip):
    global plc_client, plc_client_on
    with plc_lock:
        if not plc_client_on:
            try:
                if plc_client:
                    plc_client.close()
                plc_client = LogixDriver(plc_ip)
                plc_client.open()
                plc_client_on = True
                print("Reconnected to PLC")
            except Exception as e:
                print(f"Failed to reconnect to PLC: {e}")

def put_text_with_background(frame, text, position, font, font_scale, text_color, thickness, bg_color=(0, 0, 0)):
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    cv2.rectangle(frame, (x, y - text_height - baseline), (x + text_width, y + baseline), bg_color, -1)
    cv2.putText(frame, text, position, font, font_scale, text_color, thickness)

# ------------------- Config -------------------
rtsp_url = "rtsp://admin:admin@123@192.168.10.17:554"
PLC_IP_ADDRESS = '192.168.10.222'
tag_list = [
    'BSM_AI_CAMERA_READ.0',  # CUTTER READY
    'BSM_AI_CAMERA_READ.1',  # EMERGENCY
    'BSM_AI_CAMERA_READ.2',  # CYLINDER
    'BSM_AI_CAMERA_READ.3',  # CONVEYOR FORWARD
    'BSM_AI_CAMERA_READ.4',  # CONVEYOR REVERSE
]

model = YOLO(r"C:\New folder\Desktop\final testing\best.pt")
cap = cv2.VideoCapture(rtsp_url)
if not cap.isOpened():
    print("Error: Unable to open video stream.")
    exit()

fps = 30
frame_count = 0

# ROI and jamming parameters
top_left = (450, 195)
bottom_right = (670, 500)
box_color = (0, 255, 0)
box_thickness = 2

soap_start_frame = None
soap_reset_frame = None
soap_jamming = False

bar_jamming = False
bar_jam_reset_frame = None

initialize_plc(PLC_IP_ADDRESS)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Reconnecting to camera...")
        cap.release()
        cap = cv2.VideoCapture(rtsp_url)
        continue

    try:
        plc_data = plc_client.read(*tag_list)
        plc_values = [tag.value for tag in plc_data]

        flapper_open = plc_values[2]  # CYLINDER
        conveyor_moving = plc_values[3]  # CONVEYOR FORWARD

        if not flapper_open or not conveyor_moving:
            status_text = "BYPASSED: Flapper Open or Conveyor Stopped"
            put_text_with_background(frame, status_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            soap_detected = False
            bar_count = 0

            results = model(frame, conf=0.6, verbose=False)[0]
            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                cv2.putText(frame, f"{label}", (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                if label.lower() == "soap":
                    soap_detected = True
                elif label.lower() == "bars":
                    bar_count += 1

            # --- Soap Jamming Logic ---
            if soap_detected:
                if soap_start_frame is None:
                    soap_start_frame = frame_count
                elif frame_count - soap_start_frame >= 4 * fps:
                    soap_jamming = True
                soap_reset_frame = None
            else:
                if soap_jamming:
                    if soap_reset_frame is None:
                        soap_reset_frame = frame_count
                    elif frame_count - soap_reset_frame >= 4 * fps:
                        soap_jamming = False
                        soap_start_frame = None
                        soap_reset_frame = None
                else:
                    soap_start_frame = None
                    soap_reset_frame = None

            # --- Bar Jamming Logic ---
            if bar_count >= 3:
                bar_jamming = True
                bar_jam_reset_frame = None
            else:
                if bar_jamming:
                    if bar_jam_reset_frame is None:
                        bar_jam_reset_frame = frame_count
                    elif frame_count - bar_jam_reset_frame >= 10 * fps:
                        bar_jamming = False
                        bar_jam_reset_frame = None

            # --- Display Status ---
            if soap_jamming or bar_jamming:
                status_text = "JAMMING"
                status_color = (0, 0, 255)
            else:
                status_text = "NORMAL"
                status_color = (0, 255, 0)

            cv2.rectangle(frame, top_left, bottom_right, box_color, box_thickness)
            put_text_with_background(frame, status_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 3)

    except Exception as e:
        print(f"PLC read error: {e}")
        thread = threading.Thread(target=reset_plc, args=(PLC_IP_ADDRESS,))
        thread.start()

    frame = cv2.resize(frame, (960, 540))
    cv2.imshow('PLC Camera Jamming Detection', frame)

    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
