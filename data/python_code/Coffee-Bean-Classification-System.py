import serial  # Serial communication with Arduino
import time    # Time management
import cv2     # OpenCV for video processing
from ultralytics import YOLO  # YOLO for object detection
from collections import Counter, deque  # For counting and maintaining history
import threading  # For handling concurrent tasks
import os  # For file and directory operations
from datetime import datetime  # For timestamping logs

# --- การตั้งค่า (Configuration) ---
DEBUG_MODE = True
SERIAL_PORT = 'COM5'  # Change to your Arduino port
BAUD_RATE = 9600
CAMERA_INDEX = 0

YOLO_MODEL_PATH = 'YOUR_YOLO_MODEL_PATH.pt'  # Change to your YOLO model path

# --- การตั้งค่ากล้อง ---
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS = 30

# --- โซนและเวลา ---
ARMING_ZONE_START_X_RATIO = 0.25        # โซนดีดเมล็ดที่ไม่ดีออก (เริ่มที่ 25% ของความกว้างภาพ)
ARMING_ZONE_END_X_RATIO = 0.75          # โซนดีดเมล็ดที่ไม่ดีออก
CONFIDENCE_THRESHOLD = 0.5              # มั่นใจ 50% ขึ้นไปถึงจะนับ
LEAD_TIME_MS = 800                      # รอ 800ms ก่อนสั่งดีด
FRAMES_TO_CONFIRM_STABILITY = 5         # เห็นอย่างน้อย 5 frame จึงเชื่อว่า class ถูกต้อง
CLASS_VOTING_HISTORY = 5                # เก็บประวัติ class 5 ค่าเพื่อโหวต
OBJECT_TIMEOUT_MS = 5000                # ถ้าวัตถุหายไปเกิน 5 วินาที จะลบทิ้ง
COMMAND_COOLDOWN_MS = 300               # ระยะห่างการส่งคำสั่งไป Arduino

GOOD_CLASSES = {"A", "AA", "AAA", "B", "Dry", "Honey", "Pea berry", "Wash"}
BAD_CLASSES = {"Black", "Chipped", "Elephant ear", "Faded", "Split", "Triangle", "Weevil-infested"}

# --- NEW: Capture/Logging options ---
SAVE_FRAMES = True                  # บันทึกเฟรมเต็มเป็นระยะ
FRAME_SAVE_EVERY_N = 1              # บันทึกทุก N เฟรม
SAVE_CROPS_ON_CONFIRM = True        # บันทึกครอปเมื่อคอนเฟิร์มคลาส
SAVE_CROPS_ON_EJECT = True          # บันทึกครอปเมื่อสั่งดีด
CROP_PADDING = 8                    # เผื่อขอบครอป (พิกเซล)

# --- NEW: Run-scoped folders ---
run_tag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir = os.path.join("logs", run_tag)
cap_dir = os.path.join("captures", run_tag)
os.makedirs(run_dir, exist_ok=True)
os.makedirs(os.path.join(cap_dir, "frames"), exist_ok=True)
os.makedirs(os.path.join(cap_dir, "crops", "confirmed"), exist_ok=True)
os.makedirs(os.path.join(cap_dir, "crops", "ejected"), exist_ok=True)

# Log files (ข้อความ + CSV events)
log_filename = os.path.join(run_dir, f"arduino_log.txt")  # ตอนนี้ใช้เก็บทั้ง Arduino และ system log
event_csv_path = os.path.join(run_dir, "events.csv")
_event_csv_header_written = False

arduino_running = True

# --- อ่านข้อมมูลจาก Arduino และบันทึกลง log ---
def read_arduino_feedback(ser, display_logs):
    global log_filename
    while arduino_running:
        try:
            if ser and ser.in_waiting:
                arduino_msg = ser.readline().decode('utf-8', errors='ignore').strip()
                if arduino_msg:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    log_entry = f"[{timestamp}] {arduino_msg}"
                    display_logs.append(log_entry)
                    with open(log_filename, "a", encoding='utf-8') as f:
                        f.write(log_entry + "\n")
        except Exception as e:
            if DEBUG_MODE:
                print(f"Arduino read error: {e}")
        time.sleep(0.01)

# --- ฟังก์ชันเตรียมอุปกรณ์: Serial (เชื่อมต่อกับ Arduino) ---
def init_serial():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        print(f"Connecting to Arduino on port {SERIAL_PORT}...")
        time.sleep(2.5)
        while ser.in_waiting:
            print(f"Startup message from Arduino: {ser.readline().decode('utf-8', errors='ignore').strip()}")
        print(f"Successfully connected to Arduino on port {SERIAL_PORT}")
        return ser
    except serial.SerialException as e:
        print(f"Error: Unable to open port {SERIAL_PORT}. {e}")
        return None

# --- ฟังก์ชันเตรียมโมเดล YOLO ---
def init_yolo():
    try:
        model = YOLO(YOLO_MODEL_PATH)
        class_names = model.names
        print("YOLO model loaded successfully")
        return model, class_names
    except Exception as e:
        print(f"Error: Unable to load YOLO model from '{YOLO_MODEL_PATH}'. {e}")
        return None, None

# --- ฟังก์ชันเตรียมกล้อง ---
def init_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Cannot open camera on index {CAMERA_INDEX}.")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    print(f"Camera initialized on index {CAMERA_INDEX} with resolution {CAMERA_WIDTH}x{CAMERA_HEIGHT} at {CAMERA_FPS} FPS")
    print("Camera opened successfully")
    return cap

# --- ฟังก์ชันบันทึกข้อความลง log (จอ + ไฟล์) ---
def log_message(message):
    """เขียน log ทั้งบนจอ (display_logs) และลงไฟล์ log_filename"""
    global log_filename
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"

    # console
    if DEBUG_MODE:
        print(line)
    # สำหรับ overlay บนจอ
    display_logs.append(line)

    # เขียนลงไฟล์
    try:
        with open(log_filename, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        if DEBUG_MODE:
            print(f"File log error: {e}")

# --- ฟังก์ชันส่งคำสั่งไปยัง Arduino ---
def send_arm_command(obj_id, class_name):
    global last_command_time, ejection_total_count
    try:
        ser.write(b'A')
        ser.flush()
        last_command_time = int(time.time() * 1000)
        ejection_total_count += 1
        log_message(f"ARM: Sent command for ID {obj_id} ('{class_name}') - Total: {ejection_total_count}")
        return True
    except Exception as e:
        log_message(f"ERROR: Failed to send ARM command - {e}")
        return False

# --- NEW: Utilities for CSV & image saves ---
def _clip_int(v, lo, hi):
    return max(lo, min(int(v), hi))

def _pad_box(x1, y1, x2, y2, pad, W, H):
    return (
        _clip_int(x1 - pad, 0, W - 1),
        _clip_int(y1 - pad, 0, H - 1),
        _clip_int(x2 + pad, 0, W - 1),
        _clip_int(y2 + pad, 0, H - 1),
    )

def log_event(event, **fields):
    """เขียน event แบบมีโครงสร้างลง CSV"""
    global _event_csv_header_written
    try:
        import csv
        file_exists = os.path.exists(event_csv_path)
        with open(event_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["ts", "event", *sorted(fields.keys())])
            if not _event_csv_header_written or not file_exists:
                writer.writeheader()
                _event_csv_header_written = True
            writer.writerow({"ts": int(time.time() * 1000), "event": event, **fields})
    except Exception as e:
        if DEBUG_MODE:
            print(f"CSV log error: {e}")

_frame_counter = 0
def maybe_save_frame(frame):
    """บันทึกเฟรมเต็มเป็นระยะ"""
    global _frame_counter
    if not SAVE_FRAMES:
        return
    _frame_counter += 1    # นับเฟรม
    if _frame_counter % FRAME_SAVE_EVERY_N == 0:
        p = os.path.join(cap_dir, "frames", f"frame_{_frame_counter:06d}.jpg")
        try:
            cv2.imwrite(p, frame)
        except Exception as e:
            if DEBUG_MODE:
                print(f"Save frame error: {e}")

def save_crop(frame, box_xyxy, class_name, obj_id, bucket):
    """บันทึกรูปครอป (confirmed/ejected) โดยแยกตามคลาส"""
    try:
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = map(int, box_xyxy)
        x1, y1, x2, y2 = _pad_box(x1, y1, x2, y2, CROP_PADDING, W, H)
        crop = frame[y1:y2, x1:x2]
        cls_dir = os.path.join(cap_dir, "crops", bucket, str(class_name))
        os.makedirs(cls_dir, exist_ok=True)
        ts = int(time.time() * 1000)
        path = os.path.join(cls_dir, f"id{obj_id}_t{ts}.jpg")
        cv2.imwrite(path, crop)
        return path
    except Exception as e:
        if DEBUG_MODE:
            print(f"Save crop error: {e}")
        return None

# --- main loop ---
# --- เริ่มระบบ ---
ser = init_serial()
if not ser:
    raise SystemExit(1)
model, class_names = init_yolo()
if not model:
    raise SystemExit(1)
cap = init_camera()
if not cap:
    raise SystemExit(1)

tracked_objects = {}
last_command_time = 0
good_seed_count = 0
bad_seed_count = 0
ejection_total_count = 0
display_logs = deque(maxlen=10)

arduino_thread = threading.Thread(target=read_arduino_feedback, args=(ser, display_logs), daemon=True)
arduino_thread.start()

log_message("Enhanced Coffee Sorter System started")
log_message(f"Lead time: {LEAD_TIME_MS}ms, Arming zone: {ARMING_ZONE_START_X_RATIO*100:.0f}% - {ARMING_ZONE_END_X_RATIO*100:.0f}%")
log_event("startup", lead_ms=LEAD_TIME_MS,
          zone_start=int(ARMING_ZONE_START_X_RATIO*100),
          zone_end=int(ARMING_ZONE_END_X_RATIO*100),
          conf=CONFIDENCE_THRESHOLD)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time_ms = int(time.time() * 1000)
        frame_height, frame_width = frame.shape[:2]
        arming_zone_start_pixel = int(ARMING_ZONE_START_X_RATIO * frame_width)
        arming_zone_end_pixel = int(ARMING_ZONE_END_X_RATIO * frame_width)

        # --- Detection + Tracking ---
        results = model.track(frame, persist=True, tracker="bytetrack.yaml",
                              conf=CONFIDENCE_THRESHOLD, verbose=False)

        annotated_frame = frame.copy()

        # ปลอดภัยก่อนเข้าถึง boxes/id
        if results and len(results) > 0 and getattr(results[0], "boxes", None) is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            object_ids = boxes.id.cpu().numpy().astype(int)
            class_indices = boxes.cls.cpu().numpy().astype(int)
            box_coords_list = boxes.xyxy.cpu().numpy()

            for i, obj_id in enumerate(object_ids):
                detected_class = class_names[int(class_indices[i])]
                box_coords = box_coords_list[i]
                center_x = float((box_coords[0] + box_coords[2]) / 2.0)

                # --- duplicate suppression ---
                duplicate_found = False
                existing_id_for_log = None
                for existing_id, existing_data in tracked_objects.items():
                    time_diff = abs(current_time_ms - existing_data['first_detection_time'])
                    pos_diff = abs(center_x - existing_data['center_x'])
                    same_class = (detected_class in existing_data['class_history'])
                    if (existing_id != obj_id and time_diff < 400 and pos_diff < 40 and same_class):
                        duplicate_found = True
                        existing_id_for_log = existing_id
                        break

                if duplicate_found:
                    log_message(f"DUPLICATE: Skipped ID {obj_id} as duplicate of {existing_id_for_log}")
                    log_event("duplicate", id=obj_id, dup_of=existing_id_for_log, cls=detected_class)
                    continue

                if obj_id not in tracked_objects:
                    tracked_objects[obj_id] = {
                        'class_history': deque(maxlen=CLASS_VOTING_HISTORY),
                        'last_seen': current_time_ms,
                        'frames_seen': 0,
                        'confirmed_class': None,
                        'is_counted': False,
                        'center_x': center_x,
                        'arm_command_sent': False,
                        'first_detection_time': current_time_ms,
                        'last_box': box_coords.tolist()  # NEW: เก็บกล่องล่าสุดเพื่อครอป
                    }

                tracked_obj = tracked_objects[obj_id]
                tracked_obj.update({
                    'last_seen': current_time_ms,
                    'frames_seen': tracked_obj['frames_seen'] + 1,
                    'center_x': center_x,
                    'last_box': box_coords.tolist()
                })
                tracked_obj['class_history'].append(detected_class)

                # --- confirm class ด้วย majority vote ---
                if (not tracked_obj['is_counted'] and tracked_obj['frames_seen'] >= FRAMES_TO_CONFIRM_STABILITY):
                    most_common_class = Counter(tracked_obj['class_history']).most_common(1)[0][0]
                    tracked_obj['confirmed_class'] = most_common_class

                    if most_common_class in GOOD_CLASSES:
                        good_seed_count += 1
                    elif most_common_class in BAD_CLASSES:
                        bad_seed_count += 1

                    # ตรงนี้จะถูก log ทั้งลงจอ, ลงไฟล์ และ events.csv
                    log_message(f"CLASSIFY: ID {obj_id} confirmed as '{most_common_class}'")
                    log_event("confirm", id=obj_id, cls=most_common_class,
                              x_center=int(center_x), frames_seen=tracked_obj['frames_seen'],
                              fw=frame_width, fh=frame_height)

                    # NEW: save crop on confirm
                    if SAVE_CROPS_ON_CONFIRM and tracked_obj.get('last_box') is not None:
                        _ = save_crop(frame, tracked_obj['last_box'], most_common_class, obj_id, bucket="confirmed")

                    tracked_obj['is_counted'] = True

        # --- Decision: ส่งคำสั่งดีด ---
        system_ready = (current_time_ms - last_command_time) > COMMAND_COOLDOWN_MS
        if results and len(results) > 0 and getattr(results[0], "boxes", None) is not None:
            if system_ready:
                for obj_id, data in tracked_objects.items():
                    confirmed_class = data.get('confirmed_class')
                    if confirmed_class is None:
                        continue

                    is_bad_seed = confirmed_class in BAD_CLASSES
                    is_in_arming_zone = (arming_zone_start_pixel < data['center_x'] < arming_zone_end_pixel)
                    arm_not_sent = not data.get('arm_command_sent', False)
                    time_since_detection = current_time_ms - data['first_detection_time']
                    should_send_early = time_since_detection >= LEAD_TIME_MS

                    if (is_bad_seed and is_in_arming_zone and arm_not_sent and should_send_early):
                        if send_arm_command(obj_id, confirmed_class):
                            data['arm_command_sent'] = True

                            # NEW: crop on eject + event log
                            if SAVE_CROPS_ON_EJECT and data.get('last_box') is not None:
                                crop_path = save_crop(frame, data['last_box'], confirmed_class, obj_id, bucket="ejected")
                            else:
                                crop_path = None

                            log_event("eject", id=obj_id, cls=str(confirmed_class),
                                      x_center=int(data['center_x']),
                                      in_zone=int(is_in_arming_zone),
                                      crop=str(crop_path))
                            break

        # --- วาดผลลัพธ์ ---
        if results and len(results) > 0 and getattr(results[0], "boxes", None):
            annotated_frame = results[0].plot()

        # cleanup object ที่ค้างนานเกิน timeout
        ids_to_remove = [oid for oid, data in tracked_objects.items()
                         if current_time_ms - data['last_seen'] > OBJECT_TIMEOUT_MS]
        for oid in ids_to_remove:
            log_message(f"CLEANUP: Removed stale ID {oid}")
            log_event("cleanup", id=oid)
            del tracked_objects[oid]

        # วาด arming zone
        cv2.line(annotated_frame, (arming_zone_start_pixel, 0),
                 (arming_zone_start_pixel, frame_height), (0, 255, 255), 2)
        cv2.line(annotated_frame, (arming_zone_end_pixel, 0),
                 (arming_zone_end_pixel, frame_height), (0, 255, 255), 2)
        cv2.putText(annotated_frame, "ARMING ZONE",
                    (arming_zone_start_pixel + 5, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # สถิติหลัก
        stats_text = f"GOOD: {good_seed_count} | BAD: {bad_seed_count} | EJECTED: {ejection_total_count}"
        cv2.putText(annotated_frame, stats_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(annotated_frame, stats_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # สถานะระบบ
        system_status = "READY" if system_ready else "COOLDOWN"
        status_color = (0, 255, 0) if system_ready else (0, 165, 255)
        cv2.putText(annotated_frame, f"System: {system_status}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"System: {system_status}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)

        # แสดง log ล่าสุดบนจอ
        log_y = 110
        for log_text in list(display_logs):
            cv2.putText(annotated_frame, log_text, (10, log_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, log_text, (10, log_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            log_y += 18

        # เซฟเฟรมเป็นระยะ
        maybe_save_frame(annotated_frame)

        # แสดงภาพ
        cv2.imshow("Enhanced Coffee Sorter", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            good_seed_count = 0
            bad_seed_count = 0
            ejection_total_count = 0
            log_message("Statistics reset")
            log_event("stats_reset")

finally:
    print("Cleaning up...")
    arduino_running = False
    time.sleep(0.1)
    if cap:
        cap.release()
    if 'ser' in globals() and ser and ser.is_open:
        ser.close()
    cv2.destroyAllWindows()
    print("System shutdown complete")
