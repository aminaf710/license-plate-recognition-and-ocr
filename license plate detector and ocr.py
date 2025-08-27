import cv2
import time
from ultralytics import YOLO
from fast_plate_ocr import LicensePlateRecognizer


car_model = YOLO("yolov8n.pt")  
plate_model = YOLO("license_plate_detector.pt")  
reader = LicensePlateRecognizer('cct-xs-v1-global-model')


line_y = 550

last_plate = None
last_plate_time = 0
show_duration = 2  

cap = cv2.VideoCapture("traffic.mp4")#you should upload your video direction and name 

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % 1 != 0:
        continue



    cars = car_model(frame)[0].boxes
    for car in cars:
        x1, y1, x2, y2 = map(int, car.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if cy > line_y - 5 and cy < line_y + 5:
            roi = frame[y1:y2, x1:x2]
            plates = plate_model(roi)[0].boxes

            for plate in plates:
                px1, py1, px2, py2 = map(int, plate.xyxy[0])
                plate_crop = roi[py1:py2, px1:px2]
                plate_crop_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
                result = reader.run(plate_crop_rgb)
                plate_text = str(result)
                last_plate = (plate_text, plate_crop)
                last_plate_time = time.time()

    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 0), 2)

    if last_plate and time.time() - last_plate_time < show_duration:
        plate_text, plate_crop = last_plate


        plate_crop = cv2.cvtColor(plate_crop_rgb, cv2.COLOR_RGB2BGR)
        plate_crop = cv2.resize(plate_crop, (200, 80))
        frame[20:100, 20:220] = plate_crop

        cv2.putText(frame, plate_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("ANPR", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()

cv2.destroyAllWindows()
