# Automatic Number Plate Recognition (ANPR)

This project is an **Automatic Number Plate Recognition system** built with Python, OpenCV, and YOLO models.  
It detects vehicles in a video, locates their license plates using a custom-trained YOLO model, and finally extracts the plate text using an OCR specialized for license plates.

---

## 🔍 How it works
1. **Vehicle Detection** – Cars are detected using the pre-trained YOLOv8 model (`yolov8n.pt`).
2. **License Plate Detection** – The detected car region is passed to another YOLO model (`license_plate_detector.pt`) that finds the license plate inside the car.
3. **OCR (Optical Character Recognition)** – The cropped license plate is processed by `fast_plate_ocr` to recognize and extract the plate number.
4. The detected plate image and its text are displayed on the video in real time.

---

## ⚙️ Requirements
Install the required dependencies before running the script:

```bash
pip install ultralytics opencv-python fast_plate_ocr
