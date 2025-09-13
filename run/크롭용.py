import os
from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import *
from util import get_car, read_license_plate, write_csv

results = {}

mot_tracker = Sort()

# Load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')

# Create folder for saving cropped vehicle images
output_folder = './cropped_vehicles'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load video
cap = cv2.VideoCapture('./demo5.mp4')

vehicles = [2, 3, 5, 7]

# Read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        # 원본 프레임 복사 (크롭 이미지 저장에 사용)
        original_frame = frame.copy()
        
        results[frame_nmr] = {}
        # Detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

                # Crop the vehicle image from the original frame
                vehicle_crop = original_frame[int(y1):int(y2), int(x1):int(x2)]

                # Save the cropped vehicle image
                vehicle_image_path = os.path.join(
                    output_folder, f"frame_{frame_nmr:04d}_vehicle_{int(class_id)}_{int(score*100)}.jpg"
                )
                cv2.imwrite(vehicle_image_path, vehicle_crop)

        # Track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # Detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:
                # Crop license plate
                license_plate_crop = original_frame[int(y1):int(y2), int(x1): int(x2), :]

                # Process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # Read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {
                        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2],
                            'text': license_plate_text,
                            'bbox_score': score,
                            'text_score': license_plate_text_score
                        }
                    }

        # Show the frame with bounding boxes for debugging (optional)
        cv2.imshow('Vehicle and License Plate Detection', frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release video and destroy windows
cap.release()
cv2.destroyAllWindows()

# Write results
write_csv(results, './test.csv')
