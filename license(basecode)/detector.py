
import os
from ultralytics import YOLO
import cv2
import numpy as np
from util import get_car, read_license_plate, write_csv
from sort.sort import Sort
from correct_license_plate import correct_perspective, preprocess_license_plate

def detect_and_track(video_path, output_csv_path, vehicle_output_folder, plate_output_folder):
    """
    Performs vehicle and license plate detection and tracking on a video.
    """
    results = {}
    mot_tracker = Sort()

    coco_model = YOLO('yolov8n.pt')
    license_plate_detector = YOLO('license_plate_detector.pt')

    if not os.path.exists(vehicle_output_folder):
        os.makedirs(vehicle_output_folder)

    if not os.path.exists(plate_output_folder):
        os.makedirs(plate_output_folder)

    cap = cv2.VideoCapture(video_path)
    vehicles = [2, 3, 5, 7]

    frame_nmr = -1
    ret = True
    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        if ret:
            results[frame_nmr] = {}
            detections = coco_model(frame)[0]
            detections_ = []
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in vehicles:
                    detections_.append([x1, y1, x2, y2, score])
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.putText(frame, f"Vehicle: {int(class_id)}", (int(x1), int(y1) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    vehicle_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    vehicle_image_path = os.path.join(vehicle_output_folder, f"frame_{frame_nmr:04d}_vehicle_{int(class_id)}_{int(score*100)}.jpg")
                    cv2.imwrite(vehicle_image_path, vehicle_crop)

            try:
                track_ids = mot_tracker.update(np.asarray(detections_))
            except Exception as e:
                print(f"Error during vehicle tracking: {e}")
                track_ids = []

            license_plates = license_plate_detector(frame)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                if car_id != -1:
                    license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop)

                    plate_image_path = os.path.join(plate_output_folder, f"frame_{frame_nmr:04d}_plate_{car_id}_{int(score*100)}.jpg")
                    cv2.imwrite(plate_image_path, license_plate_crop)

                    results[frame_nmr][car_id] = {
                        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2],
                            'text': license_plate_text if license_plate_text else 'Unknown',
                            'bbox_score': score,
                            'text_score': license_plate_text_score if license_plate_text else 0
                        }
                    }

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"LP: {license_plate_text if license_plate_text else 'Unknown'}", 
                                (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow('Vehicle and License Plate Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    write_csv(results, output_csv_path)
