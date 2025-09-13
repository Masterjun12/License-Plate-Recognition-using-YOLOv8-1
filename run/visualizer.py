
import cv2
import numpy as np
import pandas as pd
import ast

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)
    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img

def generate_video(input_csv_path, video_path, output_video_path):
    """
    Generates a video with license plate detections visualized.
    """
    try:
        results = pd.read_csv(input_csv_path)
        print("CSV file loaded successfully.")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Could not open video.")
        print(f"Video {video_path} loaded successfully.")
    except Exception as e:
        print(f"Error loading video: {e}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    license_plate = {}
    for car_id in np.unique(results['car_id']):
        car_results = results[results['car_id'] == car_id]
        for _, row in car_results.iterrows():
            try:
                x1, y1, x2, y2 = ast.literal_eval(row['license_plate_bbox'])
                license_plate_number = row['license_number']
                cap.set(cv2.CAP_PROP_POS_FRAMES, row['frame_nmr'])
                ret, frame = cap.read()
                if ret:
                    license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                    license_crop = cv2.resize(license_crop, (200, 100))
                    license_plate[car_id] = {'license_crop': license_crop, 'license_plate_number': license_plate_number}
                else:
                    print(f"Error reading frame {row['frame_nmr']} for car_id {car_id}.")
            except Exception as e:
                print(f"Error processing car_id {car_id}: {e}")

    frame_nmr = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"End of video or error at frame {frame_nmr}.")
            break

        df_ = results[results['frame_nmr'] == frame_nmr]
        for _, row in df_.iterrows():
            try:
                x1, y1, x2, y2 = ast.literal_eval(row['license_plate_bbox'])
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

                license_crop = license_plate[row['car_id']]['license_crop']
                license_plate_number = license_plate[row['car_id']]['license_plate_number']

                H, W, _ = license_crop.shape
                try:
                    frame[int(y1) - H - 100:int(y1) - 100, int((x2 + x1 - W) / 2):int((x2 + x1 + W) / 2)] = license_crop
                    frame[int(y1) - H - 400:int(y1) - H - 100, int((x2 + x1 - W) / 2):int((x2 + x1 + W) / 2)] = (255, 255, 255)
                    (text_width, text_height), _ = cv2.getTextSize(license_plate_number, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                    cv2.putText(frame, license_plate_number,
                                (int((x2 + x1 - text_width) / 2), int(y1 - H - 250 + (text_height / 2))),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                except Exception as e:
                    pass
            except Exception as e:
                pass

        out.write(frame)
        frame_nmr += 1

    out.release()
    cap.release()
    print(f"Processing complete. Video saved to {output_video_path}.")
