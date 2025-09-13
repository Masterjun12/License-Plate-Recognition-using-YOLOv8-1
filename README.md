# Korean License Plate Recognition using YOLOv8

This project is an advanced implementation of license plate recognition, specifically tailored and trained for modern South Korean license plates. Built upon the powerful YOLOv8 object detection model, it features a custom-trained detector and an OCR-based validation system to ensure high accuracy in identifying and reading Korean license plates from video streams.

This system was developed by enhancing a general-purpose YOLOv8 license plate detection project with a focus on the unique characteristics and formats of Korean license plates.

## 🇰🇷 How It Works

The recognition process follows a multi-stage pipeline to ensure accuracy and robustness:

1.  **Vehicle and Plate Detection:** The system first processes a video file to detect vehicles and license plates in each frame using a custom-trained **YOLOv8 model**.
2.  **Object Tracking:** A SORT-based tracking algorithm is employed to track detected vehicles and plates across frames, assigning a unique ID to each object. This allows for data interpolation even if detection fails in some frames.
3.  **OCR Text Extraction:** For each detected license plate, **EasyOCR** is used to extract the characters. This library was chosen for its strong performance with non-Latin characters, including Hangul.
4.  **Format Validation:** A crucial step is the validation of the extracted text. The OCR output is checked against a set of predefined regular expressions that match valid South Korean license plate formats (e.g., `##가####`, `###더####`, `##서##`). A recognition is considered successful only if the text conforms to one of these patterns.
5.  **Data Processing & Visualization:** The raw detection data is saved, and missing detections for tracked objects are interpolated to create a smooth dataset. Finally, a new video is generated, visualizing the bounding boxes and recognized plate numbers.

## ✨ Key Features

-   **High-Accuracy Korean Plate Detection:** Utilizes a fine-tuned YOLOv8 model trained on a custom, manually-collected dataset of Korean vehicles.
-   **Custom Dataset:** The model was trained in **Roboflow** on a diverse dataset of Korean car images, ensuring robustness for various plate types, lighting conditions, and angles.
-   **OCR-Based Validation:** Significantly reduces false positives by validating the extracted text against known Korean license plate syntaxes.
-   **End-to-End Pipeline:** Provides a full pipeline from video input to a final, annotated video output.
-   **Robust Tracking:** Implements object tracking to maintain object identity across frames and handle temporary detection failures.

## 🛠️ Core Technologies

-   **Object Detection:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
-   **OCR:** [EasyOCR](https://github.com/JaidedAI/EasyOCR)
-   **Data Handling:** Pandas, NumPy
-   **Image/Video Processing:** OpenCV
-   **Tracking:** FilterPy (for Kalman Filter implementation in SORT)

## 🚀 Getting Started

### Prerequisites

-   Python 3.8 or higher
-   Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/License-Plate-Recognition-using-YOLOv8-1.git
    cd License-Plate-Recognition-using-YOLOv8-1
    ```

2.  **Create and activate a virtual environment:**
    -   On Windows:
        ```bash
        python -m venv venv
        venv\Scripts\activate
        ```
    -   On macOS/Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1.  **Configure the paths in `main.py`:**
    Open the `main.py` file and set the `video_path` and `output_folder` variables.

    ```python
    # --- Configuration ---
    # PLEASE FILL IN THESE PATHS
    video_path = "videos/video.mp4"  # Example: 'videos/your_video.mp4'
    output_folder = "results" # Example: 'output_data'
    # --- End of Configuration ---
    ```

2.  **Run the main script:**
    Execute the `main.py` script from your terminal to start the detection, processing, and visualization pipeline.

    ```bash
    python main.py
    ```

3.  **Check the results:**
    Once the script finishes, the `output_folder` you specified will contain:
    -   `raw_results.csv`: Raw detection data for every frame.
    -   `processed_results.csv`: Data after interpolation and processing.
    -   `output_video.mp4`: The final video with bounding boxes and recognized license plates.
    -   `detected_vehicles/` and `detected_plates/`: Cropped images of detected objects.

![img car](https://github.com/Masterjun12/License-Plate-Recognition-using-YOLOv8-1/blob/7becdf20af7b2fa9bacb9e3b733243876b1e096e/run/car.png)
![img log](https://github.com/Masterjun12/License-Plate-Recognition-using-YOLOv8-1/blob/7becdf20af7b2fa9bacb9e3b733243876b1e096e/run/log.png)


[![Demo Video](images/thumbnail.png)](https://drive.google.com/file/d/1knH4YN_X3zQmFO83JMMa7qN_JHVkbXkb/view?usp=drive_link)


## 📂 Project Structure

```
.
├── main.py                 # Main script to run the entire pipeline
├── detector.py             # Handles object detection and tracking
├── data_processor.py       # Cleans and interpolates detection data
├── visualizer.py           # Generates the final annotated video
├── util.py                 # Utility functions
├── requirements.txt        # Project dependencies
├── license_plate_detector.pt # Custom-trained YOLOv8 model for license plates
├── yolov8n_best.pt         # Pre-trained YOLOv8 model (likely for vehicles)
├── videos/                 # Directory for input videos
└── results/                # Default directory for output files
```
