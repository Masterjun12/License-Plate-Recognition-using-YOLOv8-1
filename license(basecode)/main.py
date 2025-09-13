import os
from detector import detect_and_track
from data_processor import process_missing_data
from visualizer import generate_video

def main():
    # --- Configuration ---
    video_path = os.path.join('videos', 'video.mp4')
    output_folder = 'results'
    # --- End of Configuration ---

    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        print("Please download the video and place it in the 'videos' folder.")
        return

    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Define output file paths
    raw_csv_path = os.path.join(output_folder, 'raw_results.csv')
    processed_csv_path = os.path.join(output_folder, 'processed_results.csv')
    output_video_path = os.path.join(output_folder, 'output_video.mp4')
    vehicle_output_folder = os.path.join(output_folder, 'detected_vehicles')
    plate_output_folder = os.path.join(output_folder, 'detected_plates')

    # 1. Run detection and tracking
    print("Step 1: Running detection and tracking...")
    detect_and_track(video_path, raw_csv_path, vehicle_output_folder, plate_output_folder)
    print("Detection and tracking complete.")

    # 2. Process missing data
    print("\nStep 2: Processing and interpolating data...")
    process_missing_data(raw_csv_path, processed_csv_path)
    print("Data processing complete.")

    # 3. Generate visualized video
    print("\nStep 3: Generating visualized video...")
    generate_video(processed_csv_path, video_path, output_video_path)
    print("Visualization complete.")

if __name__ == "__main__":
    main()
