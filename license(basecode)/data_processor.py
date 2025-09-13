
import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
import ast

def interpolate_bounding_boxes(data):
    """Interpolates missing bounding boxes for a given car ID."""
    
    # Ensure data is sorted by frame number
    data = data.sort_values(by='frame_nmr')
    
    # Convert string representation of list to actual list
    data['license_plate_bbox'] = data['license_plate_bbox'].apply(ast.literal_eval)
    
    # Filter out rows where bounding box is missing
    known_data = data[data['license_plate_bbox'].apply(lambda x: len(x) == 4)]
    
    if len(known_data) < 2:
        # Not enough data to interpolate
        return data

    # Frame numbers for which we have bounding boxes
    known_frames = known_data['frame_nmr'].values
    
    # Bounding box coordinates
    bboxes = np.array(known_data['license_plate_bbox'].tolist())
    
    # Create interpolation functions for each coordinate
    interp_funcs = [interp1d(known_frames, bboxes[:, i], kind='linear', fill_value="extrapolate") for i in range(4)]
    
    # All frame numbers for this car
    all_frames = data['frame_nmr'].values
    
    # Interpolate for all frames
    interpolated_bboxes = [
        [f(frame) for f in interp_funcs] for frame in all_frames
    ]
    
    # Update the dataframe with interpolated values
    data['license_plate_bbox'] = interpolated_bboxes
    
    return data

def process_missing_data(input_csv_path, output_csv_path):
    """
    Reads detection results, interpolates missing data, and saves the result.
    """
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_csv_path}")
        return

    # Group by car_id and apply interpolation
    interpolated_results = df.groupby('car_id').apply(interpolate_bounding_boxes)
    
    # Drop the extra index created by groupby
    interpolated_results = interpolated_results.reset_index(drop=True)
    
    # Save the processed data
    interpolated_results.to_csv(output_csv_path, index=False)
    print(f"Processed data saved to {output_csv_path}")

