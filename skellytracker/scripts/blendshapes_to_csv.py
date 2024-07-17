import argparse
from pathlib import Path
import pandas as pd

from skellytracker.trackers.mediapipe_blendshape_tracker.mediapipe_blendshape_tracker import MediapipeBlendshapeTracker

def main(input_video_filepath: str, output_video_filepath: str, output_csv_filepath: str):
    tracker = MediapipeBlendshapeTracker()
    output_array = tracker.process_video(input_video_filepath=Path(input_video_filepath), Path(output_video_filepath), save_data_bool=True)

    # create a dataframe consistent with livelink blendshape csv data
    df = pd.DataFrame(output_array)

    # name columns based on blendshape names
    df.columns = tracker.model_info.landmark_names

    # remove _neutral column
    df = df.drop(columns=["_neutral"])

    # add BlendShapeCount count as first column
    df.insert(0, "BlendShapeCount", tracker.model_info.num_tracked_points - 1)

    # TODO: eventually add timestamp column as first column, mocking timestamps like 00:00:00:01.001, 00:00:00:02.002, ..., 00:00:00:30.030, 00:00:01:00.031, ...

    df.to_csv(Path(output_csv_filepath), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video file and save the output video and CSV data.")
    
    parser.add_argument("input_video_filepath", type=str, help="Path to the input video file.")
    parser.add_argument("output_video_filepath", type=str, help="Path to save the output video file to.")
    parser.add_argument("output_csv_filepath", type=str, help="Path to savethe output CSV file to.")
    
    args = parser.parse_args()
    
    main(args.input_video_filepath, args.output_video_filepath, args.output_csv_filepath)