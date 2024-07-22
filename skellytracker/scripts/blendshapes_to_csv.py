import argparse
from pathlib import Path
from typing import Generator, Union
import pandas as pd

from skellytracker.trackers.mediapipe_blendshape_tracker.mediapipe_blendshape_tracker import (
    MediapipeBlendshapeTracker,
)


def create_timestamp_generator() -> Generator[str, None, None]:
    """
    Create a generator that generates timestamps in the format "00:00:00:00.000", with leading values rolling over before 30
    """
    count = 1
    refresh_value = 30
    while True:
        parts = [(count // (refresh_value**i)) % refresh_value for i in range(4)]
        yield f"{parts[3]:02d}:{parts[2]:02d}:{parts[1]:02d}:{parts[0]:02d}.{count % 1000:03d}"
        count += 1


def main(
    input_video_filepath: Union[str, Path],
    output_video_filepath: Union[str, Path],
    output_csv_filepath: Union[str, Path],
):
    tracker = MediapipeBlendshapeTracker()
    output_array = tracker.process_video(
        input_video_filepath=Path(input_video_filepath),
        output_video_filepath=Path(output_video_filepath),
        save_data_bool=True,
    )
    if output_array is None:
        raise ValueError("output_array is None, aborting")

    output_array = output_array.reshape(
        output_array.shape[0], -1
    )  # ensure output array is 2D

    df = pd.DataFrame(
        output_array,
        columns=tracker.model_info.landmark_names,
    ).drop(
        columns=["_neutral"]
    )

    timestamp_generator = create_timestamp_generator()
    df.insert(0, "Timestamp", [next(timestamp_generator) for _ in range(len(df))])

    df.insert(
        1, "BlendShapeCount", tracker.model_info.num_tracked_points
    )

    df.to_csv(Path(output_csv_filepath), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a video file and save the output video and CSV data."
    )

    parser.add_argument(
        "-i", "--input", type=Path, help="Path to the input video file."
    )
    parser.add_argument(
        "-o", "--output-video", type=Path, help="Path to save the output video file to."
    )
    parser.add_argument(
        "-c", "--output-csv", type=Path, help="Path to save the output CSV file to."
    )

    args = parser.parse_args()

    input_path = args.input or Path(input("Enter the path to the video to be processed: "))
    output_video_path = args.output_video or Path(input("Enter the path to save output video to: "))
    output_csv_path = args.output_csv or Path(input("Enter the path to save the output CSV file to: "))

    main(
        input_path, output_video_path, output_csv_path
    )
