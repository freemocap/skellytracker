import argparse
from pathlib import Path
from typing import Generator
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


def blendshapes_to_csv(
    input_video_filepath: Path,
    output_video_filepath: Path,
    output_csv_filepath: Path,
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
    ).drop(columns=["_neutral"])

    timestamp_generator = create_timestamp_generator()
    df.insert(0, "Timestamp", [next(timestamp_generator) for _ in range(len(df))])

    df.insert(1, "BlendShapeCount", tracker.model_info.num_tracked_points)

    df.to_csv(Path(output_csv_filepath), index=False)


def main():
    parser = argparse.ArgumentParser(
        prog="skellytracker_blendshapes",
        description="Process a video file to extract MediaPipe blendshapes and save the output video and CSV data."
        epilog="Thank you for using skellytracker!",
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

    output_video_path = args.output_video or Path(input(
        "Enter the path to save output video to: "
    ))

    output_csv_path = args.output_csv or Path(input(
        "Enter the path to save the output CSV file to: "
    ))

    if not input_path.exists():
        raise ValueError(f"Input video path does not exist: {input_path}")

    if output_video_path.suffix != ".mp4":
        print("Output video path must be a .mp4 file, changing extension to .mp4")
        output_video_path = output_video_path.with_suffix(".mp4")

    if output_csv_path.suffix != ".csv":
        print("Output CSV path must be a .csv file, changing extension to .csv")
        output_csv_path = output_csv_path.with_suffix(".csv")

    blendshapes_to_csv(input_path, output_video_path, output_csv_path)


if __name__ == "__main__":
    main()
