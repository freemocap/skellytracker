import logging
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
from tqdm import tqdm

from skellytracker.io.process_videos.video_handler import VideoHandler
from skellytracker.trackers.mediapipe_tracker.__mediapipe_tracker import MediapipeRecorder, MediapipeTracker, \
    MediapipeTrackerConfig
from skellytracker.trackers.mediapipe_tracker.mediapipe_annotator import MediapipeAnnotatorConfig, \
    MediapipeImageAnnotator
from skellytracker.trackers.mediapipe_tracker.mediapipe_detector import MediapipeDetector, MediapipeDetectorConfig

if TYPE_CHECKING:
    from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseTracker

logger = logging.getLogger(__name__)

def process_video(
        tracker: 'BaseTracker',
        input_video_filepath: str,
        output_data_directory: str,
        output_video_filepath: str | None = None,
        use_tqdm: bool = True,
):
    """
    Run the tracker on a video.

    :param input_video_filepath: Path to video file.
    :param output_data_directory: Path to save the data to.
    :param output_video_filepath: Path to save annotated video to, does not save video if None.
    :param use_tqdm: Whether to use tqdm to show a progress bar
    """
    if tracker.recorder is None:
        logger.warning("Tracker does not have a recorder, data will not be saved.")

    cap = cv2.VideoCapture(str(input_video_filepath))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    image_size = (width, height)

    fps = cap.get(cv2.CAP_PROP_FPS)

    if output_video_filepath is not None:
        video_handler = VideoHandler(
            output_path=output_video_filepath, frame_size=image_size, fps=fps
        )
    else:
        video_handler = None

    ret, frame = cap.read()

    number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if use_tqdm:
        iterator = tqdm(
            range(number_of_frames),
            desc=f"processing video: {Path(input_video_filepath).name}",
            total=number_of_frames,
            colour="magenta",
            unit="frames",
            dynamic_ncols=True,
        )
    else:
        iterator = range(number_of_frames)

    for _frame_number in iterator:
        if not ret or frame is None:
            logger.error(
                f"Failed to load an image from: {str(input_video_filepath)}"
            )
            raise ValueError("Failed to load an image from: " + str(input_video_filepath))

        latest_observation = tracker.process_image(frame_number=_frame_number, image=frame, record_observation=True)

        if video_handler is not None:
            annotated_image = tracker.annotator.annotate_image(image=frame, latest_observation=latest_observation)
            video_handler.add_frame(annotated_image)

        ret, frame = cap.read()

    cap.release()
    if video_handler is not None:
        video_handler.close()

    if tracker.recorder is not None:
        file_name = Path(input_video_filepath).stem
        tracker.recorder.save_array(output_path=Path(output_data_directory) / f"{file_name}.npy")
        output_array = tracker.recorder.as_array
    return output_array


if __name__ == "__main__":
    video_path = Path("/Users/philipqueen/freemocap_data/recording_sessions/session_2024-12-10_14_34_36/recording_14_38_12_gmt-7/synchronized_videos/Camera_000_synchronized.mp4")
    process_video(
        tracker=MediapipeTracker(
            config=MediapipeTrackerConfig(),
            detector=MediapipeDetector.create(MediapipeDetectorConfig()),
            annotator=MediapipeImageAnnotator.create(MediapipeAnnotatorConfig()),
            recorder=MediapipeRecorder(),
        ),
        input_video_filepath=str(video_path),
        output_data_directory="/Users/philipqueen/freemocap_data/recording_sessions/session_2024-12-10_14_34_36/recording_14_38_12_gmt-7/synchronized_videos",
        output_video_filepath="/Users/philipqueen/freemocap_data/recording_sessions/session_2024-12-10_14_34_36/recording_14_38_12_gmt-7/synchronized_videos/annotated.mp4",
    )