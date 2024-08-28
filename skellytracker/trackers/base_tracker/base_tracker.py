from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np
from tqdm import tqdm


from skellytracker.trackers.base_tracker.base_recorder import BaseCumulativeRecorder, BaseRecorder
from skellytracker.trackers.base_tracker.tracked_object import TrackedObject
from skellytracker.trackers.base_tracker.video_handler import VideoHandler
from skellytracker.trackers.demo_viewers.image_demo_viewer import ImageDemoViewer
from skellytracker.trackers.demo_viewers.webcam_demo_viewer import (
    WebcamDemoViewer,
)

logger = logging.getLogger(__name__)


class BaseTracker(ABC):
    """
    An abstract base class for implementing different tracking algorithms.
    """

    def __init__(
        self,
        recorder: BaseRecorder,
        tracked_object_names: List[str],
        **data: Any,
    ):
        self.recorder = recorder
        self.annotated_image = None
        self.tracked_objects: Dict[str, TrackedObject] = {}

        for name in tracked_object_names:
            self.tracked_objects[name] = TrackedObject(object_id=name)

    @abstractmethod
    def process_image(self, image: np.ndarray, **kwargs) -> Dict[str, TrackedObject]:
        """
        Process the input image and apply the tracking algorithm.

        :param image: An input image.
        :return: A dictionary of tracked objects
        """
        pass

    @abstractmethod
    def annotate_image(
        self, image: np.ndarray, tracked_objects: Dict[str, TrackedObject], **kwargs
    ) -> np.ndarray:
        """
        Annotate the input image with the results of the tracking algorithm.

        :param image: An input image.
        :param tracked_objects: A dictionary of tracked objects.
        :return: Annotated image
        """
        pass

    def process_video(
        self,
        input_video_filepath: Union[str, Path],
        output_video_filepath: Optional[Union[str, Path]] = None,
        save_data_bool: bool = False,
        use_tqdm: bool = True,
    ) -> Union[np.ndarray, None]:
        """
        Run the tracker on a video.

        :param input_video_filepath: Path to video file.
        :param output_video_filepath: Path to save annotated video to, does not save video if None.
        :param save_data_bool: Whether to save the data to a file.
        :param use_tqdm: Whether to use tqdm to show a progress bar
        :return: Array of tracked keypoint data if tracker has an associated recorder
        """

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

            self.process_image(frame)
            if self.recorder is not None:
                self.recorder.record(self.tracked_objects)
            if video_handler is not None:
                if self.annotated_image is None:
                    self.annotated_image = frame
                video_handler.add_frame(self.annotated_image)

            ret, frame = cap.read()

        cap.release()
        if video_handler is not None:
            video_handler.close()

        if self.recorder is not None:
            output_array = self.recorder.process_tracked_objects(image_size=image_size)
            if save_data_bool:
                self.recorder.save(
                    file_path=str(Path(input_video_filepath).with_suffix(".npy"))
                )
        else:
            output_array = None

        return output_array

    def demo(self) -> None:
        """
        Run a demo of the tracker.

        :param window_title: The title of the demo window.
        :return: None
        """
        camera_viewer = WebcamDemoViewer(
            tracker=self, recorder=self.recorder, window_title=self.__class__.__name__
        )
        camera_viewer.run()

    def image_demo(self, image_path: Path) -> None:
        """
        Run tracker on single image

        :return: None
        """

        image_viewer = ImageDemoViewer(self, self.__class__.__name__)
        image_viewer.run(image_path=image_path)


class BaseCumulativeTracker(BaseTracker):
    """
    A base class for tracking algorithms that run cumulatively, i.e are not able to process videos frame by frame.
    Throws a descriptive error for the abstract methods of BaseTracker that do not apply to this type of tracker.
    Trackers inheriting from this will need to overwrite the `process_video` method.
    """

    def __init__(
        self,
        tracked_object_names: List[str],
        recorder: BaseCumulativeRecorder,
        **data: Any,
    ):
        super().__init__(
            tracked_object_names=tracked_object_names, recorder=recorder, **data
        )

    def process_image(self, **kwargs) -> None:
        raise NotImplementedError(
            "This tracker does not support processing individual images, please use process_video instead."
        )

    def annotate_image(self, **kwargs) -> None:
        raise NotImplementedError(
            "This tracker does not support processing individual images, please use process_video instead."
        )

    @abstractmethod
    def process_video(
        self,
        input_video_filepath: Union[str, Path],
        output_video_filepath: Optional[Union[str, Path]] = None,
        save_data_bool: bool = False,
        use_tqdm: bool = True,
        **kwargs,
    ) -> Union[np.ndarray, None]:
        """
        Run the tracker on a video.

        :param input_video_filepath: Path to video file.
        :param output_video_filepath: Path to save annotated video to, does not save video if None.
        :param save_data_bool: Whether to save the data to a file.
        :param use_tqdm: Whether to use tqdm to show a progress bar
        :return: Array of tracked keypoint data
        """
        pass

    def image_demo(self, image_path: Path) -> None:
        raise NotImplementedError(
            "This tracker does not support processing individual images, please use process_video instead."
        )
