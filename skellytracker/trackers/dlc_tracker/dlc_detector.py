from __future__ import annotations
from pathlib import Path
import cv2
import torch.multiprocessing as mp
import albumentations as A
import numpy as np
import time

from deeplabcut.compat import _update_device
from deeplabcut.pose_estimation_pytorch.apis.videos import (
    VideoIterator,
    _generate_metadata,
    video_inference,
)
import deeplabcut.pose_estimation_pytorch.apis.utils as utils
from deeplabcut.pose_estimation_pytorch.apis.videos import (
    _validate_destfolder,
)
import deeplabcut.pose_estimation_pytorch.runners.shelving as shelving
from deeplabcut.core.engine import Engine
from deeplabcut.pose_estimation_pytorch.runners import DynamicCropper
from deeplabcut.pose_estimation_pytorch.task import Task
from deeplabcut.utils import auxiliaryfunctions

from skellytracker.trackers.base_tracker.base_tracker_abcs import BaseDetectorConfig, BaseDetector
from skellytracker.trackers.dlc_tracker.dlc_observation import DeepLabCutObservation

class DeepLabCutDetectorConfig(BaseDetectorConfig):
    dlc_config: str
    videotype: str = ""
    shuffle: int = 1
    trainingsetindex: int = 0
    gputouse: int | None = None
    save_as_csv: bool = False
    destfolder: str | None = None
    cropping: list[int] | None = None
    dynamic: tuple[bool, float, int] = (False, 0.5, 10)
    modelprefix: str = ""
    robust_nframes: bool = False
    use_shelve: bool = False
    auto_track: bool = True
    n_tracks: int | None = None
    animal_names: list[str] | None = None
    identity_only: bool = False
    snapshot_index: int | str | None = None
    detector_snapshot_index: int | str | None = None
    device: str | None = None
    batch_size: int | None = None
    detector_batch_size: int | None = None
    transform: A.Compose | None = None
    overwrite: bool = False
    save_as_df: bool = False


class DeepLabCutDetector(BaseDetector):
    config: DeepLabCutDetectorConfig

    @classmethod
    def create(cls, config: DeepLabCutDetectorConfig):
        return cls(
            config=config,
        )

    def detect(self, frame_number: int, image: np.ndarray) -> DeepLabCutObservation:
        raise NotImplementedError(
            "This detector does not support processing individual images, please use detect_video instead."
        )
    
    # TODO: get point names from dlc config

    def detect_video(
        self,
        video_path: str | Path,
        **torch_kwargs,
    ) -> list[DeepLabCutObservation]:
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass

        _update_device(self.config.gputouse, torch_kwargs)

        video = Path(video_path)

        # Create the output folder
        _validate_destfolder(self.config.destfolder)

        # Load the project configuration
        cfg = auxiliaryfunctions.read_config(self.config.dlc_config)
        project_path = Path(cfg["project_path"])
        train_fraction = cfg["TrainingFraction"][self.config.trainingsetindex]
        model_folder = project_path / auxiliaryfunctions.get_model_folder(
            train_fraction,
            self.config.shuffle,
            cfg,
            modelprefix=self.config.modelprefix,
            engine=Engine.PYTORCH,
        )
        train_folder = model_folder / "train"

        # Read the inference configuration, load the model
        model_cfg_path = train_folder / Engine.PYTORCH.pose_cfg_name
        model_cfg = auxiliaryfunctions.read_plainconfig(model_cfg_path)
        pose_task = Task(model_cfg["method"])

        pose_cfg_path = model_folder / "test" / "pose_cfg.yaml"
        pose_cfg = auxiliaryfunctions.read_plainconfig(pose_cfg_path)

        snapshot_index, detector_snapshot_index = utils.parse_snapshot_index_for_analysis(
            cfg,
            model_cfg,
            self.config.snapshot_index,
            self.config.detector_snapshot_index,
        )

        if self.config.cropping is None and cfg.get("cropping", False):
            cropping = [cfg["x1"], cfg["x2"], cfg["y1"], cfg["y2"]]

        # Get general project parameters
        multi_animal = cfg["multianimalproject"]
        bodyparts = model_cfg["metadata"]["bodyparts"]
        unique_bodyparts = model_cfg["metadata"]["unique_bodyparts"]
        individuals = model_cfg["metadata"]["individuals"]
        max_num_animals = len(individuals)

        if self.config.device is not None:
            model_cfg["device"] = self.config.device

        if self.config.batch_size is None:
            batch_size = cfg.get("batch_size", 1)

        if not multi_animal:
            save_as_df = True
            if self.config.use_shelve:
                print(
                    "The ``use_shelve`` parameter cannot be used for single animal "
                    "projects. Setting ``use_shelve=False``."
                )
                use_shelve = False

        dynamic = DynamicCropper.build(*self.config.dynamic)
        if pose_task != Task.BOTTOM_UP and dynamic is not None:
            print(
                "Turning off dynamic cropping. It should only be used for bottom-up "
                f"pose estimation models, but you are using a top-down model."
            )
            dynamic = None

        snapshot = utils.get_model_snapshots(snapshot_index, train_folder, pose_task)[0]
        print(f"Analyzing videos with {snapshot.path}")
        pose_runner = utils.get_pose_inference_runner(
            model_config=model_cfg,
            snapshot_path=snapshot.path,
            max_individuals=max_num_animals,
            batch_size=batch_size,
            transform=self.config.transform,
            dynamic=dynamic,
        )
        detector_runner = None

        detector_path, detector_snapshot = None, None
        if pose_task == Task.TOP_DOWN:
            if detector_snapshot_index is None:
                raise ValueError(
                    "Cannot run videos analysis for top-down models without a detector "
                    "snapshot! Please specify your desired detector_snapshotindex in your "
                    "project's configuration file."
                )

            if self.config.detector_batch_size is None:
                detector_batch_size = cfg.get("detector_batch_size", 1)

            detector_snapshot = utils.get_model_snapshots(
                detector_snapshot_index, train_folder, Task.DETECT
            )[0]
            print(f"  -> Using detector {detector_snapshot.path}")
            detector_runner = utils.get_detector_inference_runner(
                model_config=model_cfg,
                snapshot_path=detector_snapshot.path,
                max_individuals=max_num_animals,
                batch_size=detector_batch_size,
            )

        dlc_scorer = utils.get_scorer_name(
            cfg,
            self.config.shuffle,
            train_fraction,
            snapshot_uid=utils.get_scorer_uid(snapshot, detector_snapshot),
            modelprefix=self.config.modelprefix,
        )
        if self.config.destfolder is None:
            output_path = video.parent
        else:
            output_path = Path(self.config.destfolder)

        output_prefix = video.stem + dlc_scorer
        output_pkl = output_path / f"{output_prefix}_full.pickle"

        video_iterator = VideoIterator(video, cropping=cropping)

        image_size = video_iterator.video.get(cv2.CAP_PROP_FRAME_WIDTH), video_iterator.video.get(cv2.CAP_PROP_FRAME_HEIGHT)

        shelf_writer = None
        if use_shelve:
            shelf_writer = shelving.ShelfWriter(
                pose_cfg=pose_cfg,
                filepath=output_pkl,
                num_frames=video_iterator.get_n_frames(robust=self.config.robust_nframes),
            )

        runtime = [time.time()]
        predictions = video_inference(
            video=video_iterator,
            pose_runner=pose_runner,
            detector_runner=detector_runner,
            shelf_writer=shelf_writer,
            robust_nframes=self.config.robust_nframes,
        )
        print(f"PREDICTIONS INFO")
        print(f"\t{len(predictions)} frames")
        print(f"\tkeys: {predictions[0].keys()}")
        print(f"\tvalues shape: {[v.shape for v in predictions[0].values()]}")
        
        runtime.append(time.time())
        metadata = _generate_metadata(
            cfg=cfg,
            pytorch_config=model_cfg,
            dlc_scorer=dlc_scorer,
            train_fraction=train_fraction,
            batch_size=batch_size,
            cropping=cropping,
            runtime=(runtime[0], runtime[1]),
            video=video_iterator,
            robust_nframes=self.config.robust_nframes,
        )

        # TODO: how to save out metadata?
        return [
            DeepLabCutObservation.from_detection_results(frame_number=i, pose_prediction=prediction, image_size=image_size) 
            for i, prediction in enumerate(predictions)
        ]


if __name__=="__main__":
    config = DeepLabCutDetectorConfig(dlc_config="")