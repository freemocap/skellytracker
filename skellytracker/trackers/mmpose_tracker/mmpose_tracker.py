# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
import traceback
from pathlib import Path

import numpy as np
import torch
logger = logging.getLogger(__name__)
for _ in range(2):
    try:
        from mmdet.apis import inference_detector, init_detector
        from mmpose.apis import inference_topdown
        from mmpose.apis import init_model as init_pose_estimator
        from mmpose.evaluation.functional import nms
        from mmpose.registry import VISUALIZERS
        from mmpose.structures import merge_data_samples, PoseDataSample
        from mmpose.utils import adapt_mmdet_pipeline
        break
    except (ImportError, ModuleNotFoundError) as e:
        traceback.print_exc()
        logging.info("MMPose is not installed. Installing via `mim` subprocess calls (per https://mmpose.readthedocs.io/en/latest/installation.html ) ...")
        # # run mim install commands in subprocess
        # mim install mmengine
        # mim install "mmcv>=2.0.1"
        # mim install "mmdet>=3.1.0"
        # mim install "mmpose>=1.1.0"
        logger.info("running `mim install mmengine`")
        os.system("mim install mmengine")
        logger.info("`mimengine` installed successfully")

        logger.info("running `mim install \"mmcv>=2.0.1\"`")
        os.system("mim install \"mmcv>=2.0.1\"")
        logger.info("`mmcv` installed successfully")

        logger.info("running `mim install \"mmdet>=3.1.0\"`")
        os.system("mim install \"mmdet>=3.1.0\"")
        logger.info("`mmdet` installed successfully")

        logger.info("running `mim install \"mmpose>=1.1.0\"`")
        os.system("mim install \"mmpose>=1.1.0\"")
        logger.info("`mmpose` installed successfully")

        logger.info("MMPose installed successfully! Shutting down now (it should work if you run this script again).")


from skellytracker.trackers.base_tracker.base_tracker import BaseTracker

logger = logging.getLogger(__name__)


class MMPoseTracker(BaseTracker):
    def __init__(self,
                 detection_config: str,
                 detection_checkpoint: str,
                 pose_config: str,
                 pose_checkpoint: str,
                 draw_heatmap: bool = False,
                 show_kpt_idx: bool = False,
                 skeleton_style: str = 'mmpose',
                 radius: int = 2,
                 thickness: int = 1,
                 ):
        super().__init__(tracked_object_names=["mmpose_data"])
        self.device = self._get_device_id()

        self.detector = self._build_detector(detection_checkpoint, detection_config)

        self.pose_estimator = self._build_pose_estimator(pose_checkpoint=pose_checkpoint,
                                                         pose_config=pose_config,
                                                         draw_heatmap=draw_heatmap,
                                                         show_kpt_idx=show_kpt_idx,
                                                         skeleton_style=skeleton_style,
                                                         radius=radius,
                                                         thickness=thickness)

        self.visualizer = VISUALIZERS.build(self.pose_estimator.cfg.visualizer)
        self.visualizer.set_dataset_meta(self.pose_estimator.dataset_meta, skeleton_style=skeleton_style)

    def _build_pose_estimator(self,
                              pose_checkpoint: str,
                              pose_config: str,
                              draw_heatmap: bool = False,
                              show_kpt_idx: bool = False,
                              skeleton_style: str = 'mmpose',
                              radius: int = 2,
                              thickness: int = 1,
                              alpha: float = 0.8) -> torch.nn.Module:
        # build pose estimator
        pose_estimator = init_pose_estimator(
            pose_config,
            pose_checkpoint,
            device=self.device,
            cfg_options=dict(
                model=dict(test_cfg=dict(output_heatmaps=draw_heatmap))))
        # build visualizer
        pose_estimator.cfg.visualizer.radius = radius
        pose_estimator.cfg.visualizer.alpha = alpha
        pose_estimator.cfg.visualizer.line_width = thickness
        return pose_estimator

    def _build_detector(self, detection_checkpoint: str, detection_config: str) -> torch.nn.Module:
        # build detector
        # make sure the config file exists or is downloadable
        if not Path(detection_config).exists() and not detection_config.startswith("http"):
            raise FileNotFoundError(f"detection config file {detection_config} does not exist")

        # make sure the checkpoint file exists or is downloadable
        if not Path(detection_checkpoint).exists() and not detection_checkpoint.startswith("http"):
            raise FileNotFoundError(f"detection checkpoint file {detection_checkpoint} does not exist")
        detector = init_detector(detection_config, detection_checkpoint, device=self.device)
        detector.cfg = adapt_mmdet_pipeline(detector.cfg)
        return detector

    def _get_device_id(self) -> str:
        if torch.cuda.is_available():
            # use Cuda:0 if available, otherwise use CPU
            return 'cuda:0'
        else:
            logger.warning(
                'CUDA compatible GPU is not available (`torch.cuda.is_available()` returned False), use CPU instead.')
            return 'cpu'

    def process_image(self,
                      image: np.ndarray,
                      show_interval=0,
                      det_cat_id: int = 0,
                      bbox_thr: float = 0.3,
                      nms_thr: float = 0.3,
                      kpt_thr: float = 0.3,
                      draw_heatmap: bool = False,
                      show_kpt_idx: bool = False,
                      skeleton_style: str = 'mmpose',
                      draw_bbox: bool = True,
                      show: bool = True):
        """Visualize predicted keypoints (and heatmaps) of one image."""
        self.raw_image = image
        # predict bbox
        det_result = inference_detector(self.detector, image)
        pred_instance = det_result.pred_instances.cpu().numpy()
        bboxes = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        bboxes = bboxes[np.logical_and(pred_instance.labels == det_cat_id,
                                       pred_instance.scores > bbox_thr)]
        bboxes = bboxes[nms(bboxes, nms_thr), :4]

        # predict keypoints
        pose_results = inference_topdown(self.pose_estimator, image, bboxes)
        data_samples = merge_data_samples(pose_results)

        self.annotated_image = self.annotate_image(image=image,
                                                   data_samples=data_samples,
                                                   draw_heatmap=draw_heatmap,
                                                   draw_bbox=draw_bbox,
                                                   show_kpt_idx=show_kpt_idx,
                                                   skeleton_style=skeleton_style,
                                                   wait_time=show_interval,
                                                   keypoint_threshold=kpt_thr)

        # if there is no instance detected, return None
        return data_samples.get('pred_instances', None)

    def annotate_image(self, image: np.ndarray,
                       data_samples: PoseDataSample,
                       draw_heatmap: bool = False,
                       draw_bbox: bool = True,
                       show_kpt_idx: bool = False,
                       skeleton_style: str = 'mmpose',
                       wait_time: int = 0,
                       keypoint_threshold: float = 0.3) -> np.ndarray:
        self.visualizer.add_datasample(
            'result',
            image,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=draw_heatmap,
            draw_bbox=draw_bbox,
            show_kpt_idx=show_kpt_idx,
            skeleton_style=skeleton_style,
            wait_time=wait_time,
            kpt_thr=keypoint_threshold)

        return self.visualizer.get_image()


if __name__ == '__main__':
    detection_config_in = "demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py"
    detection_checkpoint_in = "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
    pose_config_in = "configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py"
    pose_checkpoint_in = "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth"

    input_in = r"C:\Users\jonma\freemocap_data\recording_sessions\session_2023-11-13_10_44_51_asl_etc\recording_12_59_19_gmt-5__ASL\synchronized_videos\Camera_000_synchronized.mp4"

    tracker = MMPoseTracker(detection_config=detection_config_in,
                            detection_checkpoint=detection_checkpoint_in,
                            pose_config=pose_config_in,
                            pose_checkpoint=pose_checkpoint_in,
                            draw_heatmap=True,
                            show_kpt_idx=True,
                            skeleton_style='mmpose',
                            radius=2,
                            thickness=1)
    tracker.process_video(video_path=input_in,
                          show=True)
