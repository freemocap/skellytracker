"""Standalone tracker recipes; applications select settings without rebuilding stages."""
from __future__ import annotations

from skellytracker.core.data_primitives.observation import Observation
from skellytracker.core.data_primitives.keypoints import Keypoints
from skellytracker.core.config.tracker_config import TrackerConfig
from skellytracker.core.config.detection_stage_config import DetectionStageConfig
from skellytracker.core.detectors.keypoint_detectors.rtmw.wholebody.rtmw_wholebody_detector import RTMWWholebodyDetectorConfig
from skellytracker.core.detectors.keypoint_detectors.mediapipe.mediapipe_model_manager import MediapipePoseModelComplexity
from skellytracker.core.detectors.object_detectors.yolox import YoloxPersonDetectorConfig
from skellytracker.core.detectors.object_detectors.keypoint_bbox import KeypointBoundingBoxDetectorConfig
from skellytracker.core.temporal_processing.temporal_processing_config import BBoxPolicyConfig, BBoxSmoothingConfig, KeypointResetPolicyConfig

_REDETECT_SECONDS = 5.0


def skeleton_model_specs(*, model_name: str, yolox_model_name: str):
    from skellytracker.core.detectors.object_detectors.yolox import YoloxPersonDetector
    from skellytracker.core.detectors.keypoint_detectors.rtmw.wholebody.rtmw_wholebody_detector import RTMWWholebodyDetector
    return [YoloxPersonDetector.model_spec(yolox_model_name), RTMWWholebodyDetector.model_spec(model_name)]


def charuco_tracker_config(board) -> TrackerConfig:
    from skellytracker.core.detectors.keypoint_detectors.charuco import CharucoDetectorConfig
    return TrackerConfig(stages=[DetectionStageConfig(name="charuco",
                         keypoint_detectors=[CharucoDetectorConfig(board=board)])])


def with_charuco_tracker(config: TrackerConfig, board) -> TrackerConfig:
    """Add board detection without dropping the selected preset's output contract."""
    if any(stage.name == "charuco" for stage in config.stages):
        raise ValueError("Tracker already declares a charuco stage")
    result = config.model_copy(deep=True)
    result.stages.extend(charuco_tracker_config(board).stages)
    return result


def human_tracker_config(
    *, preset: str, model_name: str = "rtmw-x-l_256x192", confidence_threshold: float = 0.4,
    video_fps: float = 30.0, keypoint_bbox_expansion: float = 0.05,
    model_complexity: MediapipePoseModelComplexity = MediapipePoseModelComplexity.HEAVY,
    detection_confidence: float = 0.5, presence_confidence: float = 0.5,
    tracking_confidence: float = 0.5, num_faces: int = 1,
) -> TrackerConfig:
    """Select a complete Tracker-owned recipe using application-supplied settings."""
    if preset == "mediapipe":
        return build_mediapipe_tracker_config(model_complexity=model_complexity,
            detection_confidence=detection_confidence, presence_confidence=presence_confidence,
            tracking_confidence=tracking_confidence, num_faces=num_faces)
    if preset == "rtmpose":
        return skeleton_tracker_config(model_name=model_name, confidence_threshold=confidence_threshold,
            video_fps=video_fps, keypoint_bbox_expansion=keypoint_bbox_expansion)
    if preset == "rtmw_mediapipe_hands":
        return combined_tracker_config(model_name=model_name, confidence_threshold=confidence_threshold,
            video_fps=video_fps, keypoint_bbox_expansion=keypoint_bbox_expansion,
            detection_confidence=detection_confidence, presence_confidence=presence_confidence,
            tracking_confidence=tracking_confidence)
    raise ValueError(f"Unknown human tracker preset {preset!r}")


def combined_tracker_config(
    *, model_name: str = "rtmw-x-l_256x192", confidence_threshold: float = 0.4,
    video_fps: float = 30.0, keypoint_bbox_expansion: float = 0.05,
    detection_confidence: float = 0.5, presence_confidence: float = 0.5,
    tracking_confidence: float = 0.5,
) -> TrackerConfig:
    """RTMW body/face with independently cropped MediaPipe hands."""
    from skellytracker.core.config.keypoint_assembly_config import KeypointAssemblyConfig, KeypointSourceConfig
    from skellytracker.core.io.tracker_profiles import tracker_profile, mediapipe_hand_names

    config = skeleton_tracker_config(model_name=model_name, confidence_threshold=confidence_threshold,
                                     video_fps=video_fps, keypoint_bbox_expansion=keypoint_bbox_expansion)
    body_names = tracker_profile("rtmpose").keypoint_names
    selected_names = tuple(name for name in body_names
                           if not name.startswith(("left_hand_", "right_hand_")))
    sources = [KeypointSourceConfig(stage_path=("body",), source_names=body_names,
                                    selected_names=selected_names)]
    for side in ("right", "left"):
        child = mediapipe_hand_stage(side, parent_preset="rtmpose", detection_confidence=detection_confidence,
                                          presence_confidence=presence_confidence,
                                          tracking_confidence=tracking_confidence)
        config.stages[0].children.append(child)
        names = mediapipe_hand_names(side)
        sources.append(KeypointSourceConfig(stage_path=("body", child.name),
                                            source_names=names, selected_names=names))
    config.keypoint_assembly = KeypointAssemblyConfig(target_stage="body", sources=tuple(sources))
    config.preset_name = "rtmw_mediapipe_hands"
    return config

def skeleton_tracker_config(
    *, model_name: str, confidence_threshold: float, video_fps: float, keypoint_bbox_expansion: float,
) -> TrackerConfig:
    redetect_interval = max(1, round(_REDETECT_SECONDS * video_fps))
    return TrackerConfig(
        preset_name="rtmpose",
        stages=[
            DetectionStageConfig(
                name="body",
                object_detector=YoloxPersonDetectorConfig(),
                keypoint_detectors=[
                    RTMWWholebodyDetectorConfig(
                        model_name=model_name,
                        confidence_threshold=confidence_threshold,
                    )
                ],
                bbox_policy=BBoxPolicyConfig(
                    redetect_interval=redetect_interval,
                    keypoint_bbox_expansion=keypoint_bbox_expansion,
                    # No keypoints_within_bbox_ratio check here: it compares
                    # keypoints against bbox_state.smooth_bbox, but RTMPose does
                    # not decode into smooth_bbox — rtmpose_letterbox_preprocess
                    # pads it by 1.25x and then widens it to the model's 3:4
                    # aspect. For a standing subject (tall, narrow box) the real
                    # decode window is 2-4x wider than smooth_bbox, so keypoints
                    # legitimately land outside it and the ratio sits near 0.4,
                    # under the 0.5 threshold. That forces a full-frame YOLOX
                    # re-detect every frame (~101 ms for 5 cameras), and because
                    # the fresh detection produces another narrow box, it latches
                    # on. Measured: detector ran on 45-100% of frames instead of
                    # the 0.7% redetect_interval implies. redetect_interval is
                    # the staleness bound until skellytracker's check compares
                    # against the actual decode window.
                    fitness_checks=[],
                    min_shrink_ratio_per_frame=0.995,
                    min_bbox_size_px=80.0,
                ),
                bbox_smoothing=BBoxSmoothingConfig(),
            )
        ]
    )


def mediapipe_hand_stage(
    side: str,
    *,
    parent_preset: str = "mediapipe",
    detection_confidence: float = 0.5,
    presence_confidence: float = 0.5,
    tracking_confidence: float = 0.5,
) -> DetectionStageConfig:
    """Build the shared hand crop and inference stage for either parent schema.

    Center between the parent's index and pinky tips, with a square side three
    times the larger wrist-to-tip distance. Only keypoint names depend on the
    parent preset; crop geometry, margin, smoothing and inference are shared.
    RTMW's existing finger estimates guide cropping only; assembled hand output
    still comes exclusively from MediaPipe. Missing tips suppress that crop.
    """
    from skellytracker.core.detectors.keypoint_detectors.mediapipe.hands.mediapipe_hand_detector import (
        MediapipeHandDetectorConfig,
    )

    if side not in ("left", "right"):
        raise ValueError(f"Unknown hand side {side!r}")
    if parent_preset == "mediapipe":
        index_keypoint, pinky_keypoint = f"{side}_index", f"{side}_pinky"
    elif parent_preset == "rtmpose":
        index_keypoint = f"{side}_hand_forefinger4"
        pinky_keypoint = f"{side}_hand_pinky_finger4"
    else:
        raise ValueError(f"Unsupported hand-crop parent preset {parent_preset!r}")
    wrist_keypoint = f"{side}_wrist"
    return DetectionStageConfig(
        name=f"{side}_hand",
        object_detector=KeypointBoundingBoxDetectorConfig(
            center_keypoint_names=(index_keypoint, pinky_keypoint),
            scale_keypoint_pairs=(
                (wrist_keypoint, index_keypoint),
                (wrist_keypoint, pinky_keypoint),
            ),
            scale_factor=3.0,
            min_box_size_px=30.0,
        ),
        keypoint_detectors=[
            MediapipeHandDetectorConfig(
                num_hands=1,
                assumed_handedness=side,
                min_hand_detection_confidence=detection_confidence,
                min_hand_presence_confidence=presence_confidence,
                min_hand_tracking_confidence=tracking_confidence,
            )
        ],
        bbox_smoothing=BBoxSmoothingConfig(),
    )


def build_mediapipe_tracker_config(
    *,
    model_complexity=None,
    detection_confidence: float = 0.5,
    presence_confidence: float = 0.5,
    tracking_confidence: float = 0.5,
    num_faces: int = 1,
) -> TrackerConfig:
    """Build a MediaPipe body+hands+face TrackerConfig.

    Pose is the parent "body" stage; left_hand/right_hand use shared hand crops
    (see mediapipe_hand_stage) instead of full-frame
    sibling detectors, since hands occupy a tiny fraction of a full-body frame
    and MediaPipe's palm detector struggles to find them at that scale. Face
    is also a child stage (full-frame, no crop) purely to preserve the
    existing pose/right_hand/left_hand/face point ordering that
    the prepared keypoint assembly and tracker profile expect.
    """
    from skellytracker.core.detectors.keypoint_detectors.mediapipe.body.mediapipe_pose_detector import (
        MediapipePoseDetectorConfig,
    )
    from skellytracker.core.detectors.keypoint_detectors.mediapipe.face.mediapipe_face_detector import (
        MediapipeFaceDetectorConfig,
    )
    from skellytracker.core.detectors.keypoint_detectors.mediapipe.mediapipe_model_manager import (
        MediapipePoseModelComplexity,
    )

    if model_complexity is None:
        model_complexity = MediapipePoseModelComplexity.HEAVY

    from skellytracker.core.config.keypoint_assembly_config import KeypointAssemblyConfig, KeypointSourceConfig
    from skellytracker.core.io.tracker_profiles import tracker_profile
    names = tracker_profile("mediapipe").keypoint_names
    selections = (
        (("body",), tuple(n for n in names if not n.startswith(("right_hand_", "left_hand_", "face_")))),
        (("body", "right_hand"), tuple(n for n in names if n.startswith("right_hand_"))),
        (("body", "left_hand"), tuple(n for n in names if n.startswith("left_hand_"))),
        (("body", "face"), tuple(n for n in names if n.startswith("face_"))),
    )
    return TrackerConfig(
        preset_name="mediapipe",
        keypoint_assembly=KeypointAssemblyConfig(target_stage="body", sources=tuple(
            KeypointSourceConfig(stage_path=path, source_names=points, selected_names=points)
            for path, points in selections)),
        stages=[
            DetectionStageConfig(
                name="body",
                keypoint_detectors=[
                    MediapipePoseDetectorConfig(
                        model_complexity=model_complexity,
                        num_poses=1,
                        min_pose_detection_confidence=detection_confidence,
                        min_pose_presence_confidence=presence_confidence,
                        min_pose_tracking_confidence=tracking_confidence,
                    ),
                ],
                children=[
                    mediapipe_hand_stage(
                        "right",
                        detection_confidence=detection_confidence,
                        presence_confidence=presence_confidence,
                        tracking_confidence=tracking_confidence,
                    ),
                    mediapipe_hand_stage(
                        "left",
                        detection_confidence=detection_confidence,
                        presence_confidence=presence_confidence,
                        tracking_confidence=tracking_confidence,
                    ),
                    DetectionStageConfig(
                        name="face",
                        keypoint_detectors=[
                            MediapipeFaceDetectorConfig(
                                num_faces=num_faces,
                                min_face_detection_confidence=detection_confidence,
                                min_face_presence_confidence=presence_confidence,
                                min_face_tracking_confidence=tracking_confidence,
                            )
                        ],
                    ),
                ],
                keypoint_reset_policy=KeypointResetPolicyConfig(max_consecutive_misses=10),
            )
        ]
    )


def mediapipe_tracker_config(
    *, model_complexity: MediapipePoseModelComplexity, detection_confidence: float,
    presence_confidence: float, tracking_confidence: float, num_hands: int, num_faces: int,
) -> TrackerConfig:
    """Build the two-sided human hand recipe; arbitrary hand counts are unsupported."""
    if num_hands != 2:
        raise ValueError("The human preset requires two hands, one per side")
    return build_mediapipe_tracker_config(
        model_complexity=model_complexity, detection_confidence=detection_confidence,
        presence_confidence=presence_confidence, tracking_confidence=tracking_confidence,
        num_faces=num_faces,
    )
