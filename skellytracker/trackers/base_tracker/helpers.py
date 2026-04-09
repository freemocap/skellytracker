def create_detector_from_config(detector_config: BaseDetectorConfig):
    """
    Create a detector instance from a picklable config.
    Called inside child processes — detector class imports are deferred
    to avoid pulling in mediapipe/cv2.aruco during module import.
    """

    match detector_config:
        case CharucoDetectorConfig():
            from skellytracker.trackers.charuco_tracker.charuco_detector import CharucoDetector
            return CharucoDetector.create(config=detector_config)
        case MediapipeDetectorConfig():
            from skellytracker.trackers.mediapipe_tracker import MediapipeDetector
            return MediapipeDetector.create(config=detector_config)
        case LegacyMediapipeDetectorConfig():
            from skellytracker.trackers.legacy_mediapipe_tracker.legacy_mediapipe_detector import LegacyMediapipeDetector
            return LegacyMediapipeDetector.create(config=detector_config)
        case _:
            raise TypeError(f"Unknown detector config type: {type(detector_config).__name__}")


def create_annotator_from_config(config: BaseDetectorConfig):
    """
    Create an image annotator matching the given detector config.
    Called inside child processes for drawing detection results onto frames.
    """

    match config:
        case CharucoDetectorConfig():
            from skellytracker.trackers.charuco_tracker.charuco_annotator import CharucoImageAnnotator, \
                CharucoAnnotatorConfig
            return CharucoImageAnnotator.create(config=CharucoAnnotatorConfig())
        case MediapipeDetectorConfig():
            from skellytracker.trackers.mediapipe_tracker import MediapipeAnnotator, MediapipeAnnotatorConfig
            return MediapipeAnnotator.create(config=MediapipeAnnotatorConfig())
        case LegacyMediapipeDetectorConfig():
            from skellytracker.trackers.legacy_mediapipe_tracker.legacy_mediapipe_annotator import LegacyMediapipeImageAnnotator, LegacyMediapipeAnnotatorConfig
            return LegacyMediapipeImageAnnotator.create(config=LegacyMediapipeAnnotatorConfig())
        case _:
            raise TypeError(f"Unknown detector config type for annotator: {type(config).__name__}")
