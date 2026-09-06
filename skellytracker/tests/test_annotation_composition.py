import unittest

import numpy as np
from numpy.typing import NDArray

from skellytracker.core.annotation.keypoint_annotator import KeypointAnnotator, KeypointAnnotatorConfig
from skellytracker.core.data_primitives.keypoints import Keypoints
from skellytracker.core.data_primitives.observation import Observation, StageObservation
from skellytracker.core.detectors.keypoint_detectors.charuco.charuco_annotator import CharucoAnnotator, CharucoAnnotatorConfig
from skellytracker.core.detectors.keypoint_detectors.charuco.charuco_board_definition import CharucoBoardDefinition


def stage_points(*, name: str, x: float, y: float) -> Keypoints:
    return Keypoints(names=(name,), xyz=np.array([[x, y, 0.0]]), visibility=np.ones(1))


class AnnotationCompositionTests(unittest.TestCase):
    def test_board_and_other_stages_preserve_base_in_either_order(self) -> None:
        for board_first in (False, True):
            with self.subTest(board_first=board_first):
                base = np.zeros((360, 600, 3), dtype=np.uint8)
                base[320:330, 20:30] = 255
                stages = [
                    StageObservation(name="body", keypoints=stage_points(name="wrist", x=40.0, y=70.0)),
                    StageObservation(name="board", keypoints=stage_points(name="CharucoCorner-0", x=180.0, y=160.0)),
                ]
                if board_first:
                    stages.reverse()
                renderer = KeypointAnnotator(
                    config=KeypointAnnotatorConfig(),
                    stage_annotators={"board": CharucoAnnotator(
                        config=CharucoAnnotatorConfig(show_tracks=None),
                        board_def=CharucoBoardDefinition.create_letter_size_5x3(),
                    )},
                )
                output = renderer.annotate(image=base, observation=Observation(
                    frame_number=0, image_size=(360, 600), stages={stage.name: stage for stage in stages},
                ))
                self.assertTrue(np.any(output[65:76, 35:46]))
                self.assertTrue(np.any(output[153:168, 173:188]))
                np.testing.assert_array_equal(output[320:330, 20:30], base[320:330, 20:30])
                self.assertFalse(np.any(base[:300]))

    def test_nested_stage_is_drawn_when_parent_has_no_detections(self) -> None:
        renderer = KeypointAnnotator(config=KeypointAnnotatorConfig())
        base = np.zeros((100, 100, 3), dtype=np.uint8)
        observation = Observation(frame_number=0, image_size=(100, 100), stages={
            "parent": StageObservation(name="parent", children={
                "child": StageObservation(name="child", keypoints=stage_points(name="point", x=50.0, y=50.0)),
            }),
        })
        output = renderer.annotate(image=base, observation=observation)
        self.assertTrue(np.any(output[45:56, 45:56]))
        self.assertFalse(np.any(base))

    def test_specialized_stage_must_preserve_image_dimensions(self) -> None:
        class InvalidAnnotator:
            def annotate(self, image: NDArray[np.uint8], keypoints: Keypoints) -> NDArray[np.uint8]:
                return image[:1]
        renderer = KeypointAnnotator(config=KeypointAnnotatorConfig(), stage_annotators={"bad": InvalidAnnotator()})
        observation = Observation(frame_number=0, image_size=(100, 100), stages={
            "bad": StageObservation(name="bad", keypoints=stage_points(name="point", x=50.0, y=50.0)),
        })
        with self.assertRaisesRegex(ValueError, "changed image shape"):
            renderer.annotate(image=np.zeros((100, 100, 3), dtype=np.uint8), observation=observation)
