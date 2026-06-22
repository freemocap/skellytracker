import numpy as np
import pytest


class TestRtPoseDefinition:
    def test_point_count(self):
        from skellytracker.trackers.rt_pose_tracker.names_and_connections import (
            COCO_17_NAMES,
            RT_POSE_DEFINITION,
        )

        assert len(COCO_17_NAMES) == 17
        assert RT_POSE_DEFINITION.num_tracked_points == 17

    def test_point_names(self):
        from skellytracker.trackers.rt_pose_tracker.names_and_connections import COCO_17_NAMES

        expected_first = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")
        assert COCO_17_NAMES[:5] == expected_first

    def test_connection_count(self):
        from skellytracker.trackers.rt_pose_tracker.names_and_connections import RT_POSE_DEFINITION

        assert len(RT_POSE_DEFINITION.connections) == 16

    def test_connection_indices_in_range(self):
        from skellytracker.trackers.rt_pose_tracker.names_and_connections import RT_POSE_DEFINITION

        indices = RT_POSE_DEFINITION.connection_indices()
        assert len(indices) == 16
        for i, j in indices:
            assert 0 <= i < 17
            assert 0 <= j < 17

    def test_empty_point_cloud(self):
        from skellytracker.trackers.rt_pose_tracker.names_and_connections import (
            COCO_17_NAMES,
            RT_POSE_DEFINITION,
        )

        cloud = RT_POSE_DEFINITION.empty_point_cloud()
        assert cloud.xyz.shape == (17, 3)
        assert cloud.visibility.shape == (17,)
        assert np.all(np.isnan(cloud.xyz))
        assert cloud.names == COCO_17_NAMES


torch = pytest.importorskip("torch", reason="torch not installed")


class TestRtPoseObservation:
    def _make_tensors(self, n_persons: int = 1):
        keypoints_xy = torch.rand(n_persons, 17, 2) * 640
        scores = torch.rand(n_persons, 17)
        return keypoints_xy, scores

    def test_from_detection_results_single_person(self):
        from skellytracker.trackers.rt_pose_tracker.names_and_connections import COCO_17_NAMES
        from skellytracker.trackers.rt_pose_tracker.rt_pose_observation import RtPoseObservation

        kp, sc = self._make_tensors(1)
        obs = RtPoseObservation.from_detection_results(
            frame_number=0, keypoints_xy=kp, scores=sc, image_size=(640, 480)
        )
        assert obs.points.n_points == 17
        assert obs.points.xyz.shape == (17, 3)
        assert obs.points.visibility.shape == (17,)
        assert obs.points.names == COCO_17_NAMES
        assert obs.frame_number == 0
        assert obs.image_size == (640, 480)
        assert np.all(obs.points.xyz[:, 2] == 0.0)

    def test_from_detection_results_multi_person_takes_first(self):
        from skellytracker.trackers.rt_pose_tracker.rt_pose_observation import RtPoseObservation

        kp, sc = self._make_tensors(3)
        obs = RtPoseObservation.from_detection_results(
            frame_number=5, keypoints_xy=kp, scores=sc, image_size=(640, 480)
        )
        assert obs.points.n_points == 17
        np.testing.assert_allclose(obs.points.xyz[:, :2], kp[0].numpy(), rtol=1e-5)

    def test_from_detection_results_no_person(self):
        from skellytracker.trackers.rt_pose_tracker.rt_pose_observation import RtPoseObservation

        empty_kp = torch.zeros(0, 17, 2)
        empty_sc = torch.zeros(0, 17)
        obs = RtPoseObservation.from_detection_results(
            frame_number=0, keypoints_xy=empty_kp, scores=empty_sc, image_size=(640, 480)
        )
        assert obs.points.n_points == 17
        assert np.all(np.isnan(obs.points.xyz))

    def test_to_2d_array(self):
        from skellytracker.trackers.rt_pose_tracker.rt_pose_observation import RtPoseObservation

        kp, sc = self._make_tensors(1)
        obs = RtPoseObservation.from_detection_results(
            frame_number=0, keypoints_xy=kp, scores=sc, image_size=(640, 480)
        )
        assert obs.to_2d_array().shape == (17, 2)

    def test_to_json_string(self):
        from skellytracker.trackers.rt_pose_tracker.rt_pose_observation import RtPoseObservation

        kp, sc = self._make_tensors(1)
        obs = RtPoseObservation.from_detection_results(
            frame_number=0, keypoints_xy=kp, scores=sc, image_size=(640, 480)
        )
        json_str = obs.to_json_string()
        assert '"frame_number": 0' in json_str
        assert '"rt_pose"' in json_str


class TestRtPoseAnnotator:
    def _make_obs(self, scores_value: float = 0.9):
        from skellytracker.trackers.rt_pose_tracker.rt_pose_observation import RtPoseObservation

        kp = torch.rand(1, 17, 2) * 400 + 120
        sc = torch.ones(1, 17) * scores_value
        return RtPoseObservation.from_detection_results(
            frame_number=0, keypoints_xy=kp, scores=sc, image_size=(640, 480)
        )

    def test_annotate_image_draws_something(self):
        from skellytracker.trackers.rt_pose_tracker.rt_pose_annotator import (
            RtPoseAnnotator,
            RtPoseAnnotatorConfig,
        )

        annotator = RtPoseAnnotator.create(RtPoseAnnotatorConfig())
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        obs = self._make_obs(scores_value=0.9)
        annotated = annotator.annotate_image(image, obs)
        assert annotated.shape == image.shape
        assert not np.array_equal(annotated, image)

    def test_annotate_image_low_confidence_skips_drawing(self):
        from skellytracker.trackers.rt_pose_tracker.rt_pose_annotator import (
            RtPoseAnnotator,
            RtPoseAnnotatorConfig,
        )

        annotator = RtPoseAnnotator.create(RtPoseAnnotatorConfig(confidence_threshold=0.9))
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        obs = self._make_obs(scores_value=0.0)
        annotated = annotator.annotate_image(image, obs)
        assert np.array_equal(annotated, image)


@pytest.mark.skipif(
    pytest.importorskip("transformers", reason="transformers not installed") is None,
    reason="transformers not installed",
)
class TestRtPoseTrackerIntegration:
    def test_tracker_create_and_process_image(self, test_image):
        from skellytracker.trackers.rt_pose_tracker.__rt_pose_tracker import RtPoseTracker

        tracker = RtPoseTracker.create()
        obs = tracker.process_image(frame_number=0, image=test_image)
        assert obs.points.n_points == 17
        assert obs.to_2d_array().shape == (17, 2)
