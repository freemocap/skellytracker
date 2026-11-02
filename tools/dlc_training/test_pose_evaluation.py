import cv2
import numpy as np
from pose_evaluation import fit_predicted_pose, pose_errors


def test_independent_pose_recovers_known_projection():
    rng = np.random.default_rng(21)
    points = rng.uniform(-20, 20, (50, 3))
    intrinsics = np.array([[700., 0, 320], [0, 700., 240], [0, 0, 1]])
    rotation_vector = np.array([.2, -.3, .15])
    translation = np.array([4., -3., 240.])
    pixels = cv2.projectPoints(points, rotation_vector, translation, intrinsics, None)[0][:, 0]
    pose, count, reason = fit_predicted_pose(points, pixels, np.ones(50), intrinsics, (640, 480))
    assert reason == 'ok' and count == 50
    target = np.eye(4)
    target[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
    target[:3, 3] = translation
    np.testing.assert_allclose(pose, target, atol=1e-7)
    np.testing.assert_allclose(pose_errors(pose, target, points), 0, atol=1e-6)


def test_failure_does_not_return_target_pose():
    points = np.zeros((5, 3))
    pose, count, reason = fit_predicted_pose(points, np.ones((5, 2)), np.ones(5), np.eye(3), (640, 480))
    assert pose is None and reason == 'insufficient_points' and count == 5


def test_object_displacement_and_translation_units():
    target = np.eye(4)
    fitted = target.copy()
    fitted[:3, 3] = [3, 4, 0]
    np.testing.assert_allclose(pose_errors(fitted, target, np.eye(3)), [0, 5, 5])
