from __future__ import annotations

import cv2
import numpy as np
import pytest

from skellytracker.core.detectors.keypoint_detectors.aruco import (
    ArucoAnnotator,
    ArucoAnnotatorConfig,
    ArucoDetector,
    ArucoDetectorConfig,
)
from skellytracker.core.sessions.cpu_session import CpuSession, CpuSessionConfig


@pytest.fixture(scope="module")
def aruco_ids() -> tuple[int, ...]:
    return (0, 1)


@pytest.fixture(scope="module")
def cpu_session() -> CpuSession:
    session = CpuSession.create(CpuSessionConfig())
    yield session
    session.close()


@pytest.fixture(scope="module")
def detector(aruco_ids, cpu_session) -> ArucoDetector:
    config = ArucoDetectorConfig(aruco_ids=aruco_ids)
    return ArucoDetector.create(config, cpu_session)


@pytest.fixture(scope="module")
def aruco_test_image() -> np.ndarray:
    """Synthetic image with ArUco marker ID 0 centered on a white background."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_img = cv2.aruco.generateImageMarker(dictionary, 0, 200)
    image = np.ones((480, 640), dtype=np.uint8) * 255
    y_offset = (480 - 200) // 2
    x_offset = (640 - 200) // 2
    image[y_offset : y_offset + 200, x_offset : x_offset + 200] = marker_img
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


class TestArucoDetector:
    def test_detect_returns_correct_shape(self, detector, aruco_ids, aruco_test_image):
        kpts = detector.detect(aruco_test_image)
        expected_total = len(aruco_ids) * 4
        assert kpts.xyz.shape == (expected_total, 3)
        assert kpts.visibility.shape == (expected_total,)
        assert len(kpts.names) == expected_total

    def test_detect_finds_marker(self, detector, aruco_test_image):
        kpts = detector.detect(aruco_test_image)
        assert kpts.n_valid > 0

    def test_detect_corner_names(self, detector, aruco_ids, aruco_test_image):
        kpts = detector.detect(aruco_test_image)
        for marker_id in aruco_ids:
            for j in range(4):
                assert f"ArucoMarker-{marker_id}-corner-{j}" in kpts.names

    def test_detect_name_pattern(self, detector, aruco_test_image):
        kpts = detector.detect(aruco_test_image)
        for name in kpts.names:
            assert name.startswith("ArucoMarker-")
            parts = name.split("-")
            assert len(parts) == 4
            assert parts[3].isdigit()
            assert int(parts[3]) in range(4)

    def test_detect_empty_on_blank_image(self, detector):
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        kpts = detector.detect(blank)
        assert np.all(np.isnan(kpts.xyz))
        assert np.all(kpts.visibility == 0.0)

    def test_detect_visibility_in_range(self, detector, aruco_test_image):
        kpts = detector.detect(aruco_test_image)
        assert np.all(kpts.visibility >= 0.0)
        assert np.all(kpts.visibility <= 1.0)

    def test_detect_z_is_zero_for_detected(self, detector, aruco_test_image):
        kpts = detector.detect(aruco_test_image)
        detected_mask = kpts.visibility > 0.0
        assert np.all(kpts.xyz[detected_mask, 2] == 0.0)

    def test_connections_is_empty(self):
        assert ArucoDetector.connections() == ()

    def test_undetected_marker_is_nan(self, detector, aruco_ids, aruco_test_image):
        kpts = detector.detect(aruco_test_image)
        # ID 1 is not in the test image — should be NaN
        for j in range(4):
            name = f"ArucoMarker-1-corner-{j}"
            idx = kpts.index_of(name)
            assert np.isnan(kpts.xyz[idx, 0])
            assert kpts.visibility[idx] == 0.0


class TestArucoAnnotator:
    def test_annotate_preserves_shape(self, detector, aruco_ids, aruco_test_image):
        kpts = detector.detect(aruco_test_image)
        annotator = ArucoAnnotator(config=ArucoAnnotatorConfig(), aruco_ids=aruco_ids)
        annotated = annotator.annotate(aruco_test_image, kpts)
        assert annotated.shape == aruco_test_image.shape

    def test_annotate_returns_different_image(self, detector, aruco_ids, aruco_test_image):
        kpts = detector.detect(aruco_test_image)
        annotator = ArucoAnnotator(config=ArucoAnnotatorConfig(), aruco_ids=aruco_ids)
        annotated = annotator.annotate(aruco_test_image, kpts)
        assert not np.array_equal(annotated, aruco_test_image)

    def test_annotate_blank_does_not_crash(self, detector, aruco_ids):
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        kpts = detector.detect(blank)
        annotator = ArucoAnnotator(config=ArucoAnnotatorConfig(), aruco_ids=aruco_ids)
        annotated = annotator.annotate(blank, kpts)
        assert annotated.shape == blank.shape
