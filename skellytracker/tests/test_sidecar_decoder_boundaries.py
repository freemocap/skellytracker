import numpy as np
import pytest

from skellytracker.core.detectors.object_detectors.yolox.yolox_decode import (
    yolox_detection_decode,
)
from skellytracker.core.detectors.processing.object_detection_decode import (
    decode_nms_baked_in,
)
from skellytracker.core.sidecar.model import DetectionDecodeSpec, SidecarModel
from skellytracker.core.sidecar.runtime import sidecar_model_spec
from skellytracker.tests.sidecar.test_runtime import _detector_sidecar_dict


def test_raw_anchor_decode_preserves_input_and_honors_threshold():
    raw = np.zeros((1, 84, 6), dtype=np.float32)
    raw[0, 0, 4:] = [0.9, 0.8]
    original = raw.copy()
    boxes, scores = yolox_detection_decode(
        [raw], 1.0, (64, 64), 0.5, 0.5, DetectionDecodeSpec()
    )
    np.testing.assert_array_equal(raw, original)
    np.testing.assert_allclose(boxes, [[-4, -4, 4, 4]])
    np.testing.assert_allclose(scores, [0.72])
    boxes, _ = yolox_detection_decode(
        [raw], 1.0, (64, 64), 0.8, 0.5, DetectionDecodeSpec()
    )
    assert boxes.shape == (0, 4)


def test_raw_anchor_decode_rejects_wrong_input_size():
    with pytest.raises(ValueError, match="anchor count"):
        yolox_detection_decode(
            [np.zeros((1, 84, 6))], 1.0, (128, 128), 0.5, 0.5, DetectionDecodeSpec()
        )


def test_nms_decode_does_not_silently_drop_second_camera():
    with pytest.raises(ValueError, match="one image"):
        decode_nms_baked_in(np.zeros((2, 4, 5)), 1.0, 0.5, DetectionDecodeSpec())


def test_nhwc_model_descriptor_uses_height_width(tmp_path):
    data = _detector_sidecar_dict(layout="NHWC", resize=None)
    data["sizes"]["nano"]["input"] = {"shape": [-1, 240, 320, 3]}
    spec = sidecar_model_spec(
        SidecarModel.model_validate(data),
        size="nano",
        batch_key="2",
        precision="fp32",
        name="toy",
        sidecar_dir=tmp_path,
    )
    assert spec.input_size == (240, 320)
