import pytest
import numpy as np


from skellytracker.trackers.mediapipe_pose_tracker.mediapipe_pose_model_info import (
    MediapipePoseModelInfo,
)
from skellytracker.trackers.mediapipe_pose_tracker.mediapipe_pose_tracker import (
    MediapipePoseTracker,
)


@pytest.mark.usefixtures("test_image")
def test_process_image(test_image):
    tracker = MediapipePoseTracker(model=MediapipePoseModelInfo.lite_model)
    tracked_objects = tracker.process_image(test_image)

    assert len(tracked_objects) == 1
    assert tracked_objects["pose_landmarks"] is not None
    assert tracked_objects["pose_landmarks"].extra["landmarks"] is not None


@pytest.mark.usefixtures("test_image")
def test_annotate_image(test_image):
    tracker = MediapipePoseTracker(model=MediapipePoseModelInfo.lite_model)
    tracker.process_image(test_image)

    assert tracker.annotated_image is not None


@pytest.mark.usefixtures("test_image")
def test_record(test_image):
    tracker = MediapipePoseTracker(model=MediapipePoseModelInfo.lite_model)
    tracked_objects = tracker.process_image(test_image)
    tracker.recorder.record(tracked_objects=tracked_objects)
    assert len(tracker.recorder.recorded_objects) == 1
    assert len(tracker.recorder.recorded_objects[0]) == 1

    processed_results = tracker.recorder.process_tracked_objects(
        image_size=test_image.shape[:2]
    )
    assert processed_results is not None
    assert processed_results.shape == (
        1,
        MediapipePoseModelInfo.num_tracked_points,
        3,
    )

    expected_results = np.array(
        [
            [
                [737.8633880615234, 80.54470896720886, -998.2334136962891],
                [764.2823028564453, 75.75023889541626, -961.2982177734375],
                [774.1596984863281, 77.29020237922668, -961.5772247314453],
                [783.9544677734375, 78.95700216293335, -961.7597198486328],
                [732.2080993652344, 72.30973720550537, -962.3476409912109],
                [722.0333099365234, 71.47456169128418, -962.5445556640625],
                [712.2611999511719, 70.70373773574829, -962.5089263916016],
                [794.1661071777344, 87.91862726211548, -680.0176239013672],
                [693.6607360839844, 75.13109922409058, -688.594970703125],
                [747.1823120117188, 93.111652135849, -888.8249206542969],
                [706.9366455078125, 88.960622549057, -891.3322448730469],
                [825.7304382324219, 152.64504075050354, -480.5356216430664],
                [542.7473449707031, 127.7156674861908, -478.25557708740234],
                [831.0322570800781, 223.27731370925903, -375.77526092529297],
                [401.58756256103516, 182.30609893798828, -388.03733825683594],
                [848.9546203613281, 291.1856746673584, -535.1567077636719],
                [243.98448944091797, 230.03113746643066, -559.8851776123047],
                [857.1239471435547, 304.9538826942444, -580.6122207641602],
                [199.0410614013672, 243.99523258209229, -611.9493865966797],
                [843.4824371337891, 306.8528652191162, -674.7665405273438],
                [205.25150299072266, 248.1018877029419, -705.6087493896484],
                [836.2294006347656, 301.3501524925232, -576.1663436889648],
                [224.90955352783203, 243.31161260604858, -602.7046585083008],
                [632.6493453979492, 269.25140619277954, 17.635692358016968],
                [481.7048645019531, 253.44205856323242, -17.26976752281189],
                [543.5726547241211, 357.85727977752686, 161.3166618347168],
                [412.00138092041016, 343.3478808403015, -276.2481880187988],
                [505.1728057861328, 406.4367628097534, 955.2671813964844],
                [389.64683532714844, 408.2848262786865, 294.84291076660156],
                [491.57684326171875, 409.3576240539551, 1025.6639099121094],
                [396.6215515136719, 414.53372955322266, 340.49427032470703],
                [489.5238494873047, 449.50209617614746, 836.7296600341797],
                [350.2317810058594, 448.2351493835449, 65.97118854522705],
            ]
        ]
    )
    assert np.allclose(
        processed_results, expected_results, atol=1
    )


@pytest.mark.usefixtures("test_image")
def test_record_missing_data(test_image):
    tracker = MediapipePoseTracker(model=MediapipePoseModelInfo.lite_model)
    tracked_objects = tracker.process_image(test_image)
    tracked_objects["pose_landmarks"].extra["landmarks"] = None
    tracker.recorder.record(tracked_objects=tracked_objects)

    processed_results = tracker.recorder.process_tracked_objects(
        image_size=test_image.shape[:2]
    )

    assert np.all(np.isnan(processed_results)), "Pose landmarks not converted to NaN"
