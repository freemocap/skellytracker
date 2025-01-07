import cv2
import pytest
import numpy as np


from skellytracker.trackers.charuco_tracker.charuco_tracker import CharucoTracker


@pytest.mark.usefixtures("charuco_test_image")
def test_process_image(charuco_test_image):
    charuco_squares_x_in = 7
    charuco_squares_y_in = 5
    number_of_charuco_markers = (charuco_squares_x_in - 1) * (charuco_squares_y_in - 1)
    charuco_ids = [str(index) for index in range(number_of_charuco_markers)]

    tracker = CharucoTracker(
        tracked_object_names=charuco_ids,
        squares_x=charuco_squares_x_in,
        squares_y=charuco_squares_y_in,
        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250),
    )
    tracked_objects = tracker.process_image(charuco_test_image)

    expected_results = {
        "0": (307.99796, 110.00571),
        "1": (336.93832, 120.001335),
        "2": (366.70923, 130.58067),
        "3": (396.36816, 141.78345),
        "4": (425.60236, 153.44989),
        "5": (455.322, 165.67236),
        "6": (294.39023, 135.20029),
        "7": (323.91107, 145.17943),
        "8": (353.4799, 155.70189),
        "9": (383.16925, 167.31921),
        "10": (412.6318, 179.24583),
        "11": (442.47086, 191.43738),
        "12": (280.9244, 160.91164),
        "13": (310.19556, 171.59216),
        "14": (339.86594, 182.31856),
        "15": (369.6749, 193.77588),
        "16": (399.5786, 205.84789),
        "17": (429.53903, 218.09035),
        "18": (267.12946, 187.75053),
        "19": (296.7608, 198.35608),
        "20": (326.31488, 209.67616),
        "21": (356.24832, 220.93948),
        "22": (386.3587, 233.16882),
        "23": (416.3993, 245.57997),
    }

    assert len(tracked_objects) == len(charuco_ids)
    for id in charuco_ids:
        tracked_corner = tracked_objects[id]
        assert tracked_corner.pixel_x is not None
        assert tracked_corner.pixel_y is not None
        assert np.allclose(
            (tracked_corner.pixel_x, tracked_corner.pixel_y), expected_results[id]
        )


@pytest.mark.usefixtures("test_image")
def test_image_without_charuco(test_image):
    charuco_squares_x_in = 7
    charuco_squares_y_in = 5
    number_of_charuco_markers = (charuco_squares_x_in - 1) * (charuco_squares_y_in - 1)
    charuco_ids = [str(index) for index in range(number_of_charuco_markers)]

    tracker = CharucoTracker(
        tracked_object_names=charuco_ids,
        squares_x=charuco_squares_x_in,
        squares_y=charuco_squares_y_in,
        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250),
    )
    tracked_objects = tracker.process_image(test_image)

    assert len(tracked_objects) == len(charuco_ids)
    for id in charuco_ids:
        tracked_corner = tracked_objects[id]
        assert tracked_corner.pixel_x is None
        assert tracked_corner.pixel_y is None


@pytest.mark.usefixtures("charuco_test_image")
def test_annotate_image(charuco_test_image):
    charuco_squares_x_in = 7
    charuco_squares_y_in = 5
    number_of_charuco_markers = (charuco_squares_x_in - 1) * (charuco_squares_y_in - 1)
    charuco_ids = [str(index) for index in range(number_of_charuco_markers)]

    tracker = CharucoTracker(
        tracked_object_names=charuco_ids,
        squares_x=charuco_squares_x_in,
        squares_y=charuco_squares_y_in,
        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250),
    )
    tracker.process_image(charuco_test_image)

    assert tracker.annotated_image is not None
    assert not np.all(tracker.annotated_image == charuco_test_image)


@pytest.mark.usefixtures("charuco_test_image")
def test_record(charuco_test_image):
    charuco_squares_x_in = 7
    charuco_squares_y_in = 5
    number_of_charuco_markers = (charuco_squares_x_in - 1) * (charuco_squares_y_in - 1)
    charuco_ids = [str(index) for index in range(number_of_charuco_markers)]

    tracker = CharucoTracker(
        tracked_object_names=charuco_ids,
        squares_x=charuco_squares_x_in,
        squares_y=charuco_squares_y_in,
        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250),
    )
    tracked_objects = tracker.process_image(charuco_test_image)
    tracker.recorder.record(tracked_objects=tracked_objects)
    assert len(tracker.recorder.recorded_objects) == 1

    processed_results = tracker.recorder.process_tracked_objects()
    assert processed_results is not None
    assert processed_results.shape == (1, len(charuco_ids), 2)

    # expected_results = np.array(
    #     [
    #         [
    #             [392.56927490234375, 140.40118408203125, np.nan],
    #             [414.94940185546875, 132.90655517578125, np.nan],
    #             [386.4353332519531, 125.51483154296875, np.nan],
    #             [446.2061767578125, 157.98883056640625, np.nan],
    #             [373.6619873046875, 138.93646240234375, np.nan],
    #             [453.78662109375, 265.081787109375, np.nan],
    #             [317.9375305175781, 231.9653778076172, np.nan],
    #             [465.893310546875, 396.12274169921875, np.nan],
    #             [220.12176513671875, 325.96636962890625, np.nan],
    #             [465.23358154296875, 499.0487365722656, np.nan],
    #             [142.17066955566406, 407.7397155761719, np.nan],
    #             [352.325439453125, 468.44671630859375, np.nan],
    #             [268.8867492675781, 448.7227783203125, np.nan],
    #             [310.899658203125, 630.5478515625, np.nan],
    #             [227.7810821533203, 617.9011840820312, np.nan],
    #             [269.08587646484375, 733.4285888671875, np.nan],
    #             [213.1557159423828, 741.54541015625, np.nan],
    #         ]
    #     ]
    # )
    # assert np.allclose(processed_results[:, :, :2], expected_results[:, :, :2], atol=1e-2)
    # assert np.isnan(processed_results[:, :, 2]).all()
