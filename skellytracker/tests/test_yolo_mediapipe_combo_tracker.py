import pytest
import numpy as np


from skellytracker.trackers.mediapipe_tracker.mediapipe_model_info import (
    MediapipeModelInfo,
)
from skellytracker.trackers.yolo_mediapipe_combo_tracker.yolo_mediapipe_combo_tracker import (
    YOLOMediapipeComboTracker,
)


@pytest.mark.usefixtures("test_image")
def test_process_image(test_image):
    tracker = YOLOMediapipeComboTracker(
        model_size="nano",
        model_complexity=0,
    )
    tracked_objects = tracker.process_image(test_image)

    assert len(tracked_objects) == 4
    assert tracked_objects["pose_landmarks"] is not None
    assert tracked_objects["pose_landmarks"].extra["landmarks"] is not None
    assert tracked_objects["right_hand_landmarks"] is not None
    assert tracked_objects["right_hand_landmarks"].extra["landmarks"] is not None
    assert tracked_objects["left_hand_landmarks"] is not None
    assert tracked_objects["left_hand_landmarks"].extra["landmarks"] is not None
    assert tracked_objects["face_landmarks"] is not None
    assert tracked_objects["face_landmarks"].extra["landmarks"] is not None


@pytest.mark.usefixtures("test_image")
def test_annotate_image(test_image):
    tracker = YOLOMediapipeComboTracker(
        model_size="nano",
        model_complexity=0,
    )
    tracker.process_image(test_image)

    assert tracker.annotated_image is not None


@pytest.mark.usefixtures("test_image")
def test_record_no_buffer(test_image):
    tracker = YOLOMediapipeComboTracker(
        model_size="nano",
        model_complexity=0,
        bounding_box_buffer_percentage=0,
    )
    tracked_objects = tracker.process_image(test_image)
    tracker.recorder.record(tracked_objects=tracked_objects)
    assert len(tracker.recorder.recorded_objects) == 1
    assert len(tracker.recorder.recorded_objects[0]) == 4

    processed_results = tracker.recorder.process_tracked_objects(
        image_size=test_image.shape[:2]
    )
    assert processed_results is not None
    assert processed_results.shape == (
        1,
        MediapipeModelInfo.num_tracked_points,
        3,
    )

    expected_results = np.array(
        [
            [
                [734.0492248535156, 75.59177935123444, -449.20562744140625],
                [754.4589996337891, 72.58412718772888, -415.50220489501953],
                [763.2180786132812, 74.40457999706268, -415.83839416503906],
                [771.6508483886719, 76.37354135513306, -416.0350799560547],
                [728.7122344970703, 69.72385704517365, -415.2852249145508],
                [720.6207275390625, 69.46459472179413, -415.3756332397461],
                [712.7108001708984, 69.28384602069855, -415.2159881591797],
                [775.8018493652344, 86.12753391265869, -225.70308685302734],
                [695.6344604492188, 76.76759541034698, -217.69142150878906],
                [742.9339599609375, 88.6310863494873, -379.02774810791016],
                [707.8136444091797, 84.86442804336548, -376.9959259033203],
                [811.6831207275391, 147.56260871887207, -169.21770095825195],
                [545.4062652587891, 127.61826038360596, -121.42580986022949],
                [826.0182189941406, 214.36566352844238, -159.21534538269043],
                [393.8945770263672, 183.3740472793579, -79.19782161712646],
                [847.5737762451172, 278.7761664390564, -306.181640625],
                [249.47546005249023, 228.41687679290771, -214.11535263061523],
                [856.395263671875, 297.3132348060608, -339.6835708618164],
                [209.2629051208496, 241.929030418396, -246.96613311767578],
                [842.9067230224609, 297.31162548065186, -410.8124542236328],
                [221.88369750976562, 245.78999519348145, -311.5140151977539],
                [840.1897430419922, 290.84245920181274, -338.3836364746094],
                [240.34542083740234, 239.5378303527832, -245.53743362426758],
                [630.5397033691406, 273.2768440246582, -38.868794441223145],
                [483.09581756591797, 259.9127697944641, 38.287367820739746],
                [544.7227478027344, 356.7100238800049, -56.52329444885254],
                [423.65062713623047, 341.21434450149536, -15.60276746749878],
                [489.99610900878906, 415.41388034820557, 364.4203567504883],
                [402.4936294555664, 406.1299180984497, 426.3703155517578],
                [481.54624938964844, 417.5926923751831, 401.75079345703125]
            ]
        ]
    )
    assert np.allclose(
        processed_results[:, :30, :], expected_results[:, :30, :], atol=2
    )


@pytest.mark.usefixtures("test_image")
def test_record_buffer_by_image_size(test_image):
    tracker = YOLOMediapipeComboTracker(
        model_size="nano",
        model_complexity=0,
        bounding_box_buffer_percentage=10,
        buffer_size_method="buffer_by_image_size",
    )
    tracked_objects = tracker.process_image(test_image)
    tracker.recorder.record(tracked_objects=tracked_objects)
    assert len(tracker.recorder.recorded_objects) == 1
    assert len(tracker.recorder.recorded_objects[0]) == 4

    processed_results = tracker.recorder.process_tracked_objects(
        image_size=test_image.shape[:2]
    )
    assert processed_results is not None
    assert processed_results.shape == (
        1,
        MediapipeModelInfo.num_tracked_points,
        3,
    )

    expected_results = np.array(
        [
            [
                [728.0298614501953, 81.64286971092224, -610.2832412719727],
                [750.8615875244141, 77.6283860206604, -579.5733261108398],
                [759.3793487548828, 79.25343990325928, -579.7866821289062],
                [767.2960662841797, 80.93595743179321, -579.8876571655273],
                [725.2166748046875, 74.2837142944336, -579.0089416503906],
                [717.9613494873047, 73.75618278980255, -579.1227722167969],
                [710.8364105224609, 73.35093855857849, -579.0372085571289],
                [776.0733032226562, 88.33156406879425, -371.30306243896484],
                [698.2285308837891, 77.81440258026123, -365.50621032714844],
                [739.5992279052734, 93.25603008270264, -529.6377563476562],
                [706.4430236816406, 88.42501759529114, -527.9743576049805],
                [814.8622131347656, 150.64239621162415, -258.2798194885254],
                [542.1471786499023, 127.73746848106384, -233.91403198242188],
                [830.4381561279297, 219.39132928848267, -191.65653228759766],
                [393.3975601196289, 186.7146635055542, -156.63755416870117],
                [854.3965911865234, 281.2304949760437, -331.9000244140625],
                [247.9853630065918, 232.6991629600525, -261.71335220336914],
                [858.5227966308594, 298.2491970062256, -368.53302001953125],
                [201.9668197631836, 247.1028184890747, -290.7101631164551],
                [845.6241607666016, 298.88365745544434, -447.0509338378906],
                [214.96929168701172, 250.2567958831787, -370.0352096557617],
                [841.1384582519531, 292.7099847793579, -367.3851013183594],
                [232.3802375793457, 245.90842008590698, -297.63317108154297],
                [641.7411041259766, 268.18910121917725, 2.854318618774414],
                [479.5010757446289, 254.63873147964478, -2.875555157661438],
                [548.7208557128906, 351.0516142845154, 62.11021900177002],
                [419.9262237548828, 342.3294997215271, -214.46731567382812],
                [498.5383605957031, 404.56350803375244, 594.5463562011719],
                [393.65840911865234, 409.21239852905273, 177.89228439331055],
                [485.1183319091797, 407.21795082092285, 640.0887298583984]
            ]
        ]
    )
    assert np.allclose(
        processed_results[:, :30, :], expected_results[:, :30, :], atol=2
    )


@pytest.mark.usefixtures("test_image")
def test_record_buffer_by_box_size(test_image):
    tracker = YOLOMediapipeComboTracker(
        model_size="nano",
        model_complexity=0,
        bounding_box_buffer_percentage=10,
        buffer_size_method="buffer_by_box_size",
    )
    tracked_objects = tracker.process_image(test_image)
    tracker.recorder.record(tracked_objects=tracked_objects)
    assert len(tracker.recorder.recorded_objects) == 1
    assert len(tracker.recorder.recorded_objects[0]) == 4

    processed_results = tracker.recorder.process_tracked_objects(
        image_size=test_image.shape[:2]
    )
    assert processed_results is not None
    assert processed_results.shape == (
        1,
        MediapipeModelInfo.num_tracked_points,
        3,
    )

    expected_results = np.array(
        [
            [
                [727.6454162597656, 79.77076292037964, -540.1993942260742],
                [752.6686859130859, 75.90135455131531, -512.2594451904297],
                [761.5357971191406, 77.72411942481995, -512.5232696533203],
                [770.0499725341797, 79.67693388462067, -512.6552200317383],
                [727.4240875244141, 72.33454763889313, -513.2247161865234],
                [720.2451324462891, 71.7227303981781, -513.3129501342773],
                [713.6248779296875, 71.2197893857956, -513.2208633422852],
                [777.7094268798828, 87.80843138694763, -329.8748016357422],
                [700.7302093505859, 77.01472878456116, -327.09617614746094],
                [738.1064605712891, 91.85711860656738, -469.6743392944336],
                [705.0703430175781, 86.99585616588593, -468.95755767822266],
                [816.8413543701172, 148.99025201797485, -236.60505294799805],
                [545.7090377807617, 128.4170651435852, -215.12311935424805],
                [829.9649047851562, 217.5825333595276, -180.53359985351562],
                [393.4770202636719, 186.34923934936523, -137.62948036193848],
                [850.8171844482422, 278.8819098472595, -301.76015853881836],
                [252.39004135131836, 231.54401063919067, -213.04611206054688],
                [856.624755859375, 296.855628490448, -332.9359817504883],
                [206.20765686035156, 247.86327838897705, -232.10474014282227],
                [841.011962890625, 297.72024393081665, -401.48365020751953],
                [218.29219818115234, 251.11578941345215, -304.64466094970703],
                [837.4552917480469, 290.8706760406494, -332.48313903808594],
                [235.5687713623047, 246.9295048713684, -244.80098724365234],
                [639.0462875366211, 270.3206419944763, -4.09934788942337],
                [485.26336669921875, 256.3050699234009, 3.9014053344726562],
                [545.6769943237305, 352.01712369918823, 29.53470230102539],
                [422.9113006591797, 340.9791684150696, -94.13705825805664],
                [492.47596740722656, 410.52337646484375, 530.4650115966797],
                [392.1165466308594, 407.2278642654419, 321.3273620605469],
                [478.7405014038086, 410.47574043273926, 573.6537933349609]
            ]
        ]
    )
    assert np.allclose(
        processed_results[:, :30, :], expected_results[:, :30, :], atol=2
    )
