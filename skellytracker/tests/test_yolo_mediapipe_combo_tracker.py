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
        MediapipeModelInfo.num_tracked_points_total,
        3,
    )

    expected_results = np.array(
        [
            [
                [738.8536071777344, 76.31869554519653, -481.7720413208008],
                [759.4620513916016, 73.04917931556702, -447.4338912963867],
                [767.7369689941406, 74.78330790996552, -447.7566909790039],
                [775.9542083740234, 76.69687628746033, -447.9768753051758],
                [733.1284332275391, 70.18355548381805, -444.6321487426758],
                [724.0713500976562, 69.79607820510864, -444.6977233886719],
                [715.2742004394531, 69.48489904403687, -444.5585632324219],
                [781.5924835205078, 86.53032660484314, -254.4525909423828],
                [696.5961456298828, 76.82911455631256, -237.26619720458984],
                [747.4419403076172, 89.09302175045013, -410.7792282104492],
                [711.8292236328125, 85.32133162021637, -406.14917755126953],
                [817.0913696289062, 149.26077961921692, -197.34506607055664],
                [550.1156997680664, 127.63105988502502, -134.19671058654785],
                [826.7817687988281, 219.57494258880615, -203.67210388183594],
                [398.47522735595703, 181.2681484222412, -91.3637638092041],
                [851.7813873291016, 281.61057472229004, -359.1621780395508],
                [255.78140258789062, 227.07135200500488, -215.78874588012695],
                [860.0052642822266, 299.12840366363525, -398.2832717895508],
                [214.94976043701172, 240.31760215759277, -246.70122146606445],
                [842.4692535400391, 299.00474309921265, -469.0795135498047],
                [223.54368209838867, 244.55156564712524, -314.70577239990234],
                [839.307861328125, 293.14420223236084, -392.55226135253906],
                [243.3875274658203, 238.81153106689453, -249.17715072631836],
                [633.1937408447266, 271.7419123649597, -41.56535625457764],
                [487.84019470214844, 257.71958112716675, 41.05104446411133],
                [540.9595489501953, 355.4134225845337, -49.43230152130127],
                [425.3153610229492, 342.35915422439575, 0.6983578205108643],
                [495.6198501586914, 415.8044958114624, 373.1425476074219],
                [404.4118881225586, 408.9740037918091, 422.87906646728516],
                [488.1563949584961, 420.5964231491089, 408.3951187133789],
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
        MediapipeModelInfo.num_tracked_points_total,
        3,
    )

    expected_results = np.array(
        [
            [
                [725.8726501464844, 77.78485000133514, -578.1089401245117],
                [749.6672058105469, 74.10527765750885, -547.0100784301758],
                [757.8436279296875, 75.96873700618744, -547.2169494628906],
                [765.7240295410156, 77.99112796783447, -547.3413848876953],
                [727.5205993652344, 71.15575969219208, -543.3893585205078],
                [720.9223175048828, 70.88138580322266, -543.4630966186523],
                [714.7412872314453, 70.67722678184509, -543.4294128417969],
                [775.4538726806641, 86.85999155044556, -345.21995544433594],
                [704.6842956542969, 76.9049459695816, -323.65928649902344],
                [735.5438995361328, 89.88833963871002, -500.3486633300781],
                [705.4883575439453, 85.32030165195465, -494.2338180541992],
                [815.977783203125, 151.07946753501892, -238.29364776611328],
                [545.4529190063477, 127.35316157341003, -207.39572525024414],
                [830.6045532226562, 220.57188749313354, -186.20323181152344],
                [393.6454391479492, 185.49273490905762, -133.2785701751709],
                [852.9004669189453, 282.4034786224365, -326.99546813964844],
                [253.29069137573242, 231.7178177833557, -237.40156173706055],
                [856.9590759277344, 299.4645380973816, -370.6595993041992],
                [204.9266815185547, 247.12258100509644, -266.78409576416016],
                [843.2437896728516, 300.2303194999695, -445.24627685546875],
                [216.95337295532227, 250.35706758499146, -345.37818908691406],
                [838.8117980957031, 294.3657445907593, -361.9205856323242],
                [235.01935958862305, 245.7415223121643, -273.3503723144531],
                [639.3540573120117, 269.0041923522949, 9.443286061286926],
                [479.8305892944336, 254.52640056610107, -9.362931251525879],
                [550.330696105957, 351.41347646713257, 100.7327651977539],
                [416.71329498291016, 345.4322361946106, -122.93316841125488],
                [492.806396484375, 408.0310249328613, 640.2886199951172],
                [386.9985580444336, 409.86316680908203, 257.8059768676758],
                [477.6706314086914, 407.6251745223999, 684.3799591064453],
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
        MediapipeModelInfo.num_tracked_points_total,
        3,
    )

    expected_results = np.array(
        [
            [
                [728.3435821533203, 78.09315919876099, -558.6944961547852],
                [751.5196228027344, 74.23020422458649, -531.4052581787109],
                [760.3465270996094, 75.99349915981293, -531.690788269043],
                [768.6563873291016, 77.85936176776886, -531.838493347168],
                [726.8606567382812, 71.32155239582062, -529.503173828125],
                [719.5426177978516, 70.99209666252136, -529.5947647094727],
                [712.640380859375, 70.73852062225342, -529.5146560668945],
                [775.8566284179688, 86.55619382858276, -352.61619567871094],
                [698.9208221435547, 77.0798796415329, -336.12987518310547],
                [738.851318359375, 90.26093602180481, -489.7779846191406],
                [705.5562591552734, 86.13133192062378, -485.1258850097656],
                [817.701416015625, 150.06993770599365, -263.64919662475586],
                [545.6637573242188, 128.16892862319946, -222.39032745361328],
                [829.2931365966797, 220.91928720474243, -216.90860748291016],
                [393.985595703125, 185.74366092681885, -151.75250053405762],
                [851.4951324462891, 283.2047510147095, -343.17344665527344],
                [256.5696907043457, 230.2279257774353, -239.59754943847656],
                [858.1615447998047, 301.1690926551819, -378.3252716064453],
                [208.00434112548828, 246.21910572052002, -261.22087478637695],
                [841.8076324462891, 301.7204689979553, -447.80059814453125],
                [221.08816146850586, 249.46983575820923, -332.62786865234375],
                [837.1615600585938, 295.68843841552734, -374.7406768798828],
                [239.96286392211914, 245.01717567443848, -271.5579605102539],
                [636.837272644043, 271.4845061302185, -11.809548139572144],
                [485.2410888671875, 257.02911615371704, 11.53409481048584],
                [548.1143188476562, 353.8384509086609, 17.83995270729065],
                [422.95154571533203, 340.9801125526428, -41.5569543838501],
                [491.4305877685547, 412.4346113204956, 535.1440048217773],
                [394.57775115966797, 404.82945442199707, 416.2659454345703],
                [478.27564239501953, 412.3047065734863, 579.522819519043],
            ]
        ]
    )
    assert np.allclose(
        processed_results[:, :30, :], expected_results[:, :30, :], atol=2
    )
