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
                [735.7643890380859, 77.78585314750671, -485.70934295654297],
                [757.0420074462891, 73.8272774219513, -451.7356872558594],
                [765.8688354492188, 75.42142689228058, -452.0623016357422],
                [774.5138549804688, 77.09728181362152, -452.29907989501953],
                [729.3473052978516, 70.99358797073364, -452.5171661376953],
                [720.2278137207031, 70.5675083398819, -452.57137298583984],
                [711.7780303955078, 70.23908793926239, -452.44258880615234],
                [780.3971099853516, 86.27676665782928, -254.8146629333496],
                [694.5964813232422, 76.93311989307404, -251.65258407592773],
                [745.7817077636719, 89.95153248310089, -411.6321563720703],
                [709.8857879638672, 86.26414954662323, -411.1183166503906],
                [817.0114135742188, 149.35277938842773, -178.73090744018555],
                [546.7483139038086, 128.01610708236694, -154.72180366516113],
                [825.9496307373047, 219.85299110412598, -142.79363632202148],
                [394.10709381103516, 183.29222917556763, -92.53458023071289],
                [850.7412719726562, 284.1442108154297, -277.7159309387207],
                [251.08980178833008, 229.20471668243408, -194.44988250732422],
                [860.8124542236328, 301.0432004928589, -309.02509689331055],
                [211.39860153198242, 242.8161120414734, -219.8578643798828],
                [843.7083435058594, 300.99024295806885, -385.36643981933594],
                [222.63912200927734, 246.6996932029724, -290.17900466918945],
                [839.26025390625, 295.74596643447876, -312.1304130554199],
                [240.05075454711914, 241.5028166770935, -226.83252334594727],
                [632.852668762207, 273.6677813529968, -22.331013679504395],
                [484.76829528808594, 259.1326332092285, 21.80124521255493],
                [549.3550491333008, 360.4587650299072, -38.61574411392212],
                [425.41954040527344, 343.2985496520996, -20.899696350097656],
                [486.68445587158203, 415.20827293395996, 411.0944366455078],
                [399.84127044677734, 408.2763719558716, 398.0601501464844],
                [476.32495880126953, 416.31832122802734, 451.4645767211914],
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
                [732.0687866210938, 79.0345823764801, -635.7109069824219],
                [753.8626098632812, 75.13578772544861, -608.0960464477539],
                [761.7510223388672, 76.88950717449188, -608.355827331543],
                [769.5530700683594, 78.81806373596191, -608.4667587280273],
                [729.2195892333984, 72.18442976474762, -607.5491333007812],
                [721.5387725830078, 71.86003804206848, -607.6824188232422],
                [714.1613006591797, 71.62245333194733, -607.6252746582031],
                [775.9226226806641, 86.93966388702393, -409.65084075927734],
                [698.8643646240234, 77.10046291351318, -402.1929931640625],
                [741.6652679443359, 90.84179520606995, -557.6795196533203],
                [709.4770050048828, 86.31318032741547, -555.5589294433594],
                [814.6518707275391, 149.26713109016418, -293.33160400390625],
                [544.8312759399414, 127.3474645614624, -270.33700942993164],
                [831.3512420654297, 219.2579483985901, -218.43671798706055],
                [392.59105682373047, 186.1086130142212, -182.6705551147461],
                [854.2007446289062, 282.19802141189575, -344.9155044555664],
                [248.35845947265625, 233.24616193771362, -273.1560516357422],
                [857.7934265136719, 299.5019817352295, -382.82962799072266],
                [202.44756698608398, 248.4213924407959, -300.88146209716797],
                [844.1407775878906, 300.61514139175415, -460.82714080810547],
                [212.99942016601562, 251.683087348938, -378.4083938598633],
                [839.8464965820312, 294.21818017959595, -380.2419662475586],
                [230.61071395874023, 246.78754091262817, -307.7861785888672],
                [639.9309539794922, 268.61598014831543, -0.536465011537075],
                [481.90975189208984, 254.10432815551758, 0.47791849821805954],
                [546.6202926635742, 351.9411635398865, 89.65752601623535],
                [416.8575668334961, 342.4706482887268, -87.49273300170898],
                [496.87782287597656, 406.89836025238037, 635.2282333374023],
                [393.65081787109375, 407.12096214294434, 370.8687210083008],
                [481.2765884399414, 408.57420444488525, 681.8212890625],
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
                [731.2718200683594, 77.88420975208282, -548.9945602416992],
                [754.6127319335938, 74.30741965770721, -521.489372253418],
                [762.8125762939453, 76.33775532245636, -521.7532348632812],
                [771.2681579589844, 78.52324604988098, -521.8650436401367],
                [730.0675964355469, 70.9510749578476, -521.0283660888672],
                [722.1942138671875, 70.54689288139343, -521.1295318603516],
                [714.8076629638672, 70.24498343467712, -521.0332870483398],
                [779.2241668701172, 87.20496118068695, -340.54332733154297],
                [700.3346252441406, 76.72817766666412, -330.6981658935547],
                [741.3204956054688, 90.19120931625366, -479.1018295288086],
                [708.2732391357422, 85.63711881637573, -476.2978744506836],
                [813.9542388916016, 149.21152353286743, -249.2682647705078],
                [545.847282409668, 128.2720971107483, -214.6741485595703],
                [831.6414642333984, 217.95576810836792, -199.20928955078125],
                [392.5309371948242, 185.60349941253662, -141.00683212280273],
                [850.6895446777344, 280.559663772583, -331.45111083984375],
                [252.97996520996094, 230.9888792037964, -229.0725326538086],
                [856.6841125488281, 298.4851026535034, -367.7482604980469],
                [203.86322021484375, 247.05041885375977, -250.86517333984375],
                [841.0806274414062, 299.39956426620483, -436.93775177001953],
                [215.9295654296875, 250.416419506073, -324.9034881591797],
                [836.7790985107422, 292.6754379272461, -363.13419342041016],
                [234.16423797607422, 246.12889766693115, -261.9901657104492],
                [639.0184783935547, 271.9633984565735, -11.131852865219116],
                [485.26206970214844, 257.6048684120178, 10.823948383331299],
                [544.7049331665039, 353.29198837280273, 29.705591201782227],
                [422.3688507080078, 341.57193660736084, -53.83963108062744],
                [484.64855194091797, 412.5028896331787, 527.0954132080078],
                [392.8578186035156, 406.18433475494385, 385.8837127685547],
                [471.3254165649414, 411.8346977233887, 569.1376495361328],
            ]
        ]
    )
    assert np.allclose(
        processed_results[:, :30, :], expected_results[:, :30, :], atol=2
    )
