import cv2

# Constants for key actions
KEY_INCREASE_EXPOSURE = ord("w")
KEY_DECREASE_EXPOSURE = ord("s")
KEY_RESET_EXPOSURE = ord("r")
KEY_QUIT = ord("q")

DEFAULT_EXPOSURE = -7


class CameraViewer:

    def __init__(self, tracker, window_title: str = None, **kwargs):
        self.tracker = tracker
        if window_title is None:
            window_title = f"{tracker.__class__.__name__}"

        self.window_title = window_title
    def _set_exposure(self, cap, exposure):
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)

    def run(self):
        cap = cv2.VideoCapture(0)
        exposure = DEFAULT_EXPOSURE
        self._set_exposure(cap, exposure)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.tracker.process_image(frame)
            annotated_image = self.tracker.annotated_image

            key = cv2.waitKey(1) & 0xFF
            if key == KEY_QUIT:
                break
            elif key == KEY_INCREASE_EXPOSURE:
                exposure += 1
                self._set_exposure(cap, exposure)
            elif key == KEY_DECREASE_EXPOSURE:
                exposure -= 1
                self._set_exposure(cap, exposure)
            elif key == KEY_RESET_EXPOSURE:
                exposure = DEFAULT_EXPOSURE
                self._set_exposure(cap, exposure)

            cv2.imshow(f"{self.window_title}: exposure: {exposure} - (w/s: exp +/-, r: reset, q: quit)",
                       annotated_image)

        cap.release()
        cv2.destroyAllWindows()
