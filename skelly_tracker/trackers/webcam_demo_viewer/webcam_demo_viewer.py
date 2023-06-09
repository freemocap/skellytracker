import cv2

# Constants for key actions
KEY_INCREASE_EXPOSURE = ord("w")
KEY_DECREASE_EXPOSURE = ord("s")
KEY_RESET_EXPOSURE = ord("r")
KEY_QUIT = ord("q")

class WebcamDemoViewer:
    DEFAULT_EXPOSURE = -7

    def __init__(self, tracker, window_title: str = None, default_exposure: int = DEFAULT_EXPOSURE):
        """
        Initialize with a tracker and optional window title and default exposure.
        """
        self.tracker = tracker
        self.default_exposure = default_exposure
        if window_title is None:
            window_title = f"{tracker.__class__.__name__}"
        self.window_title = window_title

    def _set_exposure(self, cap, exposure):
        """
        Set the exposure of the camera.
        """
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)

    def _show_overlay(self, image, text):
        """
        Overlay text on the image.
        """
        y0, dy = 30, 25  # y0 - initial y value, dy - offset between lines
        for i, line in enumerate(text.split('\n')):
            y = y0 + i * dy
            cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .75, (100, 0, 255), 2)


    def run(self):
        """
        Run the camera viewer.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        exposure = self.default_exposure
        self._set_exposure(cap, exposure)


        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to read frame.")
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
                exposure = self.default_exposure
                self._set_exposure(cap, exposure)

            self._show_overlay(annotated_image, f"Exposure: {exposure}\n"
                                                f"Controls: \n`w`/`s`: exposure +/- \n'r': reset \n'q': quit")
            cv2.imshow(self.window_title, annotated_image)

        cap.release()
        cv2.destroyAllWindows()
