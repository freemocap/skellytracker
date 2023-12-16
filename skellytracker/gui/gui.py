import logging

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QFileDialog,
    QMainWindow,
    QLabel,
)

from skellytracker.gui.widgets.run_button_widget import RunButtonWidget


logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self):
        logger.info("Initializing the main window")
        super().__init__()

        self.setGeometry(100, 100, 600, 600)

        widget = QWidget()
        self._layout = QVBoxLayout()
        self._layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        widget.setLayout(self._layout)
        self.setCentralWidget(widget)

        self.folder_open_button = QPushButton("Load an image")
        self._layout.addWidget(self.folder_open_button)
        self.folder_open_button.clicked.connect(self._open_session_folder_dialog)

        self._path_to_folder_label = QLabel("No image file selected")
        self._layout.addWidget(self._path_to_folder_label)

        self.run_button = RunButtonWidget(self)
        self._layout.addWidget(self.run_button)
        self.run_button.run_button_widget.clicked.connect(
            lambda: print("Run button clicked")
        )

    def _open_session_folder_dialog(self):
        file_types = "Image files (*.jpg *.jpeg *.png)"
        self._image_path, _ = QFileDialog.getOpenFileName(
            None, "Choose an image", "", file_types
        )
        self._path_to_folder_label.setText(self._image_path)


def run_gui_window():
    app = QApplication([])
    win = MainWindow()
    win.show()
    app.exec()


if __name__ == "__main__":
    run_gui_window()
