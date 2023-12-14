import logging

from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel


logger = logging.getLogger(__name__)
class RunButtonWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self._layout = QVBoxLayout()

        self._title = QLabel(f"Detect skeleton in image:")
        self._layout.addWidget(self._title)
        
        self.run_button_widget = QPushButton('Run',self)
        self._layout.addWidget(self.run_button_widget)

        self.setLayout(self._layout)
