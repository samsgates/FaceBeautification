from ui.FaceBeautificationGUI_ui import Ui_FaceBeautificationGUI
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from FaceBeautification import FaceBeautification


class MyLabel(QLabel):

    zoom = pyqtSignal(bool, int, int)
    move_ = pyqtSignal(int, int)

    def __init__(self, parent):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.OpenHandCursor)
        self.img_ratio = 1.0
        self.x_value = 0
        self.y_value = 0
        self.last_cursor = None

    def reset(self):
        self.img_ratio = 1.0

    def set_image(self, img: QImage):
        img = img.scaled(img.size() * self.img_ratio)
        self.setPixmap(img)

    def mb_zoom(self, zoom_in, x, y, x_value, y_value):
        ix, iy = x / self.img_ratio, y / self.img_ratio
        step = 0.1
        if not zoom_in and self.img_ratio > 2 * step:
            self.img_ratio -= step
        if zoom_in and self.img_ratio < 10:
            self.img_ratio += step
        self.x_value = int(ix * self.img_ratio - x + x_value)
        self.y_value = int(iy * self.img_ratio - y + y_value)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setCursor(Qt.ClosedHandCursor)
            self.last_cursor = event.pos()
        else:
            event.ignore()

    def mouseMoveEvent(self, event):
        if self.last_cursor is not None:
            p = event.pos() - self.last_cursor
            self.move_.emit(p.x(), p.y())
        else:
            event.ignore()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setCursor(Qt.OpenHandCursor)
            self.last_cursor = None
        else:
            event.ignore()

    def wheelEvent(self, event):
        self.zoom.emit(event.angleDelta().y() > 0, event.x(), event.y())


class FaceBeautificationGUI(QMainWindow, Ui_FaceBeautificationGUI):

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.m_pic = MyLabel(self.m_pic_scroll_body)
        self.m_pic_scroll_layout.addWidget(self.m_pic)
        self.m_pic.zoom.connect(self.mb_zoom_images)
        self.m_pic.move_.connect(self.mb_move_images)
        self.m_demo = MyLabel(self.m_demo_scroll_body)
        self.engine = FaceBeautification()
        self.m_demo_scroll_layout.addWidget(self.m_demo)
        self.m_demo.zoom.connect(self.mb_zoom_images)
        self.m_demo.move_.connect(self.mb_move_images)
        m_pic_bar = self.m_pic_scroll.horizontalScrollBar()
        m_demo_bar = self.m_demo_scroll.horizontalScrollBar()
        m_pic_bar.valueChanged.connect(m_demo_bar.setValue)
        m_demo_bar.valueChanged.connect(m_pic_bar.setValue)
        m_pic_bar.rangeChanged.connect(self.mb_pic_horizontal_scroll_range_changed)
        m_demo_bar.rangeChanged.connect(self.mb_demo_horizontal_scroll_range_changed)
        m_pic_bar = self.m_pic_scroll.verticalScrollBar()
        m_demo_bar = self.m_demo_scroll.verticalScrollBar()
        m_pic_bar.valueChanged.connect(m_demo_bar.setValue)
        m_demo_bar.valueChanged.connect(m_pic_bar.setValue)
        m_pic_bar.rangeChanged.connect(self.mb_pic_vertical_scroll_range_changed)
        m_demo_bar.rangeChanged.connect(self.mb_demo_vertical_scroll_range_changed)

        self.setWindowIcon(self.style().standardIcon(QStyle.SP_DialogResetButton))
        self.m_action_open.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        self.m_action_save.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.m_action_undo.setIcon(self.style().standardIcon(QStyle.SP_ArrowLeft))
        self.m_action_redo.setIcon(self.style().standardIcon(QStyle.SP_ArrowRight))
        self.m_action_exit.setIcon(self.style().standardIcon(QStyle.SP_DialogCancelButton))
        self.m_action_about.setIcon(self.style().standardIcon(QStyle.SP_MessageBoxInformation))

        self.m_action_open.triggered.connect(self.mb_action_open)
        self.m_action_save.triggered.connect(self.mb_action_save)
        self.m_action_undo.triggered.connect(self.mb_action_undo)
        self.m_action_redo.triggered.connect(self.mb_action_redo)
        self.m_action_about.triggered.connect(self.mb_action_about)
        self.m_action_reset.triggered.connect(self.mb_action_reset)

        self.m_knn_apply.clicked.connect(self.mb_knn_apply)
        self.m_bigger_eyes_apply.clicked.connect(self.mb_bigger_eyes_apply)
        self.m_thinner_outline_apply.clicked.connect(self.mb_thinner_outline_apply)
        self.m_remove_beverage.clicked.connect(self.mb_remove_beverage)
        self.m_whitening.clicked.connect(self.mb_whitening)

        self.m_bigger_eyes_rate.valueChanged.connect(self.display_rate(self.m_bigger_eyes_rate_label))
        self.m_thinner_outline_rate.valueChanged.connect(self.display_rate(self.m_thinner_outline_rate_label))

        self.m_bigger_eyes_rate.setValue(5)
        self.m_thinner_outline_rate.setValue(5)
        self.m_action_save.setEnabled(False)
        self.m_action_reset.setEnabled(False)
        self.m_options.setEnabled(False)
        self.check_state()

    @staticmethod
    def display_rate(label):
        def wrapper(value):
            label.setText('%.2f' % (1 + value / 100))
        return wrapper

    def mb_action_open(self):
        filename, _ = QFileDialog().getOpenFileName(self, "Select Image", "", "Image Files(*.jpg *.png *.gif)")
        if not filename:
            return
        if not self.engine.load_image(filename):
            QMessageBox().information(self, "Error", "No face found in the image.", QMessageBox.Ok)
            return
        self.m_action_save.setEnabled(True)
        self.m_action_reset.setEnabled(True)
        self.m_options.setEnabled(True)

        self.m_pic.reset()
        self.m_demo.reset()
        self.check_state()
        self.m_pic.set_image(self.engine.get_original_image())
        self.m_demo.set_image(self.engine.get_beautified_image())

    def mb_action_save(self):
        filename, _ = QFileDialog().getSaveFileName(
            self, "Save Image", "", "PNG File(*.png);;JPEG File(*.jpg);;GIF File(*.gif)")
        if not filename:
            return
        self.engine.save_image(filename)

    def mb_action_undo(self):
        self.engine.prev_sequence()
        self.check_state()
        self.m_demo.set_image(self.engine.get_beautified_image())

    def mb_action_redo(self):
        self.engine.next_sequence()
        self.check_state()
        self.m_demo.set_image(self.engine.get_beautified_image())

    def mb_action_about(self):
        QMessageBox().information(self, "About", "All Rights Preserved.", QMessageBox.Ok)

    def mb_action_reset(self):
        self.engine.reset_sequence()
        self.check_state()
        self.m_demo.set_image(self.engine.get_beautified_image())

    def mb_knn_apply(self):
        gender = 'male' if self.m_gender_male.isChecked() else 'female'
        self.engine.apply_knn(
            gender=gender,
            eyebrows=self.m_knn_eyebrows.isChecked(),
            eyes=self.m_knn_eyes.isChecked(),
            nose=self.m_knn_nose.isChecked(),
            mouth=self.m_knn_mouth.isChecked(),
            outline=self.m_knn_outline.isChecked()
        )
        self.check_state()
        self.m_demo.set_image(self.engine.get_beautified_image())

    def mb_bigger_eyes_apply(self):
        rate = self.m_bigger_eyes_rate.value() / 100
        self.engine.apply_bigger_eyes(rate)
        self.check_state()
        self.m_demo.set_image(self.engine.get_beautified_image())

    def mb_thinner_outline_apply(self):
        rate = self.m_thinner_outline_rate.value() / 100
        self.engine.apply_thinner_eyes(rate)
        self.check_state()
        self.m_demo.set_image(self.engine.get_beautified_image())

    def mb_remove_beverage(self):
        self.engine.apply_remove_beverage()
        self.check_state()
        self.m_demo.set_image(self.engine.get_beautified_image())

    def mb_whitening(self):
        self.engine.apply_whitening()
        self.check_state()
        self.m_demo.set_image(self.engine.get_beautified_image())

    def check_state(self):
        if self.engine.is_sequence_empty():
            self.m_action_undo.setEnabled(False)
            self.m_action_redo.setEnabled(False)
        else:
            self.m_action_undo.setEnabled(not self.engine.at_sequence_front())
            self.m_action_redo.setEnabled(not self.engine.at_sequence_end())

    def resizeEvent(self, *args, **kwargs):
        if not self.engine.is_sequence_empty():
            self.m_pic.set_image(self.engine.get_original_image())
            self.m_demo.set_image(self.engine.get_beautified_image())

    def mb_zoom_images(self, zoom_in, x, y):
        if self.engine.is_sequence_empty():
            return
        x_bar = self.m_pic_scroll.horizontalScrollBar()
        y_bar = self.m_pic_scroll.verticalScrollBar()

        self.m_pic.mb_zoom(zoom_in, x, y, x_bar.value(), y_bar.value())
        self.m_demo.mb_zoom(zoom_in, x, y, x_bar.value(), y_bar.value())
        self.m_pic.set_image(self.engine.get_original_image())
        self.m_demo.set_image(self.engine.get_beautified_image())

    def mb_move_images(self, x, y):
        x_bar = self.m_pic_scroll.horizontalScrollBar()
        y_bar = self.m_pic_scroll.verticalScrollBar()
        x_bar.setValue(x_bar.value() - x)
        y_bar.setValue(y_bar.value() - y)

    def mb_pic_horizontal_scroll_range_changed(self):
        self.m_pic_scroll.horizontalScrollBar().setValue(self.m_pic.x_value)

    def mb_demo_horizontal_scroll_range_changed(self):
        self.m_demo_scroll.horizontalScrollBar().setValue(self.m_demo.x_value)

    def mb_pic_vertical_scroll_range_changed(self):
        self.m_pic_scroll.verticalScrollBar().setValue(self.m_pic.y_value)

    def mb_demo_vertical_scroll_range_changed(self):
        self.m_demo_scroll.verticalScrollBar().setValue(self.m_demo.y_value)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.XButton1 and self.m_action_undo.isEnabled():
            self.mb_action_undo()
        elif event.button() == Qt.XButton2 and self.m_action_redo.isEnabled():
            self.mb_action_redo()
