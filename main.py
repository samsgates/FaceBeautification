import sys
from FaceBeautificationGUI import FaceBeautificationGUI
from PyQt5.QtWidgets import QApplication

if __name__ == '__main__':
    app = QApplication(sys.argv)
    client_gui = FaceBeautificationGUI()
    client_gui.show()
    app.exec_()
    sys.exit()
