# -*- coding: utf-8 -*-

try:
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *
except ImportError:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *


class FloodFillConfig(QGroupBox):

    def __init__(self, parent=None):
        super(FloodFillConfig, self).__init__(parent)
        self.redLabel = QLabel("Red: 50")
        self.greenLabel = QLabel("Green: 20")
        self.blueLabel = QLabel("Blue: 50")
        self.redSlider = QSlider(Qt.Horizontal)
        self.redSlider.setMinimum(10)
        self.redSlider.setMaximum(120)
        self.redSlider.setValue(50)
        self.greenSlider = QSlider(Qt.Horizontal)
        self.greenSlider.setMinimum(10)
        self.greenSlider.setMaximum(120)
        self.greenSlider.setValue(20)
        self.blueSlider = QSlider(Qt.Horizontal)
        self.blueSlider.setMinimum(10)
        self.blueSlider.setMaximum(120)
        self.blueSlider.setValue(50)

        sliderLayout = QVBoxLayout()
        sliderLayout.addWidget(self.redLabel)
        sliderLayout.addWidget(self.redSlider)
        sliderLayout.addWidget(self.greenLabel)
        sliderLayout.addWidget(self.greenSlider)
        sliderLayout.addWidget(self.blueLabel)
        sliderLayout.addWidget(self.blueSlider)
        self.setLayout(sliderLayout)

        self.connect(self.redSlider, SIGNAL("valueChanged(int)"), self.redChanged)
        self.connect(self.greenSlider, SIGNAL("valueChanged(int)"), self.greenChanged)
        self.connect(self.blueSlider, SIGNAL("valueChanged(int)"), self.blueChanged)

    def getRedValue(self):
        return self.redSlider.value()

    def getGreenValue(self):
        return self.blueSlider.value()

    def getBlueValue(self):
        return self.blueSlider.value()

    def redChanged(self):
        self.redLabel.setText("Red: %d" % self.redSlider.value())

    def greenChanged(self):
        self.greenLabel.setText("Green: %d" % self.greenSlider.value())

    def blueChanged(self):
        self.blueLabel.setText("Blue: %d" % self.blueSlider.value())

    def getRedSlider(self):
        return self.redSlider

    def getGreenSlider(self):
        return self.greenSlider

    def getBlueSlider(self):
        return self.blueSlider

    def setEnabled(self):
        self.redSlider.setEnabled(True)
        self.greenSlider.setEnabled(True)
        self.blueSlider.setEnabled(True)

    def setDisabled(self):
        self.redSlider.setEnabled(False)
        self.greenSlider.setEnabled(False)
        self.blueSlider.setEnabled(False)

