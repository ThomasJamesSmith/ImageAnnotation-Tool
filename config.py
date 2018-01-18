# -*- coding: utf-8 -*-
# Configuration setting of image annotation tool

import os

try:
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *
except ImportError:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *

DEFAULT_FILLING_COLOR = QColor(Qt.red) #QColor(Qt.white)
DEFAULT_BACKGROUND_COLOR = QColor(Qt.black)

# Configuration of flood-fill
connectivity_Floodfill = 4
fixed_range = True


def outputDir(input_fname):
    """Define output directory of an input image"""
    baseDir = os.path.dirname(input_fname)
    return os.path.join(baseDir, "Output_Images")

def outputFile(input_fname):
    """Define the output file path of an input image.
    Save to ./Output_Images/xxx.png. (Remain the original name)"""
    outputFileDir = outputDir(input_fname)
    filename = os.path.splitext(os.path.basename(input_fname))[0]
    filename += ".png"
    return os.path.join(outputFileDir, filename)

def getLabelColor(imgPath):
    imgRoot = os.path.dirname(imgPath)
    path = os.path.join(imgRoot, "label.txt")
    if not os.path.exists(path):
        return None

    file = open(path, 'r')
    data = file.read().splitlines()
    dict = {}
    for info in data:
        label = info.split(',')
        red = int(label[0])
        green = int(label[1])
        blue = int(label[2])
        name = label[3]
        dict[name] = QColor(red, green, blue)

    file.close()
    return dict