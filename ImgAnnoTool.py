# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# This is Image Annotation Tool for annotating plantations.
# Created by Jingxiao Ma, and Thomas J. Smith.
# Languages: python 2 (2.7)
# Sys requirement: Linux / Windows 7 / OS X 10.8 or later versions
# Package requirement: PyQt 4 / OpenCv 2 / numpy


# compile new icons
# C:\Python27\Lib\site-packages\PyQt4\pyrcc4.exe -o qrc_resources.py resources.qrc

import os
import platform
from copy import copy
import sys
import qrc_resources
import numpy as np
import cv2
import csv
import config
from colorDialog import *
from FloodFillConfig import *
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries, find_boundaries
from skimage import io
import sip
import scipy
from Queue import Queue
#from threading import Thread
import time
from worker import *
import pdb

# RECOMMEND: Use PyQt4
try:
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *
except ImportError:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *


__version__ = '2.1'


class OpeningDirDialog(QMessageBox):
    def __init__(self, parent=None):
        super(OpeningDirDialog, self).__init__(parent)

        self.setText('Do you want to load images or videos?')
        self.addButton(QPushButton('Images'), QMessageBox.YesRole)
        self.addButton(QPushButton('Videos'), QMessageBox.NoRole)
        self.addButton(QPushButton('Cancel'), QMessageBox.RejectRole)


class LoadingVideoDialog(QMessageBox):

    def __init__(self, parent= None):
        super(LoadingVideoDialog, self).__init__()
        self.checkbox1 = QCheckBox()
        self.checkbox2 = QCheckBox()
        self.checkbox3 = QCheckBox()
        self.checkbox1.setChecked(True)
        self.checkbox2.setChecked(True)
        
        self.setWindowTitle("Loading Video...")
        self.checkbox1.setText("Do you want to reverse the video?")
        self.checkbox2.setText("Do you want just the first frame?")
        self.checkbox3.setText("Do you want to apply the superpixel algorithm?")
        #Access the Layout of the MessageBox to add the Checkbox
        layout = self.layout()
        layout.addWidget(self.checkbox1, 1,1)
        layout.addWidget(self.checkbox2, 1,2)
        layout.addWidget(self.checkbox3, 1,3)        
        self.setStandardButtons(QMessageBox.Cancel |QMessageBox.Ok)
        self.setDefaultButton(QMessageBox.Ok)
        self.setIcon(QMessageBox.Warning)

    def exec_(self, *args, **kwargs):
        """
        Override the exec_ method so you can return the value of the checkbox
        """
        return QMessageBox.exec_(self, *args, **kwargs), [self.checkbox1.isChecked(),self.checkbox2.isChecked(),self.checkbox3.isChecked()]


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.image = QImage()
        self.video = QMovie()

        self.cvimage = None         # Original image
        self.ioimage = None         # Original image
        self.outputMask = None      # Output image
        self.displayImage = None
        self.regionSegments = None

        self.allImages = []
        self.finished = False   # Indicate whether the image annotation is finished
        self.filename = None    # Path of the image file
        self.filepath = None    # Directory of the image file
        self.dirty = False      # Whether modified
        self.isLoading = True
        self.colourLabels = None
        self.hideImg = False
        self.hideSP = False
        self.spButton = 0
        self.clusterButton = 0
        self.floodMask = False
        self.hideCluster = False
        self.showSuggestedCluster = False
        self.showClusterOutlines = False
        self.hideMask = False
        self.isLaballing = False
        self.finishChoosingArea = False
        self.spActive = False
        self.spNum = 550
        self.sp_cluster_active = True
        self.alphaMaskNum = 0.5
        self.alphaClusterNum = 0.5
        self.spMask = None
        self.clusterMask = None
        self.suggestedClusterMask = None
        self.q = Queue(maxsize=0)
        self.q_Video = Queue(maxsize=0)
        self.q_Cluster = Queue(maxsize=1)
        self.num_threads = 7
        self.firstDone = False
        self.spMassActive = False
        self.spMassTotal = 0
        self.spMassComplete = 0
        self.mutex = QMutex()
        self.loadMultiVideoAns = []
        self.videoComplete = 0
        self.videoTotal = 0
        self.loadingVideoActive = False
        self.videoName = ""
        self.reverse = False
        self.sp_queue = []
        self.cluster_queue = []

        self.currentColor = config.DEFAULT_FILLING_COLOR
        self.currentLabelColor = config.DEFAULT_FILLING_COLOR
        self.currentOutlineColor = config.DEFAULT_FILLING_COLOR
        self.backgroundColor = config.DEFAULT_BACKGROUND_COLOR
        self.colorLabelDict = {}
        self.origin = QPoint()
        self.begin = QPoint()
        self.end = QPoint()
        self.points = []
        self.lines = []
        self.thickPen = QPen(Qt.DotLine)
        self.thickPen.setColor(Qt.red)
        self.thickPen.setWidth(2)
        self.maskFF = None
        self.pointFF = QPoint()
        self.choosingPointFF = False
        self.chooseFFPointSpinBoxValue = None

        self.historyStack = []
        self.redoStack = []

        # Define a QLabel to hold the image
        self.imageLabel = QLabel()
        self.imageLabel.setContextMenuPolicy(Qt.ActionsContextMenu)
        self.imageLabel.setBackgroundRole(QPalette.Base)
        self.imageLabel.setSizePolicy(QSizePolicy.Ignored,
                                      QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(True)
        self.imageLabel.setMouseTracking(False)
        # Overload paint event of image label
        self.imageLabel.paintEvent = self.paintLabel

        # Create scroll area for image
        self.scrollArea = QScrollArea()
        self.scrollArea.setAlignment(Qt.AlignCenter)
        self.scrollArea.setBackgroundRole(QPalette.Dark)
        self.scrollArea.setWidget(self.imageLabel)
        self.setCentralWidget(self.scrollArea)

        # Add a dock widget for viewing all files
        self.filesDockWidget = QDockWidget("Files", self)
        self.filesDockWidget.setObjectName("FilesDockWidget")
        self.filesDockWidget.setAllowedAreas(Qt.LeftDockWidgetArea|
                                             Qt.RightDockWidgetArea)
        self.fileListWidget = QListWidget()
        self.filesDockWidget.setWidget(self.fileListWidget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.filesDockWidget)
        self.filesDockWidget.setMinimumWidth(200)
        self.filesDockWidget.setMinimumHeight(200)
        self.fileListWidget.itemDoubleClicked.connect(self.fileItemDoubleClicked)

        self.filesDockWidget.setFeatures(QDockWidget.NoDockWidgetFeatures)

        # Add a dock widget for configuring flood fill
        self.floodFillDockWidget = QDockWidget("Flood fill setting", self)
        self.floodFillDockWidget.setObjectName("floodFillDockWidget")
        self.floodFillDockWidget.setAllowedAreas(Qt.LeftDockWidgetArea |
                                              Qt.RightDockWidgetArea)
        self.statusWidget = QListWidget()
        self.floodFillDockWidget.setWidget(self.statusWidget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.floodFillDockWidget)

        self.floodFillConfig = FloodFillConfig()
        self.floodFillDockWidget.setWidget(self.floodFillConfig)
        self.connect(self.floodFillConfig.getRedSlider(), SIGNAL("valueChanged(int)"),
                     self.changeFloodFill)
        self.connect(self.floodFillConfig.getGreenSlider(), SIGNAL("valueChanged(int)"),
                     self.changeFloodFill)
        self.connect(self.floodFillConfig.getBlueSlider(), SIGNAL("valueChanged(int)"),
                     self.changeFloodFill)
        self.redDiff = self.floodFillConfig.getRedValue()
        self.greenDiff = self.floodFillConfig.getGreenValue()
        self.blueDiff = self.floodFillConfig.getBlueValue()
        self.floodFillConfig.setDisabled()
        self.floodFillDockWidget.setMinimumWidth(200)
        self.floodFillDockWidget.setFeatures(QDockWidget.NoDockWidgetFeatures)

        # Add a dock widget for viewing actions
        self.logDockWidget = QDockWidget("Log", self)
        self.logDockWidget.setObjectName("LogDockWidget")
        self.logDockWidget.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        self.listWidget = QListWidget()
        self.logDockWidget.setWidget(self.listWidget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.logDockWidget)

        # Set status bar
        self.sizeLabel = QLabel()
        self.sizeLabel.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        status = self.statusBar()
        status.setSizeGripEnabled(False)
        status.addPermanentWidget(self.sizeLabel)
        status.showMessage("Ready", 5000)

        ###
        # open          = Ctrl+O
        # Dir Open      = Ctrl+D
        # Move file     = Ctrl+M
        # Save          = Ctrl+S
        # Undo          = Ctrl+Z
        # Re-do         = Ctrl+Y
        # Quit          = Ctrl+Q
        # Palette       = Ctrl+L
        # Colour Picker = Ctrl+P
        # Zoom in       = Ctrl+=
        # Zoom out      = Ctrl+-
        # Hide Image    = Ctrl+1
        # Hide Mask     = Ctrl+2
        # Hide Sp       = Ctrl+3
        # Hide Cluster  = Ctrl+4
        # Show Sug Clus = Ctrl+5
        # Show Clus Out = Ctrl+6
        # Chan Out Col  = Ctrl+7
        # None Geo      = Alt+1
        # Rectangle     = Alt+2
        # Elipse        = Alt+3
        # Polygon       = Alt+4
        # None SP       = Alt+5
        # Add SP        = Alt+6
        # None Cluster  = Alt+7
        # Add Cluster   = Alt+7
        # Run Cluster   = Alt+C
        # Run SP        = Alt+S
        # Clear         = Alt+X
        # Hide Log      = Alt+L
        # ###

        # Create actions
        fileOpenAction = self.createAction("&Open...", self.fileOpen, "Ctrl+O",
                                           "open", "Open an existing image file")
        dirOpenAction = self.createAction("&Dir Open...", self.dirOpen, "Ctrl+D",
                                          "open", "Open an existing directory")
        fileMoveAction = self.createAction("&Move file\n to done...", self.fileMove, "Ctrl+M",
                                           "move", "Move current file to done folder")

        self.saveAction = self.createAction("&Save...", self.saveFile, "Ctrl+S",
                                            "save", "Save modified image")
        self.saveAction.setEnabled(False)

        self.undoAction = self.createAction("&Undo...", self.undo, "Ctrl+Z",
                                            "undo", "Undo the last operation. "
                                            "NOTE: This operation is irreversible.")
        self.redoAction = self.createAction("&Re-do...", self.redo, "Ctrl+Y",
                                            "redo", "re-do the last undone operation.")
        self.undoAction.setEnabled(False)
        self.redoAction.setEnabled(False)

        quitAction = self.createAction("&Quit...", self.close, "Ctrl+Q",
                                       "close", "Close the application")
        zoomInAction = self.createAction("&Zoom\nIn...", self.zoomIn, "Ctrl+=",
                                         "zoom-in", "Zoom in image")
        zoomOutAction = self.createAction("&Zoom\nout...", self.zoomOut, "Ctrl+-",
                                          "zoom-out", "Zoom out image")
        hideLogViewerAction = self.createAction("&Hide Log...", self.hideLog, "Alt+L",
                                                None, "Hide log dock")

        self.hideOriginalAction = self.createAction("&Hide\nImage", self.hideButtonClick, "Ctrl+1",
                                                    "hide", "Hide original image", True, "toggled(bool)")
        self.hideMaskAction = self.createAction("&Hide\nMask", self.hideButtonClick, "Ctrl+2",
                                                    "hide", "Hide original image", True, "toggled(bool)")
        self.hideOriginalAction.setChecked(False)
        self.hideMaskAction.setChecked(False)

        self.paletteAction = self.createAction("&Palette...", self.chooseColor, "Ctrl+L",
                                               None, "Choose the colour to label items")

        # self.pickerAction = self.createAction("&Colour Picker...", self.pickColor, "Ctrl+P",
        #                                        "picker", "Picker the colour from mask")

        self.confirmAction = self.createAction("&Confirm...", self.confirmEdit, QKeySequence.InsertParagraphSeparator,
                                          "done", "Fill in the area with selected colour")
        self.confirmAction.setEnabled(False)
        self.deleteAction = self.createAction("&Unlabel...", self.deleteLabel, "Del",
                                         "delete", "Delete area and make it background")
        self.deleteAction.setEnabled(False)
        self.floodFillAction = self.createAction("&Flood\nFill", self.setFloodFillAction, "Ctrl+F",
                                                 "flood-fill", "Apply flood-fill to selected area", True,
                                                 "toggled(bool)")
        self.floodFillAction.setEnabled(False)

        self.clearAction = self.createAction("&Clear...", self.clearLabel, "Alt+X",
                                              "delete", "Clear all.")
        self.clearAction.setEnabled(True)

        
        self.spAction = self.createAction("&Superpixel", self.runSuperpixelAlg, "Alt+s", "superpixel",
                                          "Run superpixel Algorithm")
        
        self.hidespAction = self.createAction("&Hide\nSuperpixels", self.hideButtonClick, "Ctrl+3",
                                                    "hide", "Hide superpixel overlay", True, "toggled(bool)")


        # Create group of actions for superpixels
        spGroup = QActionGroup(self)
        #
        self.spMouseAction = self.createAction("&None...", self.setMouseAction, "Alt+5",
                                               "cursor", "No Action", True, "toggled(bool)")        
        spGroup.addAction(self.spMouseAction)
        #
        self.spAddAction = self.createAction("&Add/Remove\nSuperpixel", self.labelSPAdd, "Alt+6",
                                             "plus_minus", "Add(left click)/Remove(right click) superpixel to segment",
                                             True, "toggled(bool)")
        spGroup.addAction(self.spAddAction)
        #
        # self.spSubAction = self.createAction("&Subtract \nSuperpixel", self.labelSPAdd, "Ctrl+}",
        #                                      "SPsub", "Subtract superpixel from segment", True, "toggled(bool)")
        # spGroup.addAction(self.spSubAction)
        #
        self.spMouseAction.setChecked(True)
        self.spMouseAction.setEnabled(False)
        self.spAddAction.setEnabled(False)
        # self.spSubAction.setEnabled(False)
        self.hidespAction.setEnabled(False)

        # Create group of actions for cluster
        clusterGroup = QActionGroup(self)
        self.clusterAction = self.createAction("&Cluster", self.runClusterAlg, "Alt+c", "cluster",
                                               "add cluster overlay")
        self.clusterMouseAction = self.createAction("&None...", self.setMouseAction, "Alt+7",
                                                    "cursor", "No Action", True, "toggled(bool)")
        clusterGroup.addAction(self.clusterMouseAction)

        self.clusterAddAction = self.createAction("&Add/Remove \nCluster", self.labelClusterAdd, "Alt+8",
                                                  "plus_minus", "Add(left click)/Remove(right click) cluster to segment"
                                                  , True, "toggled(bool)")

        clusterGroup.addAction(self.clusterAddAction)
        # self.clusterSubAction = self.createAction("&Subtract \nCluster", self.labelClusterAdd, "Ctrl+}",
        #                                           "SPsub", "Subtract cluster frp, segment", True, "toggled(bool)")
        # clusterGroup.addAction(self.clusterSubAction)

        self.hideClusterAction = self.createAction("&Hide\nCluster", self.hideButtonClick, "Ctrl+4",
                                                   "hide", "Hide cluster overlay", True, "toggled(bool)")
        self.showSuggestedClusterAction = self.createAction("&Show\nSuggested\nCluster", self.hideButtonClick, "Ctrl+5",
                                                            "hide", "Show suggested cluster overlay", True,
                                                            "toggled(bool)")

        self.showClusterOutlinesAction = self.createAction("&Show\nCluster\nOutline", self.hideButtonClick, "Ctrl+6",
                                                            "hide", "Show cluster outline overlay", True,
                                                            "toggled(bool)")
        self.clusterPaletteAction = self.createAction("&Change Outline\nColour", self.chooseOutlineColor, "Ctrl+7",
                                                      None, "Choose the colour to label items")
        clusterGroup.addAction(self.clusterPaletteAction)
        self.clusterPaletteAction.setEnabled(False)

        self.clusterMouseAction.setChecked(True)
        self.clusterMouseAction.setEnabled(False)
        self.clusterAddAction.setEnabled(False)
        # self.clusterSubAction.setEnabled(False)
        self.hideClusterAction.setEnabled(False)
        self.showSuggestedClusterAction.setEnabled(False)
        self.showClusterOutlinesAction.setEnabled(False)

        labelGroup = QActionGroup(self)
        self.labelAction = self.createAction("&Open\nLabels", self.openLabels, "Alt+f", "labels",
                                             "Open label file, create if doesn't exist")
        labelGroup.addAction(self.labelAction)
        self.labelAction.setEnabled(False)

        self.labelAddAction = self.createAction("&Add New \nSemantic Label", self.labelToFileAdd, "Ctrl+{",
                                             "SPadd", "Add new semantic label to labels file", True)
        labelGroup.addAction(self.labelAddAction)
        self.labelAddAction.setEnabled(False)

        self.labelPaletteAction = self.createAction("&Choose Label\nColour", self.chooseLabelColor, "Ctrl+L",
                                                    None, "Choose the colour to label items")
        labelGroup.addAction(self.labelPaletteAction)
        self.labelPaletteAction.setEnabled(False)

        ################################################################################################################
        #   self.labelTextBox = QTextEdit(self)
        #   self.labelTextBox.resize(150, 50)
        ################################################################################################################



        
        helpAboutAction = self.createAction("&About...", self.helpAbout, None, "helpabout")
        helpHelpAction = self.createAction("&Help...", self.helpHelp, None, "help")

        # Create group of actions for labelling image
        editGroup = QActionGroup(self)
        self.mouseAction = self.createAction("&None...", self.setMouseAction, "Alt+1",
                                        "cursor", "No Action", True, "toggled(bool)")
        editGroup.addAction(self.mouseAction)
        self.rectLabelAction = self.createAction("&Rect...", self.labelRectOrEllipse, "Alt+2",
                                       "rectangle", "Annotation a rectangle area", True, "toggled(bool)")
        editGroup.addAction(self.rectLabelAction)
        self.ellipseLabelAction = self.createAction("&Ellipse...", self.labelRectOrEllipse, "Alt+3",
                                                 "ellipse", "Annotation an ellipse area", True, "toggled(bool)")
        editGroup.addAction(self.ellipseLabelAction)
        self.polygonLabelAction = self.createAction("&Polygon...", self.labelPolygon, "Alt+4",
                                                    "polygon", "Annotation an irregular polygon area",
                                                    True, "toggled(bool)")
        editGroup.addAction(self.polygonLabelAction)
        self.mouseAction.setChecked(True)

        self.resetableActions = ((self.hideOriginalAction, False),
                                 (self.hideMaskAction, False),
                                 (self.mouseAction, True),
                                 (self.spMouseAction, True),
                                 (self.clusterMouseAction, True))

        # Set spin box
        self.zoomSpinBox = QSpinBox()
        self.zoomSpinBox.setRange(1, 400)
        self.zoomSpinBox.setSuffix(" %")
        # Need auto-fit later
        self.zoomSpinBox.setValue(100)
        self.zoomSpinBox.setToolTip("Zoom the image")
        self.zoomSpinBox.setStatusTip(self.zoomSpinBox.toolTip())
        self.zoomSpinBox.setButtonSymbols(QAbstractSpinBox.NoButtons)
        # self.zoomSpinBox.setFocusPolicy(Qt.NoFocus)
        self.connect(self.zoomSpinBox,
                     SIGNAL("valueChanged(int)"), self.showImage)

        self.lastSpinboxValue = self.zoomSpinBox.value()

        self.spSpinBox = QSpinBox()
        self.spSpinBox.setRange(50, 5000)
        self.spSpinBox.setSingleStep(50)
        self.spSpinBox.setValue(1500)
        self.spSpinBox.setToolTip("Set number of Superpixels")
        self.spSpinBox.setStatusTip(self.spSpinBox.toolTip())
        self.spSpinBox.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.connect(self.spSpinBox, SIGNAL("valueChanged(int)"), self.updateSPNum)

        self.sigmaSpinBox = QSpinBox()
        self.sigmaSpinBox.setRange(1, 10)
        self.sigmaSpinBox.setValue(2)
        self.sigmaSpinBox.setToolTip("Set Sigma")
        self.sigmaSpinBox.setStatusTip(self.sigmaSpinBox.toolTip())
        self.sigmaSpinBox.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.connect(self.sigmaSpinBox, SIGNAL("valueChanged(int)"), self.updateSPSigma)

        self.compSpinBox = QSpinBox()
        self.compSpinBox.setRange(-100, 100)
        self.compSpinBox.setSingleStep(10)
        self.compSpinBox.setValue(10)
        self.compSpinBox.setToolTip("Set Compactness")
        self.compSpinBox.setStatusTip(self.compSpinBox.toolTip())
        self.compSpinBox.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.connect(self.compSpinBox, SIGNAL("valueChanged(int)"), self.updateSPCompactness)

        self.spNum = self.spSpinBox.value()
        self.sigma = self.sigmaSpinBox.value()
        self.compactness = self.compSpinBox.value()

        self.alphaMaskSpinBox = QDoubleSpinBox()
        self.alphaMaskSpinBox.setRange(0, 1)
        self.alphaMaskSpinBox.setValue(0.5)
        self.alphaMaskSpinBox.setSingleStep(0.01)
        self.alphaMaskSpinBox.setToolTip("Set alpha number for mask overlays")
        self.alphaMaskSpinBox.setStatusTip(self.alphaMaskSpinBox.toolTip())
        self.alphaMaskSpinBox.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.connect(self.alphaMaskSpinBox,
                     SIGNAL("valueChanged(double)"), self.updateAlphaMaskNum)
        self.alphaMaskNum = self.alphaMaskSpinBox.value()

        self.alphaClusterSpinBox = QDoubleSpinBox()
        self.alphaClusterSpinBox.setRange(0, 1)
        self.alphaClusterSpinBox.setValue(0.5)
        self.alphaClusterSpinBox.setSingleStep(0.01)
        self.alphaClusterSpinBox.setToolTip("Set alpha number for cluster overlays")
        self.alphaClusterSpinBox.setStatusTip(self.alphaClusterSpinBox.toolTip())
        self.alphaClusterSpinBox.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.connect(self.alphaClusterSpinBox,
                     SIGNAL("valueChanged(double)"), self.updateAlphaClusterNum)
        self.alphaClusterNum = self.alphaClusterSpinBox.value()

        # Create color dialog
        self.colorDialog = ColorDialog(parent=self)

        # Create menu bar
        self.fileMenu = self.menuBar().addMenu("&File")
        self.fileMenuActions = (fileOpenAction, dirOpenAction, fileMoveAction, self.saveAction, None, quitAction)
        self.connect(self.fileMenu, SIGNAL("aboutToShow()"),
                     self.updateFileMenu)
        self.fileMenu.setMaximumWidth(400)

        editMenu = self.menuBar().addMenu("&Edit")
        # self.addActions(editMenu, (self.undoAction, self.redoAction, None, self.paletteAction, self.pickerAction, None,
        self.addActions(editMenu, (self.undoAction, self.redoAction, None, self.paletteAction , None,
                                   self.confirmAction, self.deleteAction, self.floodFillAction,
                                   None, self.mouseAction, self.rectLabelAction,
                                   self.ellipseLabelAction, self.polygonLabelAction))

        viewMenu = self.menuBar().addMenu("&View")
        self.addActions(viewMenu, (zoomInAction, zoomOutAction, self.hideOriginalAction, self.hideMaskAction,
                                   None, hideLogViewerAction))

        helpMenu = self.menuBar().addMenu("&Help")
        self.addActions(helpMenu, (helpAboutAction, helpHelpAction, None))

        # Create tool bar
        self.toolBar = self.addToolBar("File&Edit")
        self.toolBar.setAllowedAreas(Qt.TopToolBarArea)
        self.toolBar.setMovable(False)
        self.toolBar.setIconSize(QSize(24, 24))
        self.toolBar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.toolBar.setObjectName("ToolBar")
        self.toolBarActions_1 = (fileOpenAction, dirOpenAction, self.saveAction, fileMoveAction, self.undoAction,
                                 self.redoAction, quitAction, self.clearAction, None, zoomInAction)
        self.addActions(self.toolBar, self.toolBarActions_1)
        self.toolBar.addWidget(self.zoomSpinBox)

        self.toolBarActions_2 = (zoomOutAction, None, self.hideOriginalAction, self.hideMaskAction,
                                 self.hidespAction, self.hideClusterAction, self.showSuggestedClusterAction,
                                 self.showClusterOutlinesAction, self.clusterPaletteAction, None)
        # self.toolBarActions_2 = (zoomOutAction, None, self.hideOriginalAction, self.hideMaskAction,
        #                          self.hidespAction, self.clusterPaletteAction, None)

        self.addActions(self.toolBar, self.toolBarActions_2)
        self.toolBar.addWidget(self.alphaMaskSpinBox)
        self.toolBar.addWidget(self.alphaClusterSpinBox)

        self.toolBarActions_3 = (None, self.labelAction, self.labelAddAction, self.labelPaletteAction, None,
                                 self.paletteAction, self.confirmAction, self.deleteAction,
                                 self.floodFillAction, None, self.mouseAction, self.rectLabelAction,
                                 self.ellipseLabelAction, self.polygonLabelAction, None)
        self.addActions(self.toolBar, self.toolBarActions_3)
        self.toolBar.addWidget(self.spSpinBox)
        self.toolBar.addWidget(self.sigmaSpinBox)
        self.toolBar.addWidget(self.compSpinBox)


        self.toolBarActions_4 = (self.spAction, self.spMouseAction, self.spAddAction, None,
                                 self.clusterAction, self.clusterMouseAction, self.clusterAddAction, None)
        # self.toolBarActions_4 = (self.spAction, self.spMouseAction, self.spAddAction, None)

        self.addActions(self.toolBar, self.toolBarActions_4)

        self.colorLabelBar = QToolBar("Labels and colours")
        self.addToolBar(Qt.LeftToolBarArea, self.colorLabelBar)
        self.colorLabelBar.setObjectName("LabelToolBar")
        self.colorLabelBar.setIconSize(QSize(36, 36))
        self.colorLabelBar.setMovable(False)
        self.colorLabelBar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.labelsGroup = None


        # Restore application's setting
        settings = QSettings()
        self.recentFiles = settings.value("RecentFiles").toStringList()
        size = settings.value("MainWindow/Size",
                              QVariant(QSize(800, 600))).toSize()
        self.resize(size)
        # Put in the middle of screen by default
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        position = settings.value("MainWindow/Position",
                                  QVariant(QPoint((screen.width() - size.width()) / 2,
                                                  (screen.height() - size.height()) / 2))).toPoint()
        self.move(position)
        self.restoreState(
            settings.value("MainWindow/State").toByteArray())
        self.setWindowTitle("Image Annotation Tool")
        # Load the file that displayed last time the app was closed
        QTimer.singleShot(0, self.loadInitFile)
        # Overload mouse wheel event to zoom image
        self.imageLabel.wheelEvent = self.mouseWheelEvent
        
        self.threadpool = QThreadPool()
        for i in range(self.num_threads-2):
            worker = Worker(self.runMassSuperpixelQueue, self.q)
            worker.signals.progress.connect(self.loadFirstSPImage)
            self.threadpool.start(worker)
        
        self.workerVideo = Worker(self.runVideoQueue, self.q_Video)
        self.workerVideo.signals.progress.connect(self.loadedVideo)
        self.threadpool.start(self.workerVideo)

        self.workerCluster = Worker(self.runClusterQueue, self.q_Cluster)
        self.workerCluster.signals.progress.connect(self.loadedCluster)
        self.threadpool.start(self.workerCluster)

        self.keys = {}

        for key, value in vars(Qt).iteritems():
            if isinstance(value, Qt.Key):
                self.keys[key] = value
                self.keys[value] = [key, 0]

###############################################################################
###############################################################################

    def keyReleaseEvent(self, event):
        if not event.isAutoRepeat():
            # self.updateStatus("Key Released: ")
            key = self.keys[event.key()]
            key[1] = 0
            self.keys[event.key()] = key
            # self.updateStatus(str(self.keys[event.key()]))

    def keyPressEvent(self, event):
        if not event.isAutoRepeat():
            key = self.keys[event.key()]
            key[1] = 1
            self.keys[event.key()] = key
            # self.updateStatus("Key Pressed: ")
            # self.updateStatus(str(self.keys[event.key()]))

    def labelDialog(self):
        text, result = QInputDialog.getText(self, 'Sematic label Name', 'Enter label name: ')
        if result == True:
            return str(text)

    def setDirty(self):
        """Call this method when applying changes"""
        self.dirty = True
        self.saveAction.setEnabled(True)

    def setClean(self):
        """Call this method when saving changes"""
        self.dirty = False
        self.saveAction.setEnabled(False)
        self.historyStack = []
        self.redoStack = []
        self.undoAction.setEnabled(False)
        self.redoAction.setEnabled(False)

    def createAction(self, text, slot=None, shortcut=None, icon=None, tip=None, checkable=False, signal="triggered()"):
        """Quickly create action"""
        action = QAction(text, self)
        if slot == self.chooseColor or slot == self.chooseOutlineColor or slot == self.chooseLabelColor:
            icon = QPixmap(50, 50)
            if slot == self.chooseColor:
                icon.fill(self.currentColor)
            elif slot == self.chooseOutlineColor:
                icon.fill(self.currentOutlineColor)
            elif slot == self.chooseLabelColor:
                icon.fill(self.currentLabelColor)
            action.setIcon(QIcon(icon))
        elif icon is not None:
            action.setIcon(QIcon(":/%s.png" % icon))
        if shortcut is not None:
            action.setShortcut(shortcut)
        if tip is not None:
            action.setToolTip(tip)
            action.setStatusTip(tip)
        if slot is not None:
            self.connect(action, SIGNAL(signal), slot)
        if checkable:
            action.setCheckable(True)
        return action

    def addActions(self, target, actions):
        """Add actions to menu bars or tool bars"""
        for action in actions:
            if action is None:
                target.addSeparator()
            else:
                target.addAction(action)

    def updateFileMenu(self):
        """Update file menu to show recent open files"""
        self.fileMenu.clear()
        self.addActions(self.fileMenu, self.fileMenuActions[:-1])
        current = QString(self.filename) \
                if self.filename is not None else None
        recentFiles = []
        for fname in self.recentFiles:
            if fname != current and QFile.exists(fname):
                recentFiles.append(fname)

        if recentFiles:
            self.fileMenu.addSeparator()
            for i, fname in enumerate(recentFiles):
                action = QAction(QIcon(":/file.png"), "&%d %s" %
                            (i + 1, QFileInfo(fname).filename()), self)
                action.setData(QVariant(fname))
                self.connect(action, SIGNAL("triggered()"),
                             self.loadImage)
                self.fileMenu.addAction(action)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.fileMenuActions[-1])

    def updateToolBar(self, updated=False):
        """Update toolbar to show color labels"""
        if self.filename == None:
            return
        if self.colourLabels is None or updated:
            self.colourLabels = config.getLabelColor(self.filename)

        if self.colourLabels is None:
            self.colorLabelBar.hide()
            self.labelAction.setEnabled(True)
        else:
            self.labelAction.setEnabled(False)
            self.labelAddAction.setEnabled(True)
            self.labelPaletteAction.setEnabled(True)
            self.colorLabelBar.clear()
            self.colorLabelBar.show()
            self.labelsGroup = QActionGroup(self)
            self.colorLabelDict = {}
            shortcuts = []
            for i in range(0, 10):
                for j in [0] + range(i + 1, 10):
                    if i == 0:
                        shortcuts.append(j)
                    else:
                        shortcuts.append([i, j])
            count = 0
            for label in self.colourLabels.keys():
                shortcut = "Shift+"
                if count < 10:
                    shortcut += str(shortcuts[count])
                else:
                    shortcut += str(shortcuts[count][0]) + "+" + str(shortcuts[count][1])

                count += 1
                action = self.createAction(label, self.chooseColor_2, shortcut,
                                           None, "Colour the label with user specified colour",
                                           True, "toggled(bool)")
                icon = QPixmap(50, 50)
                icon.fill(self.colourLabels[label])
                action.setIcon(QIcon(icon))
                self.colorLabelBar.addAction(action)
                self.labelsGroup.addAction(action)
                self.colorLabelDict[action] = self.colourLabels[label]

    def addRecentFile(self, fname):
        """Add files to recentfile array"""
        if fname is None:
            return
        if not self.recentFiles.contains(fname):
            self.recentFiles.prepend(QString(fname))
            while self.recentFiles.count() > 9:
                self.recentFiles.takeLast()

    def updateStatus(self, message):
        """Update message on status bar and window title"""
        self.statusBar().showMessage(message, 5000)
        self.listWidget.addItem(message)
        self.listWidget.scrollToBottom()
        if self.filename is not None:
            self.setWindowTitle("Image Annotation Tool - %s[*]" % \
                                os.path.basename(self.filename))
        else:
            self.setWindowTitle("Image Annotation Tool[*]")
        self.setWindowModified(self.dirty)
        print message

    def okToContinue(self):
        """Check if there is unsaved change"""
        if self.dirty:
            reply = QMessageBox.question(self,
                                         "Image Annotation Tool - Unsaved changes",
                                         "Do you want to save unsaved changes?",
                                         QMessageBox.Yes | QMessageBox.No |
                                         QMessageBox.Cancel)
            if reply == QMessageBox.Cancel:
                return False
            elif reply == QMessageBox.Yes:
                self.saveFile()

        return True

    def saveFile(self):
        """Save file to the directory which is specified in config.py"""
        if self.outputMask is None:
            print "Noting to save"
            return
        if self.spSegments is not None:
            path = config.outputFile(self.filename)
            dirSplit = path.split('.')
            np.savetxt(dirSplit[0] + ".csv", self.spSegments, delimiter=",", fmt="%d")
            
        output = cv2.cvtColor(self.outputMask, cv2.COLOR_RGB2BGR)
        cv2.imwrite(config.outputFile(self.filename).decode('utf-8').encode('gbk'), output)
        self.updateStatus("Save to %s" % config.outputFile(self.filename))
        self.setClean()

    def colorListWidget(self, current):
        """Use different color to label files in list:
        Red: Unlabelled. Green: Labelled. Blue: Current"""
        for i in range(0, len(self.allImages)):
            item = self.fileListWidget.item(i)
            if current == self.allImages[i]:
                item.setForeground(Qt.blue)
                item.setIcon(QIcon(":/writing.png"))
            elif os.path.exists(config.outputFile(self.allImages[i])):
                item.setForeground(Qt.green)
                item.setIcon(QIcon(":/done.png"))
            else:
                item.setForeground(Qt.red)
                item.setIcon(QIcon(":/delete.png"))

    def fileMove(self):
        self.okToContinue()
        path = config.outputFile(self.filename)
        path_split = path.split('\\')
        current_dir = "/".join(path_split[0:len(path_split)-2]) + "/"
        done_dir = current_dir + "done/"
        if not os.path.exists(done_dir):
            os.makedirs(done_dir)

        file_name_dir_split = self.filename.split("\\")
        file_name_w_ext = file_name_dir_split[-1]
        self.updateStatus(file_name_w_ext)
        file_name_w_ext_split = file_name_w_ext.split(".")

        os.rename(self.filename, done_dir + file_name_w_ext)

        current_pred = current_dir + file_name_w_ext_split[0] + "_suggested." + file_name_w_ext_split[-1]
        current_avg = current_dir + file_name_w_ext_split[0] + "_avg." + file_name_w_ext_split[-1]

        files = False

        if os.path.exists(current_pred):
            new_pred = done_dir + file_name_w_ext_split[0] + "_suggested." + file_name_w_ext_split[-1]
            os.rename(current_pred, new_pred)
            files = True

        if os.path.exists(current_avg):
            new_avg = done_dir + file_name_w_ext_split[0] + "_avg." + file_name_w_ext_split[-1]
            os.rename(current_avg, new_avg)
            files = True

        if files:
            self.updateStatus("Files moved to done")
        else:
            self.updateStatus("File moved to done")
        self.fileOpen()

    def dirOpen(self, fromVid=False, dirname=None, applySP = None):
        """Open a directory and load the first image"""
        if not self.okToContinue():
            return
        if dirname == None:
            dir = os.path.dirname(self.filename) \
                if self.filename is not None else "."
            dirname = unicode(QFileDialog.getExistingDirectory(self,
                                "Image Annotation Tool - Select Directory", dir))
        
        if dirname:
            dirname = str(dirname)
            dirname = dirname.replace("/","\\")
            image_video = True
            if not fromVid:
                dialog = OpeningDirDialog()
                answer = dialog.exec_()
                if answer == 0: #images
                    image_video = True
                elif answer == 1: #videos
                    image_video = False
                    fromVid = True
                else: # cancel
                    self.updateStatus("Open directory canceled")
                    return
                
            self.updateStatus("Open directory: %s" % dirname)
            self.filepath = dirname
            time.sleep(5)
            if image_video:
                # mass SP on load?
                if applySP == None:
                    msg = "Do you want to apply superpixels to entire directory? \nWarning this will delete any perviously labeled frames."
                    reply = QMessageBox.question(self, 'Message',
                            msg, QMessageBox.Yes, QMessageBox.No)
                    self.massSP = True if reply == QMessageBox.Yes else False
                else:
                    self.massSP = applySP

                self.allImages = self.scanAllImages(dirname)
                            
                self.fileListWidget.clear()
                if self.massSP and not self.spMassActive:
                    self.spMassTotal = len(self.allImages)
                    self.spMassComplete = 0
                    self.spMassActive = True
                    self.firstDone = False
                    self.updateStatus("SP progress: %d/%d" %(self.spMassComplete, self.spMassTotal))
                elif fromVid:
                    self.spMassTotal += len(self.allImages)
                    self.updateStatus("SP progress: %d/%d" %(self.spMassComplete, self.spMassTotal))
                else:
                    if self.massSP:
                        QMessageBox.warning(self, 'Warning', "Mass superpixel execution already running")
                    self.massSP = False
                
                count = 0
                if fromVid and self.massSP:
                    sleep = 5
                else:
                    sleep = 1
                for imgPath in self.allImages:
                    filename = os.path.basename(imgPath)
                    item = QListWidgetItem(filename)
                    self.fileListWidget.addItem(item)
                    if self.massSP:
                        self.q.put([imgPath, sleep])
                        if count > self.threadpool.maxThreadCount() and sleep != 0:
                            sleep = 0
                        else:
                            count += 1

                #
                #  first file
                if len(self.allImages) > 0:
                    if not self.massSP:
                        self.loadImage(self.allImages[0])
                        self.filename = self.allImages[0]
                        self.updateToolBar()
                        self.colorListWidget(self.allImages[0])                  
                else:
                    QMessageBox.warning(self, 'Error', "[ERROR]: No images in %s" % dirname)
                    self.spMassActive = False
                self.massSP = False
            else:
                dirname = str(dirname)
                self.allVideos = self.scanAllVideos(dirname)
                self.loadingVideoActive = False
                self.videoComplete =0
                self.videoTotal = len(self.allVideos)
                self.updateStatus("Video progress: %d/%d" %(self.videoComplete, self.videoTotal))
                for vidPath in self.allVideos:
                    self.loadVideo(vidPath, True)
            
    def scanAllImages(self, imageDir):
        """Get a list of file name of images in a directory"""
        extensions = [".%s" % format \
                      for format in QImageReader.supportedImageFormats()]
        images = []

        for root, dirs, files in os.walk(imageDir):
            for file in files:
                if file.lower().endswith(tuple(extensions)) and root == imageDir:
                    relativePath = os.path.join(root, file)
                    # path = ustr(os.path.abspath(relativePath))
                    images.append(relativePath)
        images.sort(key=lambda x: x.lower())
        return images
        
    def scanAllVideos(self, videoDir):
        """Get a list of file name of videos in a directory"""
        #extensions = [".%s" % format \ for format in QImageReader.supportedImageFormats()]
        
        extensions = [".mp4"]
        videos = []

        for root, dirs, files in os.walk(videoDir):
            for file in files:
                if file.lower().endswith(tuple(extensions)) and root == videoDir:
                    relativePath = os.path.join(root, file)
                    # path = ustr(os.path.abspath(relativePath))
                    videos.append(relativePath)
        videos.sort(key=lambda x: x.lower())
        return videos

    def fileItemDoubleClicked(self, item=None):
        """Open image if double-clickling the item on files dock"""
        if not self.okToContinue():
            return

        path = os.path.join(self.filepath, str(item.text()))
        currIndex = self.allImages.index(path)
        if currIndex < len(self.allImages):
            filename = self.allImages[currIndex]
            if filename and os.path.exists(filename):
                self.loadImage(filename)
                self.colorListWidget(filename)

    def fileOpen(self):
        """Open a file with file dialog"""
        if not self.okToContinue():
            return
            
        dir = os.path.dirname(self.filename) \
                if self.filename is not None else "."
        img_formats = ["*.%s" % unicode(format).lower() \
                   for format in QImageReader.supportedImageFormats()]
                   
        vid_formats = ["*.%s" % unicode(format).lower() \
                   for format in QMovie.supportedFormats()]
                   
        vid_formats = ["*.mp4"]
        
        #all_formats = []
        #all_formats.extend(img_formats)
        #all_formats.extend(vid_formats)
        
        
        fname = unicode(QFileDialog.getOpenFileName(self,
                            "Image Annotation Tool - Choose Image", dir,
                            "Image files (%s) ;; Video files (%s)" % (" ".join(img_formats), " ".join(vid_formats))))
                            #"All supported formats (%s) ;; Image files (%s) ;; Video files (%s)" % (" ".join(all_formats) ," ".join(img_formats), " ".join(vid_formats))))
        if fname.endswith(".mp4"):
            self.videoTotal = 1
            self.updateStatus("Video progress: %d/%d" %(self.videoComplete, self.videoTotal))
            self.loadVideo(fname)
        else:
            self.filename = fname.replace("/","\\")
            self.allImages = []
            self.allImages.append(self.filename)
            self.fileListWidget.clear()
            filename = os.path.basename(fname)
            item = QListWidgetItem(filename)
            self.fileListWidget.addItem(item)
            
            self.loadImage(fname)
            self.updateToolBar()
            self.colorListWidget(fname) 

    def loadVideo(self, fname=None, loadingMultiVideos = False):
        fsplit = fname.split("/")
        nsplit = fsplit[len(fsplit)-1].split(".")
        fsplit[len(fsplit)-1]=nsplit[0]
        dirname="/".join(fsplit) + "/"

        if not loadingMultiVideos or self.loadMultiVideoAns == []:
            dialog = LoadingVideoDialog()
            answer = dialog.exec_()
            self.loadMultiVideoAns = answer[1]

        reverse = self.loadMultiVideoAns[0]
        firstFrame = self.loadMultiVideoAns[1]
        applySP = self.loadMultiVideoAns[2]
        
        #msg = "Do you want to reverse the video?"
        #reply = QMessageBox.question(self, 'Message',
        #                msg, QMessageBox.Yes, QMessageBox.No)

        #reverse = True if reply == QMessageBox.Yes else False
        
        useDir = True
        if os.path.exists(dirname):
            msg = "Directory already exists. Do you want to use current existing directory? Warning could overwrite existing data!"
            reply = QMessageBox.question(self, 'Warning! Directory Exists',
                            msg, QMessageBox.Yes, QMessageBox.No)
            useDir = True if reply == QMessageBox.Yes else False
            if useDir:
                self.updateStatus("Directory used: %s" % dirname)
        else:
            os.makedirs(dirname)
            self.updateStatus("Directory created: %s" % dirname)
        
        if useDir:
            self.q_Video.put([fname, reverse, dirname, firstFrame])
        else:
            self.updateStatus("Action stopped: Please rename file.")
            return
        
        if dirname and not loadingMultiVideos:
            self.dirOpen(True, dirname, applySP)

    def runClusterAlg(self):
        if self.q_Cluster.qsize() == 0:
            self.q_Cluster.put([0])

    def runClusterQueue(self, q, progress_callback):
        while True:
            arg = q.get()
            self.openClusters()
            q.task_done()
            progress_callback.emit("")

    def runVideoQueue(self, q, progress_callback):
        while True:
            arg = q.get()
            self.openVideo(arg)
            q.task_done()  
            progress_callback.emit(str(arg[2]))
    
    def openVideo(self, arg):
        fname = arg[0]
        reverse = arg[1]
        dirname = arg[2]
        dirname = dirname.replace("/","\\")
        firstFrame = arg[3]
        
        vidcap = cv2.VideoCapture(fname)
        success,image = vidcap.read()
        success = True
        frames = []
        while success:
            success,image = vidcap.read()
            if success:
                #image2 = cv2.resize(image,(640,360), interpolation = cv2.INTER_CUBIC)
                frames.insert(len(frames),image)
        
        start = 0
        end = len(frames)-1
        step = 1
        name = 0;
        
        if reverse:
            #self.updateStatus("Video Reversed")
            start = end-1
            end = -1
            step = -1
        else: 
            i = 1        
            #self.updateStatus("Video Not Reversed")
        
        for i in range(start, end, step):
            split = dirname.split("\\")
            cv2.imwrite(dirname + split[len(split)-2] + "_%05d.jpg" % name, frames[i])
            if firstFrame:
                break
            name += 1

    def loadedCluster(self):
        self.updateStatus("Cluster Loaded")

    def loadedVideo(self, dir):
        self.mutex.lock()
        self.videoComplete += 1
        self.updateStatus("Video progress: %d/%d" %(self.videoComplete, self.videoTotal))
        self.mutex.unlock()
        self.dirOpen(True, dir, self.loadMultiVideoAns[2])

        if self.videoComplete == self.videoTotal:
            self.loadingVideoActive = False
            self.videoComplete = 0
            self.loadMultiVideoAns = []

    def loadImage(self, fname=None):
        """Load the newest image"""
        # If receiving signal from "recentFiles"
        if fname is None:
            action = self.sender()
            if isinstance(action,QAction):
                fname = unicode(action.data().toString())
                if not self.okToContinue():
                    return
                self.fileListWidget.clear()
            else:
                return

        if fname:
            # Read in as opencv image and then convert to QImage
            self.cvimage = cv2.imread(fname.decode('utf-8').encode('gbk'))
            if self.cvimage is None:
                message = "Failed to read %s" % fname
            else:
                if self.sp_cluster_active:
                    self.clusterDeactivate()
                self.sp_cluster_active = True
                # Create output image directory if not existing
                if not os.path.exists(config.outputDir(fname)):
                    os.makedirs(config.outputDir(fname))

                # Create a new output image directory if not existing
                if os.path.exists(config.outputFile(fname)):
                    self.outputMask = cv2.imread(
                                    config.outputFile(fname).decode('utf-8').encode('gbk'))
                else:
                    self.outputMask = np.zeros(self.cvimage.shape, np.uint8)
                    cv2.rectangle(self.outputMask, (0, 0),
                                  (self.cvimage.shape[1], self.cvimage.shape[0]),
                                  (self.backgroundColor.red(), self.backgroundColor.green(),
                                   self.backgroundColor.blue()), -1)
                
                dir = config.outputFile(fname)
                dirSplit = dir.split('.')
                if os.path.exists(dirSplit[0] + ".csv"):
                    self.spSegments = np.int64(np.genfromtxt(dirSplit[0] + ".csv", delimiter=','))
                    self.spMask = np.uint8(mark_boundaries(np.zeros(self.cvimage.shape, np.uint8),
                                                           self.spSegments, color=(1, 0, 0)))*255
                    self.spActivate()
                else:
                    self.spDeactivate()
                    self.spSegments = None
                    self.spMask = None

                self.addRecentFile(self.filename)
                self.sizeLabel.setText("Image size: %d x %d" %
                                       (self.cvimage.shape[1], self.cvimage.shape[0]))
                # Reset all actions to default
                for action, check in self.resetableActions:
                    action.setChecked(check)
                # Convert to RGB color space
                cv2.cvtColor(self.cvimage, cv2.COLOR_BGR2RGB, self.cvimage)
                cv2.cvtColor(self.outputMask, cv2.COLOR_BGR2RGB, self.outputMask)
                self.isLoading = True
                self.showImage()
                self.filename = fname.replace("/", "\\")
                self.filepath = os.path.dirname(self.filename)
                self.allImages = []
                self.allImages.append(self.filename)
                self.fileListWidget.clear()
                filename = os.path.basename(fname)
                item = QListWidgetItem(filename)
                self.fileListWidget.addItem(item)
                self.dirty = False

                message = "Loaded %s" % os.path.basename(fname)
            self.updateStatus(message)
            self.setClean()

    def showImage(self, percent=None):
        """Transfer opencv image into QImage and update to draw on imagelabel"""
        if self.cvimage is None:
            return

        height, width, bytesPerComponent = self.cvimage.shape
        bytesPerLine = bytesPerComponent * width

        if self.hideOriginalAction.isChecked():
            if self.floodFillAction.isChecked():
                self.displayImage = self.applyMask()
            else:
                self.displayImage = self.outputMask
        # If current operation is loading a new image, convert opencv color space
        # to qimage and auto-zoom the image
        elif self.isLoading:
            self.displayImage = self.applyMask()
            self.isLoading = False

            heightRatio = self.scrollArea.height() * 1.00 / height
            widthRatio = self.scrollArea.width() * 1.00 / width
            if heightRatio > widthRatio:
                self.zoomSpinBox.setValue(widthRatio * 100)
            else:
                self.zoomSpinBox.setValue(heightRatio * 100)
        # Neither hiding original image, nor loading new image
        else:
            self.displayImage = self.applyMask()

        if percent is None:
            percent = self.zoomSpinBox.value()

        factor = percent / 100.0
        self.image = QImage(self.displayImage.data, width, height, bytesPerLine, QImage.Format_RGB888)

        width = self.image.width() * factor
        height = self.image.height() * factor
        self.image = self.image.scaled(width, height, Qt.KeepAspectRatio)
        self.imageLabel.setMinimumSize(width, height)
        self.imageLabel.setMaximumSize(width, height)
        self.imageLabel.update()

    def applyMask(self):
        """Apply mask to origin image and get the displayable image"""
        dst = self.cvimage
        inverted = False

        if self.hideImg:
            return self.outputMask
        
        gray_output = cv2.cvtColor(self.outputMask, cv2.COLOR_RGB2GRAY)
        ret, mask_output = cv2.threshold(gray_output, 2, 255, cv2.THRESH_BINARY)
        masked_output = cv2.bitwise_and(self.outputMask, self.outputMask, mask=mask_output)
        
        if (masked_output != 0).any() and not self.hideMask:
            inverted = True
            masked_image_output = cv2.bitwise_and(self.cvimage, self.cvimage, mask=mask_output)
            temp = cv2.addWeighted(masked_output, self.alphaMaskNum, masked_image_output, 1.0-self.alphaMaskNum, 0)
            origin = cv2.bitwise_and(self.cvimage, self.cvimage, mask=cv2.bitwise_not(mask_output))
            dst = cv2.add(temp, origin)
            
        if self.spMask is not None and not self.hideSP:  # and show sp
            # inverted = True
            # io.imsave("self.spMask.png",self.spMask)
            # gray_sp = cv2.cvtColor(self.spMask, cv2.COLOR_RGB2GRAY)
            # ret, mask_sp = cv2.threshold(gray_sp, 2, 255, cv2.THRESH_BINARY)
            # mask_sp_inverted = cv2.bitwise_not(mask_sp)
            # masked_sp = cv2.bitwise_and(self.spMask, self.spMask, mask=mask_sp)
            # masked_out_sp = cv2.bitwise_and(dst, dst, mask=mask_sp_inverted)
            # dst = cv2.add(masked_sp, masked_out_sp)
            output = dst.copy()
            colour = [float(self.currentOutlineColor.red()) / 255.0,
                      float(self.currentOutlineColor.green()) / 255.0,
                      float(self.currentOutlineColor.blue()) / 255.0]
            boundaries = mark_boundaries(output, self.spSegments, colour)
            boundaries_2 = boundaries * 255
            boundaries_3 = boundaries_2.astype(np.uint8)
            dst = boundaries_3

        if self.regionSegments is not None and not self.hideCluster and not self.showSuggestedCluster:
            # change to overlay
            output = dst.copy()
            cv2.addWeighted(self.clusterMask, self.alphaClusterNum, output, 1.0 - self.alphaClusterNum, 0, output)
            dst = output

        if self.suggestedClusterMask is not None and self.showSuggestedCluster:  # and show suggested cluster only
            output = dst.copy()
            cv2.addWeighted(self.suggestedClusterMask, self.alphaClusterNum, output, 1.0 - self.alphaClusterNum, 0, output)
            dst = output

        if self.suggestedClusterMask is not None and self.showClusterOutlines:
            output = dst.copy()
            colour = [float(self.currentOutlineColor.red()) / 255.0,
                      float(self.currentOutlineColor.green()) / 255.0,
                      float(self.currentOutlineColor.blue()) / 255.0]
            boundaries = mark_boundaries(output, self.regionSegments, colour)
            # boundaries_2 = self.normalise(boundaries, 255)
            boundaries_2 = boundaries * 255
            boundaries_3 = boundaries_2.astype(np.uint8)

            dst = boundaries_3
            # io.imsave("regionSegments.png", self.regionSegments)
            # dst = mark_boundaries(self.regionSegments, self.ioimage)
            # dst = self.regionSegments

        if self.choosingPointFF:
            factor = self.chooseFFPointSpinBoxValue * 1.0 / 100
            x = int(round(self.pointFF.x() / factor))
            y = int(round(self.pointFF.y() / factor))
            seed_pt = x, y
            connectivity = config.connectivity_Floodfill
            fixed_range = config.fixed_range
            flags = connectivity
            if fixed_range:
                flags |= cv2.FLOODFILL_FIXED_RANGE
            img = self.cvimage.copy()
            self.maskFF = self.createFloodFillMask()
            cv2.floodFill(img, self.maskFF, seed_pt, (255, 255, 255),
                          (self.blueDiff, self.greenDiff, self.redDiff),
                          (self.blueDiff, self.greenDiff, self.redDiff), flags)
            height, width = self.cvimage.shape[:2]
            subtract = cv2.subtract(img, self.cvimage)
            grayimage = cv2.cvtColor(subtract, cv2.COLOR_RGB2GRAY)
            ret, mask = cv2.threshold(grayimage, 1, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            singlecolor = np.zeros((height, width, 3), np.uint8) # Create a single color image to specify the FF area
            singlecolor[:, :] = [self.currentColor.red(), self.currentColor.green(), self.currentColor.blue()]
            area_FloodFill = cv2.bitwise_and(singlecolor, singlecolor, mask=mask)
            if not self.hideImg:
                change_origin = cv2.bitwise_and(self.cvimage, self.cvimage, mask=mask_inv)
                dst = cv2.add(change_origin, area_FloodFill)

        return dst

    def undo(self):
        """Undo the last changes to the image"""
        self.redoStack.append(self.outputMask.copy())
        self.redoAction.setEnabled(True)
        old = self.historyStack.pop(-1)
        self.outputMask = old
        self.showImage()
        self.updateStatus("Undo")
        if len(self.historyStack) == 0:
            self.undoAction.setEnabled(False)

    def redo(self):
        """Undo the last changes to the image"""
        self.historyStack.append(self.outputMask.copy())
        self.undoAction.setEnabled(True)
        new = self.redoStack.pop(-1)
        self.outputMask = new
        self.showImage()
        self.updateStatus("Re-do")
        if len(self.redoStack) == 0:
            self.redoAction.setEnabled(False)

    def updateAlphaMaskNum(self):
        self.alphaMaskNum = self.alphaMaskSpinBox.value()
        self.updateStatus("Alpha Mask: " + str(self.alphaMaskNum))
        self.showImage()

    def updateAlphaClusterNum(self):
        self.alphaClusterNum = self.alphaClusterSpinBox.value()
        self.updateStatus("Alpha Cluster: " + str(self.alphaClusterNum))
        self.showImage()

    def updateSPNum(self):
        self.spNum = self.spSpinBox.value()
        self.updateStatus("# Superpixels: " + str(self.spNum))
        self.spAction.setEnabled(True)

    def updateSPSigma(self):
        self.sigma = self.sigmaSpinBox.value()
        self.updateStatus("Sigma: " + str(self.sigma))
        self.spAction.setEnabled(True)

    def updateSPCompactness(self):
        if self.compSpinBox.value == 0.0:
            self.compSpinBox.value = 10.0
        self.compactness = self.compSpinBox.value()
        self.updateStatus("Compactness: " + str(self.compactness))
        self.spAction.setEnabled(True)

    def runMassSuperpixelQueue(self, q, progress_callback):
        while True:
            self.runMassSuperpixelAlg(q.get())
            q.task_done()
            progress_callback.emit("")

    def loadFirstSPImage(self, null):
        self.mutex.lock()
        self.spMassComplete += 1
        self.updateStatus("SP progress: %d/%d" %(self.spMassComplete, self.spMassTotal))
        self.mutex.unlock()
        self.sp_cluster_active = self.clusterAction.isEnabled()
        
        if self.spMassComplete == self.spMassTotal:
            self.spMassActive = False
            self.spMassComplete = 0
        
        if not self.firstDone:
            self.firstDone = True
            time.sleep(5)
            if not len(self.allImages) == 0:
                self.loadImage(self.allImages[0])
            else:
                self.loadImage(self.filename)
            self.spActivate()

    def runMassSuperpixelAlg(self, arg):
        dir = arg[0]
        time.sleep(arg[1])
        
        img = cv2.imread(dir.decode('utf-8').encode('gbk'))
        if not os.path.exists(config.outputDir(dir)):
            os.makedirs(config.outputDir(dir))
            
        output = np.zeros(img.shape, np.uint8)
        segments = slic(img, n_segments=self.spNum, sigma=self.sigma, compactness=self.compactness)
        self.updateStatus("Actual # Superpixels: " + str(segments.max()))
        path = config.outputFile(dir)
        pathSplit = path.split('.')
        np.savetxt(pathSplit[0] + ".csv", segments, delimiter=",", fmt="%d")        
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(config.outputFile(dir).decode('utf-8').encode('gbk'), output)

    def runSuperpixelAlg(self):
        if self.dirty:
            if not self.okToContinue():
                self.updateStatus("Superpixels canceled")
                return

        self.firstDone = False
        self.q.put([self.filename, 0])
        self.spMassTotal = len(self.allImages)
        self.spMassComplete = 0
        self.spMassActive = True
        self.updateStatus("SP progress: %d/%d" %(self.spMassComplete, self.spMassTotal))

    def hideButtonClick(self):
        sending_button = self.sender()
        button_text = str(sending_button.iconText())
        button_text_split = button_text.split('\n')
        """Handle Hide button clicks"""
        if button_text_split[-1] == "Image" or button_text_split[-1] == "Superpixels" or button_text_split[-1] == "Mask":
            self.hideImg = self.hideOriginalAction.isChecked()
            self.hideSP = self.hidespAction.isChecked()
            self.hideMask = self.hideMaskAction.isChecked()
        elif button_text_split[0] == "Hide":
            if self.hideClusterAction.isChecked():
                self.hideCluster = True
                self.showSuggestedCluster = False
                self.showSuggestedClusterAction.setChecked(False)
                # self.showClusterOutlines = False
                # self.showClusterOutlinesAction.setChecked(False)
            else:
                self.hideCluster = False

        elif button_text_split[-1] == "Cluster":  # show suggested cluster
            if self.showSuggestedClusterAction.isChecked():
                self.showSuggestedCluster = True
                self.showClusterOutlines = False
                self.showClusterOutlinesAction.setChecked(False)
                self.hideCluster = False
                self.hideClusterAction.setChecked(False)
            else:
                self.showSuggestedCluster = False

        elif button_text_split[-1] == "Outline":  # show cluster outline
            if self.showClusterOutlinesAction.isChecked():
                self.clusterPaletteAction.setEnabled(True)
                self.showClusterOutlines = True
                self.showSuggestedCluster = False
                self.showSuggestedClusterAction.setChecked(False)
                # self.hideCluster = False
                # self.hideClusterAction.setChecked(False)
            else:
                self.showClusterOutlines = False
                self.clusterPaletteAction.setEnabled(False)
            
        self.showImage()

    def paintLabel(self, event):
        """First paint image, then paint label"""
        qp = QPainter(self.imageLabel)
        if self.choosingPointFF:
            qb = QBrush(QColor(255, 255, 255, 0))
        else:
            qb = QBrush(QColor(self.currentColor.red(), self.currentColor.green(),
                           self.currentColor.blue(), 150))
        qp.setBrush(qb)
        qp.setPen(Qt.DashLine)
        qp.drawPixmap(0, 0, QPixmap.fromImage(self.image))

        if self.begin is None:
            return

        if not self.mouseAction.isChecked():
            old_factor = self.lastSpinboxValue * 1.0 / 100
            factor = self.zoomSpinBox.value() * 1.0 / 100
            pointA = QPointF(self.begin.x() / old_factor * factor,
                             self.begin.y() / old_factor * factor)
            pointB = QPointF(self.end.x() / old_factor * factor,
                             self.end.y() / old_factor * factor)

        if self.rectLabelAction.isChecked() and self.isLaballing:
            qp.drawRect(QRectF(pointA, pointB))
        elif self.ellipseLabelAction.isChecked() and self.isLaballing:
            qp.drawEllipse(QRectF(pointA, pointB))
        elif self.polygonLabelAction.isChecked() and self.isLaballing:
            qp.setPen(self.thickPen)
            qp.drawLine(QLineF(pointA, pointB))
            for line in self.lines:
                qp.drawLine(line)
            if self.finishChoosingArea:
                adjustedPoints = []
                for point in self.points:
                    adjustedPoint = QPointF(point.x() / old_factor * factor,
                                            point.y() / old_factor * factor)
                    adjustedPoints.append(adjustedPoint)
                qp.drawPolygon(QPolygonF(adjustedPoints[1:]))

    def spActivate(self):
        self.spActive = True
        self.spMouseAction.setEnabled(True)
        self.spAddAction.setEnabled(True)
        # self.spSubAction.setEnabled(True)
        self.hidespAction.setEnabled(True)
        # self.spAction.setEnabled(False)
        self.spAddAction.setChecked(True)
        self.clusterPaletteAction.setEnabled(True)

    def spDeactivate(self):
        self.spActive = False
        self.spMouseAction.setEnabled(False)
        self.spAddAction.setEnabled(False)
        # self.spSubAction.setEnabled(False)
        self.hidespAction.setEnabled(False)
        self.spAction.setEnabled(True)
        self.spMouseAction.setChecked(True)
        self.clusterPaletteAction.setEnabled(False)

    def finishAreaChoosing(self):
        """Finish choosing the area, and then enable three editing choices"""
        self.finishChoosingArea = True
        self.confirmAction.setEnabled(True)
        self.deleteAction.setEnabled(True)
        self.floodFillAction.setEnabled(True)

    def notFinishAreaChoosing(self):
        """Not finish choosing areas"""
        self.finishChoosingArea = False
        self.confirmAction.setEnabled(False)
        self.deleteAction.setEnabled(False)
        self.floodFillAction.setChecked(False)
        self.floodFillAction.setEnabled(False)

    def labelPolygon(self):
        """Set mouse action when labelling polygon"""
        if self.sender().isChecked():
            self.spMouseAction.setChecked(True)
            self.setMouseAction()
            
            self.isLaballing = False
            self.notFinishAreaChoosing()
            self.showImage()
            self.imageLabel.mousePressEvent = self.startPoly
            self.imageLabel.mouseMoveEvent = self.doLabel
            self.imageLabel.mouseReleaseEvent = self.mouseReleasePoly
            # self.imageLabel.mouseDoubleClickEvent = self.finishPoly
            self.begin = None

    def mouseReleasePoly(self, event):
        pass

    def startPoly(self, event):
        if self.keys[self.keys["Key_Alt"]][1] == 1:
            self.pickColor(event.pos())
            return
        """Start labelling polygon"""
        self.imageLabel.setMouseTracking(True)
        self.lastSpinboxValue = self.zoomSpinBox.value()

        if self.begin is None:
            self.begin = event.pos()
            self.end = event.pos()
            self.points.append(self.begin)

        if self.finishChoosingArea:
            self.lines = []
            self.points = []
            self.begin = event.pos()
            self.points.append(self.begin)
            self.finishChoosingArea = False

        self.notFinishAreaChoosing()
        self.isLaballing = True
        key = self.keys[self.keys["Key_Escape"]]
        if not key[1] == 1:
            # key = self.keys[self.keys["Key_Control"]]
            # to_pop = key[1] == 1
            to_pop = event.button() == 2
            self.end = event.pos()

            if not to_pop:
                if not self.end == self.begin:
                    self.points.append(self.begin)

                if len(self.points) > 1:
                    self.lines.append(QLine(self.begin, self.end))

                self.begin = event.pos()
            else:
                if len(self.lines) > 0:
                    self.begin = self.lines[-1].p1()
                    self.lines.pop()

                if len(self.points) > 0 and (self.begin == self.points[-1] or len(self.lines == 0)):
                    self.points.pop()

                if len(self.points) == 0:
                    self.begin = None
                    self.end = None
        else:
            self.lines = []
            self.points = []
            self.begin = None
            self.end = None

        if self.begin is not None:
            self.imageLabel.update()
        # if event.button() == 2:
        key = self.keys[self.keys["Key_Control"]]
        if key[1] == 1:
            self.finishPoly(event)

    def finishPoly(self, event):
        """Finish labelling polygon"""
        self.imageLabel.setMouseTracking(False)
        self.finishAreaChoosing()
        self.points.append(event.pos())
        self.lines = []
        self.imageLabel.update()
        self.updateStatus("Choose an irregular polygon area")

    def setMouseAction(self):
        """Set mouse action when not labelling"""
        if self.sender().isChecked():
            self.isLaballing = False
            self.notFinishAreaChoosing()
            self.showImage()
            self.imageLabel.setMouseTracking(False)
            self.lines = []
            self.points = []
            self.imageLabel.mousePressEvent = self.pressImage
            self.imageLabel.mouseMoveEvent = self.moveImage
            self.imageLabel.mouseReleaseEvent = self.finishMove

    def pressImage(self, event):
        if event.button() == Qt.LeftButton:
            self.origin = QPoint(event.pos())

    def moveImage(self, event):
        """Move image with mouse dragging"""
        if not self.origin.isNull():
            changeX = self.origin.x() - event.pos().x()
            changeY = self.origin.y() - event.pos().y()
            self.scrollArea.verticalScrollBar().setValue(
                    self.scrollArea.verticalScrollBar().value() + changeY)
            self.scrollArea.horizontalScrollBar().setValue(
                    self.scrollArea.horizontalScrollBar().value() + changeX)

    def finishMove(self, event):
        pass

    def labelColourExists(self):
        for label in self.colourLabels.keys():

            if self.colourLabels[label] == QColor(self.currentLabelColor.red(),
                                                     self.currentLabelColor.green(),
                                                     self.currentLabelColor.blue()):
                self.updateStatus("Please choose another colour\nas this on is already in use")
                return True
        return False

    def labelNameExists(self):
        for label in self.colourLabels.keys():

            if label == self.labelName:
                self.updateStatus("Please choose another name\nas this on is already in use")
                return True
        return False

    def labelToFileAdd(self):
        if self.currentLabelColor is None:
            self.updateStatus("Please select a label colour.")
            return

        if self.labelColourExists():
            return

        run = True
        while run:
            self.labelName = self.labelDialog()
            run = self.labelNameExists()
        if not self.labelName is None and len(self.labelName) > 0:
            imgRoot = os.path.dirname(self.filename)
            path = os.path.join(imgRoot, "label.txt")
            self.append_to_csv(path, [self.currentLabelColor.red(),
                                      self.currentLabelColor.green(),
                                      self.currentLabelColor.blue(),
                                      self.labelName])
            self.updateToolBar(updated=True)
        return

    def openLabels(self):
        self.updateStatus(self.filename)
        self.colourLabels = config.getLabelColor(self.filename)
        if self.colourLabels is None:
            imgRoot = os.path.dirname(self.filename)
            path = os.path.join(imgRoot, "label.txt")
            self.append_to_csv(path, [0, 0, 0, "unknown"])
        self.updateToolBar()

    def clusterActivate(self):
        self.clusterActive = True
        self.clusterMouseAction.setEnabled(True)
        # self.clusterSubAction.setEnabled(True)
        self.clusterAddAction.setEnabled(True)
        self.hideClusterAction.setEnabled(True)
        self.showSuggestedClusterAction.setEnabled(True)
        self.showClusterOutlinesAction.setEnabled(True)
        self.clusterAction.setEnabled(False)
        self.clusterAddAction.setChecked(True)

    def clusterDeactivate(self):
        self.clusterActive = False
        self.clusterMouseAction.setEnabled(False)
        self.clusterAddAction.setEnabled(False)
        # self.clusterSubAction.setEnabled(False)
        self.hideClusterAction.setEnabled(False)
        self.showSuggestedClusterAction.setEnabled(False)
        self.clusterPaletteAction.setEnabled(False)
        self.showClusterOutlinesAction.setEnabled(False)
        self.clusterAction.setEnabled(True)
        self.clusterMouseAction.setChecked(True)
        self.clusterMask = None
        self.suggestedClusterMask = None
        self.showSuggestedCluster = False
        self.showClusterOutlines = False
        self.hideCluster = False
        self.regionSegments = None

    def openClusters(self):
        if self.filename is None:
            return
        self.updateStatus("Opening Cluster")
        split_file_dir = self.filename.split('.')
        avg_path = split_file_dir[0] + "_avg." + split_file_dir[1]
        pred_path = split_file_dir[0] + "_suggested." + split_file_dir[1]
        if not QFile.exists(avg_path):
            self.updateStatus("Average superpixel image does not exist: " + avg_path)
            return
        if not QFile.exists(pred_path):
            self.updateStatus("Predication output image does not exist: " + pred_path)
            return
        avg = io.imread(avg_path)
        pred = io.imread(pred_path)
        pred_2 = self.normalise(pred, 255)
        pred_2 = pred_2.astype(int)
        self.clusterMask = avg
        self.suggestedClusterMask = np.zeros(avg.shape)
        # self.spMask = avg
        self.regionSegments = self.avgToSegments(avg)
        self.segMask = None
        # for i in range(self.outputMask.shape[2]):
            # self.outputMask[:, :, i] = pred
        self.suggestedClusterMask[:, :, 1] = pred_2

        self.suggestedClusterMask = self.suggestedClusterMask.astype(np.uint8)

        # for i in range(self.clusterMask.shape[0]):
        #     for j in range(self.clusterMask.shape[1]):
        #         if not np.all(self.clusterMask[i][j] == 0):
        #             print self.clusterMask[i][j]
        self.clusterActivate()
        self.confirmEdit()
        self.hideClusterAction.setChecked(False)
        self.showSuggestedClusterAction.setChecked(False)
        self.showClusterOutlinesAction.setChecked(False)

    def findLabel(self, x, y):
        if x == 0 and y == 0:
            return 1
        # print x, y
        if not self.segMask[x][y] == 0:
            return self.segMask[x][y]
        same = []
        # left
        if (not x == 0) and \
            not self.segMask[x-1][y] == 0 and \
            np.all(self.segMaskAvg[x][y] == self.segMaskAvg[x-1][y]):
                same.append([-1, 0, self.segMask[x-1][y]])
        # upper left
        if (not x == 0 and not y == 0) and \
            not self.segMask[x - 1][y - 1] == 0 and \
            np.all(self.segMaskAvg[x][y] == self.segMaskAvg[x - 1][y - 1]):
                same.append([-1, -1, self.segMask[x-1][y-1]])
        # up
        if (not y == 0) and \
            not self.segMask[x][y-1] == 0 and \
            np.all(self.segMaskAvg[x][y] == self.segMaskAvg[x][y-1]):
                same.append([0, -1, self.segMask[x][y-1]])
        # upper right
        if (not x == self.segMaskAvg.shape[0]-1 and not y == 0) and \
            not self.segMask[x + 1][y - 1] == 0 and \
            np.all(self.segMaskAvg[x][y] == self.segMaskAvg[x + 1][y - 1]):
                same.append([1, -1, self.segMask[x+1][y-1]])

        # right
        if (not x == self.segMaskAvg.shape[0]-1) and \
            not self.segMask[x + 1][y] == 0 and \
            np.all(self.segMaskAvg[x][y] == self.segMaskAvg[x + 1][y]):
                same.append([1, 0, self.segMask[x+1][y]])
        # lower right
        if (not x == self.segMaskAvg.shape[0] - 1 and not y == self.segMaskAvg.shape[1] - 1) and \
            not self.segMask[x + 1][y + 1] == 0 and \
            np.all(self.segMaskAvg[x][y] == self.segMaskAvg[x+1][y+1]):
                same.append([1, 1, self.segMask[x+1][y+1]])
        # down
        if (not y == self.segMaskAvg.shape[1] - 1) and \
            not self.segMask[x][y + 1] == 0 and \
            np.all(self.segMaskAvg[x][y] == self.segMaskAvg[x][y+1]):
                same.append([0, 1, self.segMask[x][y+1]])
        # lower left
        if (not x == 0 and not y == self.segMaskAvg.shape[1] - 1) and \
            not self.segMask[x - 1][y + 1] == 0 and \
            np.all(self.segMaskAvg[x][y] == self.segMaskAvg[x - 1][y + 1]):
            same.append([-1, 1, self.segMask[x-1][y+1]])

        if len(same) == 0:
            label = self.avgNextLabel
            self.avgNextLabel += 1
            return label
        same = np.array(same)
        labels = same[:, 2]
        if np.all(labels == labels[0]):
            return int(labels[0])

        mask = self.segMask.reshape(self.segMask.shape[0]*self.segMask.shape[1])
        sorted_labels = np.sort(labels)
        for i in range(1, len(sorted_labels)):
            label = sorted_labels[i]
            indices = np.where(mask == label)
            for index in indices:
                mask[index] = sorted_labels[0]

        self.segMask = mask.reshape(self.segMask.shape[0], self.segMask.shape[1])
        return int(sorted_labels[0])

    def avgToSemgentHelper(self):
        for y in range(0, self.segMaskAvg.shape[1]):
            for x in range(0, self.segMaskAvg.shape[0]):
                if x == 0 and y == 0:
                    self.segMask[x][y] = 1
                    continue
                # if x < self.segMaskAvg.shape[0]:
                self.segMask[x][y] = self.findLabel(x, y)

    def avgToSegments(self, avg):
        avg_2 = avg.reshape(avg.shape[0] * avg.shape[1], avg.shape[2])
        # avg_3 = np.unique(avg_2, axis=0)
        self.segMask = np.zeros((avg.shape[0], avg.shape[1]))
        self.avgNextLabel = 2
        self.segMaskAvg = avg.copy()
        self.avgToSemgentHelper()
        return self.segMask.astype(int)

    def labelClusterAdd(self):
        """Set mouse action when adding to superpixel segments"""
        if self.sender().isChecked():
            self.mouseAction.setChecked(True)
            self.setMouseAction()

            if self.spAddAction.isEnabled():
                self.spAddAction.setChecked(False)
                # self.spSubAction.setChecked(False)
                self.spMouseAction.setChecked(True)

            self.isLaballing = False
            self.notFinishAreaChoosing()
            self.showImage()
            self.imageLabel.mousePressEvent = self.startClusterAdd
            self.imageLabel.mouseMoveEvent = self.DragClusterADD
            self.imageLabel.mouseReleaseEvent = self.stopClusterAdd

    def addCluster(self):
        label, x, y = self.getLabel(self.clusterPosition)
        if label is None:
            return
        # print self.clusterPosition.x(), self.clusterPosition.y()
        for i in range(len(self.cluster_queue)):
            if self.cluster_queue[i] == label:
                return
        self.cluster_queue.append(label)
        self.confirmEdit()
        self.showImage()

    def startClusterAdd(self, event):
        """Start labelling cluster"""
        if self.keys[self.keys["Key_Alt"]][1] == 1:
            self.pickColor(event.pos())
            return
        self.clusterButton = event.button()
        self.imageLabel.setMouseTracking(True)
        self.isLaballing = True
        self.clusterPosition = event.pos()
        self.addCluster()

    def DragClusterADD(self, event):
        if self.keys[self.keys["Key_Alt"]][1] == 1:
            return
        self.clusterPosition = event.pos()
        self.addCluster()

    def stopClusterAdd(self, event):
        self.imageLabel.setMouseTracking(False)
        self.cluster_queue = []
        self.clusterButton = 0
        #self.confirmEdit()
        #self.showImage()

    def getColour(self, pos):
        factor = self.zoomSpinBox.value() * 1.0 / 100.0
        x = int(round(pos.x() / factor))
        y = int(round(pos.y() / factor))
        return self.outputMask[y][x]

    def getLabel(self, pos):
        factor = self.zoomSpinBox.value() * 1.0 / 100.0
        x = int(round(pos.x() / factor))
        y = int(round(pos.y() / factor))
        if x < 0 or x > self.spSegments.shape[1] or y < 0 or y > self.spSegments.shape[0]:
            return None, None, None
        if not self.spButton == 0:
            label = self.spSegments[y][x]
        elif not self.clusterButton == 0:
            label = self.regionSegments[y][x]
        return label, x, y

    def addSP(self):
        label, x, y = self.getLabel(self.spPosition)
        if label is None:
            return
        for i in range(len(self.sp_queue)):
            if self.sp_queue[i] == label:
                return
        self.sp_queue.append(label)
        self.confirmEdit()
        self.showImage()

    def startSPAdd(self, event):
        """Start labelling sp"""
        self.isLaballing = True
        if self.keys[self.keys["Key_Control"]][1] == 1:
            self.floodMaskInit(event.pos())
            return
        if self.keys[self.keys["Key_Alt"]][1] == 1:
            self.pickColor(event.pos())
            return

        self.spButton = event.button()

        self.imageLabel.setMouseTracking(True)
        self.lastSpinboxValue = self.zoomSpinBox.value()
        self.spPosition = event.pos()
        self.addSP()

    def DragSPADD(self, event):
        if self.keys[self.keys["Key_Alt"]][1] == 1 or self.keys[self.keys["Key_Control"]][1] == 1:
            return

        self.spPosition = event.pos()
        self.addSP()

    def stopSPAdd(self, event):
        """Finish labelling sp"""
        self.imageLabel.setMouseTracking(False)
        self.sp_queue = []
        self.spButton = 0
        #self.confirmEdit()
        #self.showImage()

    def labelSPAdd(self):
        """Set mouse action when adding to superpixel segments"""
        if self.sender().isChecked():
            self.mouseAction.setChecked(True)
            self.setMouseAction()

            if self.clusterAddAction.isEnabled():
                self.clusterAddAction.setChecked(False)
                # self.clusterSubAction.setChecked(False)
                self.clusterMouseAction.setChecked(True)


            self.isLaballing = False
            self.notFinishAreaChoosing()
            self.showImage()
            self.imageLabel.mousePressEvent = self.startSPAdd
            self.imageLabel.mouseMoveEvent = self.DragSPADD
            self.imageLabel.mouseReleaseEvent = self.stopSPAdd
    
    def labelRectOrEllipse(self):
        """Set mouse action when labelling rectangle or ellipse"""
        if self.sender().isChecked():
            self.spMouseAction.setChecked(True)
            self.setMouseAction()
            
            self.isLaballing = False
            self.notFinishAreaChoosing()
            self.showImage()
            self.imageLabel.setMouseTracking(False)
            self.lines = []
            self.points = []
            self.imageLabel.mousePressEvent = self.startLabel
            self.imageLabel.mouseMoveEvent = self.doLabel
            self.imageLabel.mouseReleaseEvent = self.finishLabel

    def startLabel(self, event):
        """Start labelling rectangle or ecllipse"""
        if self.keys[self.keys["Key_Alt"]][1] == 1:
            self.pickColor(event.pos())
            return
        self.isLaballing = True
        self.notFinishAreaChoosing()
        self.lastSpinboxValue = self.zoomSpinBox.value()
        self.begin = event.pos()
        self.end = event.pos()
        self.imageLabel.update()

    def doLabel(self, event):
        """Choose the area of rectangle or ecllipse"""
        x = event.pos().x()
        y = event.pos().y()
        if x > self.scrollArea.horizontalScrollBar().value() + self.scrollArea.width():
            self.scrollArea.horizontalScrollBar().setValue(
                self.scrollArea.horizontalScrollBar().value() + 5)
        elif x < self.scrollArea.horizontalScrollBar().value():
            self.scrollArea.horizontalScrollBar().setValue(
                self.scrollArea.horizontalScrollBar().value() - 5)
        if y > self.scrollArea.verticalScrollBar().value() + self.scrollArea.height():
            self.scrollArea.verticalScrollBar().setValue(
                self.scrollArea.verticalScrollBar().value() + 5)
        elif y < self.scrollArea.verticalScrollBar().value():
            self.scrollArea.verticalScrollBar().setValue(
                self.scrollArea.verticalScrollBar().value() - 5)

        if x > self.image.width():
            x = self.image.width()
        elif x < 0:
            x = 0
        if y > self.image.height():
            y = self.image.height()
        elif y < 0:
            y = 0
        self.end = QPoint(x, y)
        self.imageLabel.update()

    def finishLabel(self, event):
        """Finish labelling rectangle or ecllipse"""
        if self.rectLabelAction.isChecked():
            self.updateStatus("Choose a rectangle area")
        elif self.ellipseLabelAction.isChecked():
            self.updateStatus("Choose an ellipse area")
        self.finishAreaChoosing()

    def confirmEdit(self):
        """Confirm the change you just made"""
        if not self.isLaballing:
            return

        copy = self.outputMask.copy()
        self.historyStack.append(copy)
        self.redoStack = []
        self.redoAction.setEnabled(False)

        factor = self.lastSpinboxValue * 1.0 / 100
        if self.rectLabelAction.isChecked() or self.ellipseLabelAction.isChecked():
            topleft_x = int(round(self.begin.x() / factor))
            topleft_y = int(round(self.begin.y() / factor))
            bottomright_x = int(round(self.end.x() / factor))
            bottomright_y = int(round(self.end.y() / factor))

        if self.floodFillAction.isChecked():
            self.choosingPointFF = False

            factor = self.chooseFFPointSpinBoxValue * 1.0 / 100
            x = int(round(self.pointFF.x() / factor))
            y = int(round(self.pointFF.y() / factor))
            seed_pt = x, y
            connectivity = config.connectivity_Floodfill
            fixed_range = config.fixed_range
            flags = connectivity
            if fixed_range:
                flags |= cv2.FLOODFILL_FIXED_RANGE
            img = self.cvimage.copy()
            self.maskFF = self.createFloodFillMask()
            cv2.floodFill(img, self.maskFF, seed_pt, (255, 255, 255),
                          (self.blueDiff, self.greenDiff, self.redDiff),
                          (self.blueDiff, self.greenDiff, self.redDiff), flags)
            height, width = self.cvimage.shape[:2]
            subtract = cv2.subtract(img, self.cvimage)
            grayimage = cv2.cvtColor(subtract, cv2.COLOR_RGB2GRAY)
            ret, mask = cv2.threshold(grayimage, 1, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            singlecolor = np.zeros((height, width, 3), np.uint8)  # Create a single color image to specify the FF area
            singlecolor[:, :] = [self.currentColor.red(), self.currentColor.green(), self.currentColor.blue()]
            area_FloodFill = cv2.bitwise_and(singlecolor, singlecolor, mask=mask)
            change_output = cv2.bitwise_and(self.outputMask, self.outputMask, mask=mask_inv)
            self.outputMask = cv2.add(change_output, area_FloodFill)
            self.notFinishAreaChoosing()
            self.isLaballing = False
            self.showImage()
            self.floodFillConfig.setDisabled()
            self.updateStatus("Apply flood-fill to the selected area")
        elif self.rectLabelAction.isChecked():
            cv2.rectangle(self.outputMask, (topleft_x, topleft_y),
                          (bottomright_x, bottomright_y),
                          (self.currentColor.red(), self.currentColor.green(),
                           self.currentColor.blue()), -1)
            self.isLaballing = False
            self.showImage()
            self.notFinishAreaChoosing()
            self.updateStatus("Label selected rectangle area")
        elif self.ellipseLabelAction.isChecked():
            cv2.ellipse(self.outputMask,
                        ((topleft_x + bottomright_x) / 2, (topleft_y + bottomright_y) / 2),
                        ((abs(topleft_x - bottomright_x) / 2), (abs(topleft_y - bottomright_y) / 2)),
                        0, 0, 360, (self.currentColor.red(), self.currentColor.green(),
                        self.currentColor.blue()), -1)
            self.isLaballing = False
            self.showImage()
            self.notFinishAreaChoosing()
            self.updateStatus("Label selected ellipse area")
        elif self.polygonLabelAction.isChecked():
            pts = []
            for point in self.points:
                pt = [point.x() / factor, point.y() / factor]
                pts.append(pt)
            poly = np.array([pts[1:]], dtype=np.int32)
            cv2.fillPoly(self.outputMask, poly, (self.currentColor.red(),
                        self.currentColor.green(), self.currentColor.blue()))
            self.isLaballing = False
            self.points = []
            self.showImage()
            self.notFinishAreaChoosing()
            self.updateStatus("Label selected polygon area")
        elif self.floodMask:
            indices = np.argwhere(self.floodMaskMask == 1)
            for i in range(0, len(indices)):
                self.outputMask[indices[i][0]][indices[i][1]] = [self.currentColor.red(),
                                                                 self.currentColor.green(), self.currentColor.blue()]
            self.floodMask = False
        elif self.spButton == 1:  # self.spAddAction.isChecked():
            label, x, y = self.getLabel(self.spPosition)
            if label is not None:
                indices = np.argwhere(self.spSegments == label)
                for i in range(0, len(indices)):
                    self.outputMask[indices[i][0]][indices[i][1]] = [self.currentColor.red(),
                            self.currentColor.green(), self.currentColor.blue()]
                # self.updateStatus("Superpixel at x:%d y:%d added" % (x, y))
                #self.updateStatus("Superpixel at x:%d y:%d added, label:%d" % (x, y, label))
        elif self.spButton == 2:  # self.spSubAction.isChecked():
            label, x, y = self.getLabel(self.spPosition)
            if label is not None:
                indices = np.argwhere(self.spSegments == label)
                for i in range(0, len(indices)):
                    self.outputMask[indices[i][0]][indices[i][1]] = [self.backgroundColor.red(),
                                                                     self.backgroundColor.green(),
                                                                     self.backgroundColor.blue()]
            # self.updateStatus("Superpixel at x:%d y:%d removed" % (x, y))
        elif self.clusterButton == 1:
            self.updateStatus("updating add cluster")
            label, x, y = self.getLabel(self.clusterPosition)
            if label is not None:
                indices = np.argwhere(self.regionSegments == label)
                for i in range(0, len(indices)):
                    self.outputMask[indices[i][0]][indices[i][1]] = [self.currentColor.red(),
                                                                     self.currentColor.green(),
                                                                     self.currentColor.blue()]
        elif self.clusterButton == 2:
            self.updateStatus("updating sub cluster")
            label, x, y = self.getLabel(self.clusterPosition)
            if label is not None:
                indices = np.argwhere(self.regionSegments == label)
                for i in range(0, len(indices)):
                    self.outputMask[indices[i][0]][indices[i][1]] = [self.backgroundColor.red(),
                                                                     self.backgroundColor.green(),
                                                                     self.backgroundColor.blue()]
        if not self.dirty:
            self.setDirty()
        self.undoAction.setEnabled(True)
        self.begin = None

    def clearLabel(self):
        if self.dirty:
            if not self.okToContinue():
                self.updateStatus("Clear canceled")
                return
        copy = self.outputMask.copy()
        self.historyStack.append(copy)
        self.outputMask = np.zeros(self.outputMask.shape, np.uint8)
        if not self.dirty:
            self.setDirty()
        self.undoAction.setEnabled(True)
        self.updateStatus("All annotation cleared.")
        self.showImage()

    def deleteLabel(self):
        """Unlabel the chosen area"""
        if not self.isLaballing:
            return

        copy = self.outputMask.copy()
        self.historyStack.append(copy)

        self.notFinishAreaChoosing()
        factor = self.lastSpinboxValue * 1.0 / 100
        topleft_x = int(round(self.begin.x() / factor))
        topleft_y = int(round(self.begin.y() / factor))
        bottomright_x = int(round(self.end.x() / factor))
        bottomright_y = int(round(self.end.y() / factor))
        if self.rectLabelAction.isChecked():
            cv2.rectangle(self.outputMask, (topleft_x, topleft_y),
                          (bottomright_x, bottomright_y),
                          (self.backgroundColor.red(), self.backgroundColor.green(),
                           self.backgroundColor.blue()), -1)
            self.isLaballing = False
            self.showImage()
            self.updateStatus("Unlabel selected rectangle area")
        elif self.ellipseLabelAction.isChecked():
            cv2.ellipse(self.outputMask,
                        ((topleft_x + bottomright_x) / 2, (topleft_y + bottomright_y) / 2),
                        ((abs(topleft_x - bottomright_x) / 2), (abs(topleft_y - bottomright_y) / 2)),
                        0, 0, 360, (self.backgroundColor.red(), self.backgroundColor.green(),
                                    self.backgroundColor.blue()), -1)
            self.isLaballing = False
            self.showImage()
            self.updateStatus("Unlabel selected ellipse area")
        elif self.polygonLabelAction.isChecked():
            pts = []
            for point in self.points:
                pt = [point.x() / factor, point.y() / factor]
                pts.append(pt)
            poly = np.array([pts[1:]], dtype=np.int32)
            cv2.fillPoly(self.outputMask, poly, (self.backgroundColor.red(),
                        self.backgroundColor.green(), self.backgroundColor.blue()))
            self.isLaballing = False
            self.finishChoosingArea = False
            self.points = []
            self.showImage()
            self.updateStatus("Unlabel selected polygon area")

        if not self.dirty:
            self.setDirty()
        self.undoAction.setEnabled(True)

    def changeFloodFill(self):
        """Change flood fill configuration --
        difference of pixel values in different colors"""
        self.redDiff = self.floodFillConfig.getRedValue()
        self.greenDiff = self.floodFillConfig.getGreenValue()
        self.blueDiff = self.floodFillConfig.getBlueValue()
        self.showImage()

    def createFloodFillMask(self):
        """Create and return a mask for flood fill action"""
        factor = self.lastSpinboxValue * 1.0 / 100
        topleft_x = int(round(self.begin.x() / factor)) + 1
        topleft_y = int(round(self.begin.y() / factor)) + 1
        bottomright_x = int(round(self.end.x() / factor)) + 1
        bottomright_y = int(round(self.end.y() / factor)) + 1
        height, width = self.cvimage.shape[:2]
        mask = np.zeros((height + 2, width + 2), np.uint8)
        mask[:] = 1
        if self.rectLabelAction.isChecked():
            cv2.rectangle(mask, (topleft_x, topleft_y),
                          (bottomright_x, bottomright_y), 0, -1)
        elif self.ellipseLabelAction.isChecked():
            cv2.ellipse(mask, ((topleft_x + bottomright_x) / 2, (topleft_y + bottomright_y) / 2),
                        ((abs(topleft_x - bottomright_x) / 2), (abs(topleft_y - bottomright_y) / 2)),
                        0, 0, 360, 0, -1)
        elif self.polygonLabelAction.isChecked():
            pts = []
            for point in self.points:
                pt = [point.x() / factor + 1, point.y() / factor + 1]
                pts.append(pt)
            poly = np.array([pts[1:]], dtype=np.int32)
            cv2.fillPoly(mask, poly, 0)

        return mask

    def setFloodFillAction(self):
        """Set mouse action for flood fill"""
        if self.floodFillAction.isChecked():
            self.imageLabel.mousePressEvent = self.mousePressFF
            self.imageLabel.mouseMoveEvent = self.mouseMoveFF
            self.imageLabel.mouseReleaseEvent = self.mouseReleaseFF
            self.imageLabel.mouseDoubleClickEvent = self.doubleClickFF
            self.deleteAction.setEnabled(False)
        else:
            if self.polygonLabelAction.isChecked():
                self.points = []
                self.lines = []
            self.choosingPointFF = False
            self.floodFillConfig.setDisabled()
            if self.mouseAction.isChecked():
                self.mouseAction.setChecked(False)
                self.mouseAction.setChecked(True)
            elif self.ellipseLabelAction.isChecked():
                self.ellipseLabelAction.setChecked(False)
                self.ellipseLabelAction.setChecked(True)
            elif self.rectLabelAction.isChecked():
                self.rectLabelAction.setChecked(False)
                self.rectLabelAction.setChecked(True)
            elif self.polygonLabelAction.isChecked():
                self.polygonLabelAction.setChecked(False)
                self.polygonLabelAction.setChecked(True)

    def doubleClickFF(self, event):
        pass

    def mousePressFF(self, event):
        self.pointFF = event.pos()
        self.chooseFFPointSpinBoxValue = self.zoomSpinBox.value()
        self.choosingPointFF = True
        self.floodFillConfig.setEnabled()
        self.showImage()

    def mouseMoveFF(self, event):
        pass

    def mouseReleaseFF(self, event):
        pass

    def chooseOutlineColor(self):
        color = self.colorDialog.getColor(self.currentOutlineColor,
                                          "Choose outline colour", config.DEFAULT_FILLING_COLOR)
        if color:
            self.currentOutlineColor = color
            self.updateStatus("Choose a new colour")
            icon = QPixmap(50, 50)
            icon.fill(self.currentOutlineColor)
            self.clusterPaletteAction.setIcon(QIcon(icon))
        self.showImage()

    def chooseLabelColor(self):
        """Use a palette to choose labelling color"""
        color = self.colorDialog.getColor(self.currentLabelColor,
                                "Choose labelling colour", config.DEFAULT_FILLING_COLOR)
        if color:
            self.currentLabelColor = color
            self.updateStatus("Choose a new colour")
            icon = QPixmap(50, 50)
            icon.fill(self.currentLabelColor)
            self.labelPaletteAction.setIcon(QIcon(icon))

    def floodMaskInit(self, pos):
        self.updateStatus("Flood")
        self.floodMask = True
        colour = self.getColour(pos)
        self.floodMaskMask = np.zeros((self.outputMask.shape[0], self.outputMask.shape[1]))
        factor = self.zoomSpinBox.value() * 1.0 / 100.0
        x = int(round(pos.x() / factor))
        y = int(round(pos.y() / factor))
        floodQueue = []
        floodQueue.append([y, x])
        self.floodMaskMask[y][x] = 1

        while len(floodQueue) > 0:
            cy = floodQueue[0][0]
            cx = floodQueue[0][1]
            # cy + 1
            if cy + 1 < self.outputMask.shape[0] and \
                    (self.outputMask[cy+1][cx] == colour).all() and \
                    self.floodMaskMask[cy+1][cx] == 0:
                floodQueue.append([cy+1, cx])
                self.floodMaskMask[cy+1][cx] = 1
            # cy - 1
            if cy - 1 > 0 and \
                    (self.outputMask[cy-1][cx] == colour).all() and \
                    self.floodMaskMask[cy-1][cx] == 0:
                floodQueue.append([cy-1, cx])
                self.floodMaskMask[cy-1][cx] = 1
            # cx + 1
            if cx + 1 < self.outputMask.shape[1] and \
                    (self.outputMask[cy][cx+1] == colour).all() and \
                    self.floodMaskMask[cy][cx+1] == 0:
                floodQueue.append([cy, cx+1])
                self.floodMaskMask[cy][cx+1] = 1
            # cx - 1
            if cx - 1 > 0 and \
                    (self.outputMask[cy][cx-1] == colour).all() and \
                    self.floodMaskMask[cy][cx-1] == 0:
                floodQueue.append([cy, cx-1])
                self.floodMaskMask[cy][cx-1] = 1
            # cy + 1, cx + 1
            if cy + 1 < self.outputMask.shape[0] and cx + 1 < self.outputMask.shape[1] and \
                    (self.outputMask[cy+1][cx+1] == colour).all() and \
                    self.floodMaskMask[cy+1][cx+1] == 0:
                floodQueue.append([cy+1, cx+1])
                self.floodMaskMask[cy+1][cx+1] = 1
            # cy + 1, cx - 1
            if cy + 1 < self.outputMask.shape[0] and cx - 1 > 0 and \
                    (self.outputMask[cy+1][cx-1] == colour).all() and \
                    self.floodMaskMask[cy+1][cx-1] == 0:
                floodQueue.append([cy+1, cx-1])
                self.floodMaskMask[cy+1][cx-1] = 1
            # cy - 1, cx + 1
            if cy - 1 > 0 and cx + 1 < self.outputMask.shape[1] and \
                    (self.outputMask[cy-1][cx+1] == colour).all() and \
                    self.floodMaskMask[cy-1][cx+1] == 0:
                floodQueue.append([cy-1, cx+1])
                self.floodMaskMask[cy-1][cx+1] = 1
            # cy - 1, cx - 1
            if cy - 1 > 0 and cx + 1 > 0 and \
                    (self.outputMask[cy-1][cx-1] == colour).all() and \
                    self.floodMaskMask[cy-1][cx-1] == 0:
                floodQueue.append([cy-1, cx-1])
                self.floodMaskMask[cy-1][cx-1] = 1

            floodQueue.pop(0)
        # apply mask to output mask
        self.updateStatus("flood finished")
        cv2.imwrite("flood_test.png", self.floodMaskMask*255)
        self.confirmEdit()
        self.showImage()

    def pickColor(self, pos):
        self.updateStatus("Pick colour")
        colour = self.getColour(pos)
        colour = QColor(colour[0], colour[1], colour[2])
        self.chooseColor(colour)

    def chooseColor(self, color=None):
        """Use a palette to choose labelling color"""
        if color is None:
            color = self.colorDialog.getColor(self.currentColor,
                                              "Choose labelling colour", config.DEFAULT_FILLING_COLOR)
        print color
        if color:
            self.currentColor = color
            self.updateStatus("Choose a new colour")
            icon = QPixmap(50, 50)
            icon.fill(self.currentColor)
            self.paletteAction.setIcon(QIcon(icon))
            # if not self.colorLabelBar.isHidden() and \
            #                 self.labelsGroup.checkedAction() is not None:
            #     self.labelsGroup.checkedAction().setChecked(False)
            if self.floodFillAction.isChecked():
                self.showImage()

    def chooseColor_2(self):
        """Choose and use a user specified labelling color"""
        action = self.sender()
        if action.isChecked():
            color = self.colorLabelDict[action]
            self.currentColor = color
            self.updateStatus("Prepare to label %s" % action.text())
            self.showImage()
            icon = QPixmap(50, 50)
            icon.fill(self.currentColor)
            self.paletteAction.setIcon(QIcon(icon))

    def hideLog(self):
        if self.logDockWidget.isVisible():
            self.logDockWidget.hide()
        else:
            self.logDockWidget.show()

    def zoomIn(self):
        """Zoom in by 5%"""
        if self.zoomSpinBox.value() + 5 < self.zoomSpinBox.maximum():
            self.zoomSpinBox.setValue(self.zoomSpinBox.value() + 5)
        else:
            self.zoomSpinBox.setValue(self.zoomSpinBox.maximum())

    def zoomOut(self):
        """Zoom out by 5%"""
        if self.zoomSpinBox.value() - 5 > self.zoomSpinBox.minimum():
            self.zoomSpinBox.setValue(self.zoomSpinBox.value() - 5)
        else:
            self.zoomSpinBox.setValue(self.zoomSpinBox.minimum())

    def loadInitFile(self):
        """Load the last file before closing last time."""
        settings = QSettings()
        fname = unicode(settings.value("LastFile").toString())
        if fname and QFile.exists(fname):
            self.filename = fname.replace("/", "\\")
            self.filepath = os.path.dirname(self.filename)
            self.allImages = []
            self.allImages.append(self.filename)
            self.fileListWidget.clear()
            filename = os.path.basename(fname)
            item = QListWidgetItem(filename)
            self.fileListWidget.addItem(item)
            
            self.cvimage = fname
            self.loadImage(fname)
        self.updateToolBar()

    def mouseWheelEvent(self, event):
        """Use mouse wheel to zoom the image"""
        changes = event.delta() / 120
        self.zoomSpinBox.setValue(self.zoomSpinBox.value() + changes * 2)

    def closeEvent(self, event):
        """Before closing pop out a message box to comfirm"""
        if not self.okToContinue():
            event.ignore()

        else:
            reply = None
            if self.spMassActive:
                reply = QMessageBox.question(self, "Exit", "The mass superpixel alogrithm is still running, " + 
                                                           "leaving will mean having to run full or individually next time." + 
                                                           "Do you want to exit?",
                                                            QMessageBox.Yes | QMessageBox.No)
            else:
                reply = QMessageBox.question(self, "Exit", "You are going to leave the " +
                                                           "Image Annotation Tool. Are you sure?",
                                                            QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                if self.spMassActive:
                    i=1
                
                settings = QSettings()
                recentFiles = QVariant(self.recentFiles) \
                            if self.recentFiles else QVariant()
                settings.setValue("RecentFiles", recentFiles)
                filename = QVariant(QString(self.filename)) \
                            if self.filename is not None else QVariant()
                settings.setValue("LastFile", filename)
                settings.setValue("MainWindow/Size", QVariant(self.size()))
                settings.setValue("MainWindow/Position",
                                QVariant(self.pos()))
                settings.setValue("MainWindow/State",
                                QVariant(self.saveState()))

            elif reply == QMessageBox.No:
                event.ignore()

    def helpAbout(self):
        QMessageBox.about(self, "About Image Annotation Tool",
                                """<html><head/><body>
                                <b>Image Annotation Tool</b> v %s
                                <p>Welcome to the image annotation tool. There are currently 3 mode of use.</p>
                                <ol>
                                <li>Geometric annotations</li>
                                <li>Superpixel annotations</li>
                                <li>Super-cluster annotations</li>
                                </ol>
                                <p>Example dataset types are plants, litter, and medical.</p>
                                <p>Python %s - Qt %s - PyQt %s on %s</p>
                                <p>Aurthors: </p>
                                <ul>
                                <li>Geometric Annotations: Jingxiao Ma</li> 
                                <li>Superpixel and Clusters: Thomas J. Smith</li>
                                </ul> 
                                <p>Computer Vision Lab,</p>
                                <p>School of Computer Science,</p>
                                <p>University of Nottingham</p>
                                </body></html>"""
                                %( __version__, platform.python_version(), QT_VERSION_STR,
                                PYQT_VERSION_STR, platform.system()))

    def helpHelp(self):
        QMessageBox.about(self, "Help -- Image Annotation Tool",
                          """<html><head/><body>
                          <b>Usage:</b>
                          <ol>
                          <li>Open a directory or an image.</li>
                          <li>Choose colour.</li>
                          <li>Choose a shape and region.</li>
                          <li>Confirm, unlabel or perform flood-fill.</li> 
                          <li>Save image when finish annotating.</li>
                          </ol>
                          <p>
                          For details, please check user manual in docs folder.\n
                          </p>
                          <br/>
                          <b>Keyboard Shortcuts:</b>
                          <table cellpadding=10 border=1>
                          <tr>
                          <td>open          = Ctrl+O</td>
                          <td>Directory Open= Ctrl+D</td>
                          <td>Move file     = Ctrl+M</td>
                          </tr>
                          <tr>
                          <td>Save          = Ctrl+S</td>
                          <td>Undo          = Ctrl+Z</td>
                          <td>Quit          = Ctrl+Q</td>
                          </tr>
                          <tr>
                          <td>Zoom in       = Ctrl+=</td>
                          <td>Zoom out      = Ctrl+-</td>
                          <td>Hide Log      = Ctrl+L</td>
                          </tr>
                          <tr>
                          <td>Hide Image    = Ctrl+1</td>
                          <td>Hide Mask     = Ctrl+2</td>
                          <td>Hide Superpixel Outline = Ctrl+3</td>
                          </tr>
                          <tr>
                          <td>Hide Cluster  = Ctrl+4</td>
                          <td>Show Suggested Clusters = Ctrl+5</td>
                          <td>Show Cluster Outlines = Ctrl+6</td>
                          </tr>
                          <tr>
                          <td>Change Outline Colour = Ctrl+7</td>
                          <td>None Geometric= Alt+1</td>
                          <td>Rectangle     = Alt+2</td>
                          </tr>
                          <tr>
                          <td>Elipse        = Alt+3</td>
                          <td>Polygon       = Alt+4</td>
                          <td>None SP       = Alt+5</td>
                          </tr>
                          <tr>
                          <td>Add SP        = Alt+6</td>
                          <td>None Cluster  = Alt+7</td>
                          <td>Add Cluster   = Alt+8</td>
                          </tr>
                          <tr>
                          <td>Run Cluster   = Alt+C</td>
                          <td>Run Superpixel= Alt+S</td>
                          <td>Clear         = Alt+X</td>
                          </tr>
                          </table>
                          <p>Label colour shortcuts use shift plus a number:</p>
                          <ul>
                          <li>unknown:  shift+0</li>
                          <li>colour 1: shift+1</li>
                          <li>colour 10: shift+1+0</li>
                          <li>colour 11: skip as uses duplicate key</li>
                          <li>colour 12: shift+1+2</li>
                          <li>colour 21: skip as uses same keys as 12</li>
                          </ul>
                          </body></html>""")

    @staticmethod
    def normalise(norm_input, max_num=1):
        # with warnings.catch_warnings():
        #    warnings.simplefilter("ignore")
        norm_input = norm_input.astype(float)
        min_ = abs(np.min(norm_input))
        max_ = abs(np.max(norm_input))
        i_max = min_ if min_ > max_ else max_
        if i_max == 0:
            return 0
        norm_input *= (max_num / i_max)
        return norm_input

    @staticmethod
    def append_to_csv(csv_output_path, data, multiple_instances=False):
        with open(csv_output_path, 'ab') as csv_file:
            wr = csv.writer(csv_file)
            if multiple_instances:
                for i in range(len(data)):
                    wr.writerow(data[i])
            else:
                wr.writerow(data)

    @staticmethod
    def write_to_csv(csv_output_path, data, multiple_instances=False):
        with open(csv_output_path, 'w') as csv_file:
            wr = csv.writer(csv_file)
            if multiple_instances:
                for i in range(len(data)):
                    wr.writerow(data[i])
            else:
                wr.writerow(data)


def main():
    app = QApplication(sys.argv)

    app.setOrganizationName("CS, University of Nottingham")
    app.setOrganizationDomain("cs.nott.ac.uk")
    app.setApplicationName("Image_Annotation_Tool")
    app.setWindowIcon(QIcon(":/WindowIcon.png"))

    img_anno = MainWindow()
    img_anno.show()
    app.exec_()


reload(sys)
sys.setdefaultencoding('utf-8')
main()
