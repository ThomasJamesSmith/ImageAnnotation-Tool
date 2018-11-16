# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# This is Image Annotation Tool for annotating plantations.
# Created by Jingxiao Ma.
# Languages: python 2 (2.7)
# Sys requirement: Linux / Windows 7 / OS X 10.8 or later versions
# Package requirement: PyQt 4 / OpenCv 2 / numpy

#
#   use the segments from sp as overlay
#   np.where(a == 1)
#   can select a segment, when selected search sp_img for all occurences of label
#   selected pixels are then added to the mask
#   have option to remove segment from mask
#

# compile new icons
# C:\Python27\Lib\site-packages\PyQt4\pyrcc4.exe -o qrc_resources.py resources.qrc

import os
import platform
import sys
import qrc_resources
import numpy as np
import cv2
import csv
import config
from colorDialog import *
from FloodFillConfig import *
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
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


__version__ = '1.2'


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
        self.outputMask = None      # Output image
        self.displayImage = None

        self.allImages = []
        self.finished = False   # Indicate whether the image annotation is finished
        self.filename = None    # Path of the image file
        self.filepath = None    # Directory of the image file
        self.dirty = False      # Whether modified
        self.isLoading = True
        self.colourLabels = None
        self.hideImg = False
        self.hideSP = False
        self.hideCluster = False
        self.showSuggestedCluster = False
        self.hideMask = False
        self.isLaballing = False
        self.finishChoosingArea = False
        self.spActive = False
        self.spNum = 550
        self.spMask = None
        self.clusterMask = None
        self.suggestedClusterMask = None
        self.q = Queue(maxsize=0)
        self.q_Video = Queue(maxsize=0)
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
        self.logDockWidget.setAllowedAreas(Qt.LeftDockWidgetArea |
                                           Qt.RightDockWidgetArea)
        self.listWidget = QListWidget()
        self.logDockWidget.setWidget(self.listWidget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.logDockWidget)

        # Set status bar
        self.sizeLabel = QLabel()
        self.sizeLabel.setFrameStyle(QFrame.StyledPanel|
                                     QFrame.Sunken)
        status = self.statusBar()
        status.setSizeGripEnabled(False)
        status.addPermanentWidget(self.sizeLabel)
        status.showMessage("Ready", 5000)

        # Create actions
        fileOpenAction = self.createAction("&Open...", self.fileOpen, QKeySequence.Open,
                                           "open", "Open an existing image file")
        dirOpenAction = self.createAction("&Dir Open...", self.dirOpen, "Ctrl+D",
                                          "open", "Open an existing directory")
        self.saveAction = self.createAction("&Save...", self.saveFile, QKeySequence.Save,
                                       "save", "Save modified image")
        self.saveAction.setEnabled(False)

        self.undoAction = self.createAction("&Undo...", self.undo, QKeySequence.Undo,
                                            "undo", "Undo the last operation. "
                                            "NOTE: This operation is irreversible.")
        self.undoAction.setEnabled(False)

        quitAction = self.createAction("&Quit...", self.close, QKeySequence.Quit,
                                       "close", "Close the application")
        zoomInAction = self.createAction("&Zoom\nIn...", self.zoomIn, "Alt+Z",
                                         "zoom-in", "Zoom in image")
        zoomOutAction = self.createAction("&Zoom\nout...", self.zoomOut, "Alt+O",
                                          "zoom-out", "Zoom out image")
        hideLogViewerAction = self.createAction("&Hide Log...", self.hideLog, "Alt+L",
                                                None, "Hide log dock")
        showLogViewerAction = self.createAction("&Show Log...", self.showLog, "Alt+K",
                                                None, "Show log dock")

        self.hideOriginalAction = self.createAction("&Hide\nImage", self.hideButtonClick, "Ctrl+H",
                                                    "hide", "Hide original image", True, "toggled(bool)")
        self.hideMaskAction = self.createAction("&Hide\nMask", self.hideButtonClick, "Ctrl+H",
                                                    "hide", "Hide original image", True, "toggled(bool)")
        self.hideOriginalAction.setChecked(False)
        self.hideMaskAction.setChecked(False)

        self.paletteAction = self.createAction("&Palette...", self.chooseColor, "Ctrl+L",
                                          None, "Choose the color to label items")

        self.confirmAction = self.createAction("&Confirm...", self.confirmEdit, QKeySequence.InsertParagraphSeparator,
                                          "done", "Fill in the area with selected color")
        self.confirmAction.setEnabled(False)
        self.deleteAction = self.createAction("&Unlabel...", self.deleteLabel, "Del",
                                         "delete", "Delete area and make it background")
        self.deleteAction.setEnabled(False)
        self.floodFillAction = self.createAction("&Flood\nFill", self.setFloodFillAction, "Ctrl+F",
                                                 "flood-fill", "Apply flood-fill to selected area", True, "toggled(bool)")
        self.floodFillAction.setEnabled(False)
        
        self.spAction = self.createAction("&Superpixel", self.runSuperpixelAlg, "Alt+s", "superpixel", "Run superpixel Algorithm")
        
        self.hidespAction = self.createAction("&Hide\nSuperpixels", self.hideButtonClick, "Alt+H",
                                                    "hide", "Hide superpixel overlay", True, "toggled(bool)")


        # Create group of actions for superpixels
        spGroup = QActionGroup(self)
        #
        self.spMouseAction = self.createAction("&None...", self.setMouseAction, "Alt+p",
                                               "cursor", "No Action", True, "toggled(bool)")        
        spGroup.addAction(self.spMouseAction)
        #
        self.spAddAction = self.createAction("&Add \nSuperpixel", self.labelSPAdd, "Ctrl+{",
                                             "SPadd", "Add superpixel to segment", True, "toggled(bool)")
        spGroup.addAction(self.spAddAction)
        #
        self.spSubAction = self.createAction("&Subtract \nSuperpixel", self.labelSPAdd, "Ctrl+}", 
                                             "SPsub", "Subtract superpixel from segment", True, "toggled(bool)")
        spGroup.addAction(self.spSubAction)
        #
        self.spMouseAction.setChecked(True)
        self.spMouseAction.setEnabled(False)
        self.spAddAction.setEnabled(False)
        self.spSubAction.setEnabled(False)
        self.hidespAction.setEnabled(False)

        # Create group of actions for cluster
        clusterGroup = QActionGroup(self)
        self.clusterAction = self.createAction("&Cluster", self.openClusters, "Alt+s", "cluster",
                                               "add cluster overlay")
        self.clusterMouseAction = self.createAction("&None...", self.setMouseAction, "Alt+c",
                                                    "cursor", "No Action", True, "toggled(bool)")
        clusterGroup.addAction(self.clusterMouseAction)

        self.clusterAddAction = self.createAction("&Add \nCluster", self.labelClusterAdd, "Ctrl+{",
                                                  "SPadd", "Add cluster to segment", True, "toggled(bool)")
        clusterGroup.addAction(self.clusterAddAction)
        self.clusterSubAction = self.createAction("&Subtract \nCluster", self.labelClusterAdd, "Ctrl+}",
                                                  "SPsub", "Subtract cluster frp, segment", True, "toggled(bool)")
        clusterGroup.addAction(self.clusterSubAction)

        self.hideClusterAction = self.createAction("&Hide\nCluster", self.hideButtonClick, "Alt+H",
                                                   "hide", "Hide cluster overlay", True, "toggled(bool)")
        self.showSuggestedClusterAction = self.createAction("&Show\nSuggested\nCluster", self.hideButtonClick, "Alt+H",
                                                            "hide", "Show suggested cluster overlay", True, "toggled(bool)")

        self.clusterMouseAction.setChecked(True)
        self.clusterMouseAction.setEnabled(False)
        self.clusterAddAction.setEnabled(False)
        self.clusterSubAction.setEnabled(False)
        self.hideClusterAction.setEnabled(False)
        self.showSuggestedClusterAction.setEnabled(False)

        labelGroup = QActionGroup(self)
        self.labelAction = self.createAction("&Open\nLabels", self.editLabelFile, "Alt+f", "labels",
                                             "Open label file, create if doesn't exist")
        labelGroup.addAction(self.labelAction)

        self.labelAddAction = self.createAction("&Add \nSuperpixel", self.labelToFileAdd, "Ctrl+{",
                                             "SPadd", "Add new semantic label to labels file", True)
        labelGroup.addAction(self.labelAddAction)
        self.labelAddAction.setEnabled(False)
        
        helpAboutAction = self.createAction("&About...", self.helpAbout, None, "helpabout")
        helpHelpAction = self.createAction("&Help...", self.helpHelp, None, "help")

        # Create group of actions for labelling image
        editGroup = QActionGroup(self)
        self.mouseAction = self.createAction("&None...", self.setMouseAction, "Alt+0",
                                        "cursor", "No Action", True, "toggled(bool)")
        editGroup.addAction(self.mouseAction)
        self.rectLabelAction = self.createAction("&Rect...", self.labelRectOrEllipse, "Alt+1",
                                       "rectangle", "Annotation a rectangle area", True, "toggled(bool)")
        editGroup.addAction(self.rectLabelAction)
        self.ellipseLabelAction = self.createAction("&Ellipse...", self.labelRectOrEllipse, "Alt+2",
                                                 "ellipse", "Annotation an ellipse area", True, "toggled(bool)")
        editGroup.addAction(self.ellipseLabelAction)
        self.polygonLabelAction = self.createAction("&Polygon...", self.labelPolygon, "Alt+3",
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
        self.spSpinBox.setRange(100, 1000)
        self.spSpinBox.setValue(550)
        self.spSpinBox.setToolTip("Set number of Superpixels")
        self.spSpinBox.setStatusTip(self.spSpinBox.toolTip())
        self.spSpinBox.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.connect(self.spSpinBox,
                     SIGNAL("valueChanged(int)"), self.updateSPNum)
        self.spNum = self.spSpinBox.value()

        # Label spin boxes
        self.labelRedSpinBox = QSpinBox()
        self.labelRedSpinBox.setRange(0, 255)
        self.labelRedSpinBox.setValue(0)
        self.labelRedSpinBox.setToolTip("Set red value for label")
        self.labelRedSpinBox.setStatusTip(self.labelRedSpinBox.toolTip())
        self.labelRedSpinBox.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.connect(self.labelRedSpinBox, SIGNAL("valueChanged(int)"), self.updateLabelNums)
        self.labelRedNum = self.labelRedSpinBox.value()
        self.labelRedSpinBox.setEnabled(False)

        self.labelGreenSpinBox = QSpinBox()
        self.labelGreenSpinBox.setRange(0, 255)
        self.labelGreenSpinBox.setValue(0)
        self.labelGreenSpinBox.setToolTip("Set green value for label")
        self.labelGreenSpinBox.setStatusTip(self.labelGreenSpinBox.toolTip())
        self.labelGreenSpinBox.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.connect(self.labelGreenSpinBox, SIGNAL("valueChanged(int)"), self.updateLabelNums)
        self.labelGreenNum = self.labelGreenSpinBox.value()
        self.labelGreenSpinBox.setEnabled(False)

        self.labelBlueSpinBox = QSpinBox()
        self.labelBlueSpinBox.setRange(0, 255)
        self.labelBlueSpinBox.setValue(0)
        self.labelBlueSpinBox.setToolTip("Set blue value for label")
        self.labelBlueSpinBox.setStatusTip(self.labelBlueSpinBox.toolTip())
        self.labelBlueSpinBox.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.connect(self.labelBlueSpinBox, SIGNAL("valueChanged(int)"), self.updateLabelNums)
        self.labelBlueNum = self.labelBlueSpinBox.value()
        self.labelBlueSpinBox.setEnabled(False)
        self.labelName = "New Name"


        # Create color dialog
        self.colorDialog = ColorDialog(parent=self)

        # Create menu bar
        self.fileMenu = self.menuBar().addMenu("&File")
        self.fileMenuActions = (fileOpenAction, dirOpenAction, self.saveAction, None, quitAction)
        self.connect(self.fileMenu, SIGNAL("aboutToShow()"),
                     self.updateFileMenu)
        self.fileMenu.setMaximumWidth(400)

        editMenu = self.menuBar().addMenu("&Edit")
        self.addActions(editMenu, (self.undoAction, None, self.paletteAction, None,
                                   self.confirmAction, self.deleteAction, self.floodFillAction,
                                   None, self.mouseAction, self.rectLabelAction,
                                   self.ellipseLabelAction, self.polygonLabelAction))

        viewMenu = self.menuBar().addMenu("&View")
        self.addActions(viewMenu, (zoomInAction, zoomOutAction, self.hideOriginalAction,self.hideMaskAction,
                                   None, hideLogViewerAction, showLogViewerAction))

        helpMenu = self.menuBar().addMenu("&Help")
        self.addActions(helpMenu, (helpAboutAction, helpHelpAction, None))

        # Create tool bar
        self.toolBar = self.addToolBar("File&Edit")
        self.toolBar.setAllowedAreas(Qt.TopToolBarArea)
        self.toolBar.setMovable(False)
        self.toolBar.setIconSize(QSize(24, 24))
        self.toolBar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.toolBar.setObjectName("ToolBar")
        self.toolBarActions_1 = (fileOpenAction, dirOpenAction, self.saveAction, self.undoAction,
                                 quitAction, None, zoomInAction)
        self.toolBarActions_2 = (zoomOutAction, self.hideOriginalAction, self.hideMaskAction, None,
                                 self.paletteAction, self.confirmAction, self.deleteAction,
                                 self.floodFillAction, None, self.mouseAction, self.rectLabelAction,
                                 self.ellipseLabelAction, self.polygonLabelAction,None)
        self.toolBarActions_3 = (self.spAction, self.hidespAction, self.spMouseAction,
                                 self.spAddAction, self.spSubAction, None)

        self.toolBarActions_4 = (self.clusterAction, self.hideClusterAction, self.showSuggestedClusterAction,
                                 self.clusterMouseAction, self.clusterAddAction, self.clusterSubAction, None)
        self.toolBarActions_5 = (self.labelAction, self.labelAddAction, None)

        self.addActions(self.toolBar, self.toolBarActions_1)
        self.toolBar.addWidget(self.zoomSpinBox)
        self.addActions(self.toolBar, self.toolBarActions_2)
        self.toolBar.addWidget(self.spSpinBox)
        self.addActions(self.toolBar, self.toolBarActions_3)
        self.addActions(self.toolBar, self.toolBarActions_4)
        self.addActions(self.toolBar, self.toolBarActions_5)
        self.toolBar.addWidget(self.labelRedSpinBox)
        self.toolBar.addWidget(self.labelGreenSpinBox)
        self.toolBar.addWidget(self.labelBlueSpinBox)

        self.colorLabelBar = QToolBar("Labels and colors")
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

        
###############################################################################
###############################################################################


    def setDirty(self):
        """Call this method when applying changes"""
        self.dirty = True
        self.saveAction.setEnabled(True)

    def setClean(self):
        """Call this method when saving changes"""
        self.dirty = False
        self.saveAction.setEnabled(False)
        self.historyStack = []
        self.undoAction.setEnabled(False)

    def createAction(self, text, slot=None, shortcut=None, icon=None, tip=None, checkable=False, signal="triggered()"):
        """Quickly create action"""
        action = QAction(text,self)
        if slot == self.chooseColor:
            icon = QPixmap(50, 50)
            icon.fill(self.currentColor)
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

    def updateToolBar(self):
        """Update toolbar to show color labels"""
        if self.filename == None:
            return
        if self.colourLabels is None:
            self.colourLabels = config.getLabelColor(self.filename)

        if self.colourLabels is None:
            self.colorLabelBar.hide()
        else:
            self.labelAddAction.setEnabled(True)
            self.labelRedSpinBox.setEnabled(True)
            self.labelGreenSpinBox.setEnabled(True)
            self.labelBlueSpinBox.setEnabled(True)
            self.colorLabelBar.clear()
            self.colorLabelBar.show()
            self.labelsGroup = QActionGroup(self)
            self.colorLabelDict= {}
            for label in self.colourLabels.keys():
                action = self.createAction(label, self.chooseColor_2, None,
                                                       None, "Color the label with user specified color",
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

    def dirOpen(self, fromVid = False, dirname = None, applySP = None):
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
                    self.spMask = np.uint8(mark_boundaries(np.zeros(self.cvimage.shape, np.uint8), self.spSegments, color=(1, 0, 0)))*255

                    self.spActivate()
                else:
                    self.spDeactivate()
                    self.spSegments = None
                    self.spMask = None
                self.regionSegments = None

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
        
        gray_output = cv2.cvtColor(self.outputMask, cv2.COLOR_RGB2GRAY)
        ret, mask_output = cv2.threshold(gray_output, 2, 255, cv2.THRESH_BINARY)
        masked_output = cv2.bitwise_and(self.outputMask, self.outputMask, mask=mask_output)
        
        if (masked_output!=0).any() and not self.hideMask:
            inverted = True
            masked_image_output = cv2.bitwise_and(self.cvimage, self.cvimage, mask=mask_output)
            temp = cv2.addWeighted(masked_output, 0.6, masked_image_output, 0.4, 0)
            origin = cv2.bitwise_and(self.cvimage, self.cvimage, mask=cv2.bitwise_not(mask_output))
            dst = cv2.add(temp, origin)
            
        if self.spMask is not None and not self.hideSP:  # and show sp
            inverted = True
            gray_sp = cv2.cvtColor(self.spMask, cv2.COLOR_RGB2GRAY)
            ret, mask_sp = cv2.threshold(gray_sp, 2, 255, cv2.THRESH_BINARY)
            mask_sp_inverted = cv2.bitwise_not(mask_sp)
            masked_sp = cv2.bitwise_and(self.spMask, self.spMask, mask = mask_sp)
            masked_out_sp = cv2.bitwise_and(dst, dst, mask = mask_sp_inverted)
            dst = cv2.add(masked_sp, masked_out_sp)

        if self.regionSegments is not None and not self.hideCluster and not self.showSuggestedCluster:  # and show clusters
            ### change to overlay
            inverted = True
            # print np.ndarray.min(self.clusterMask)
            # print np.ndarray.max(self.clusterMask)
            print self.clusterMask.shape
            alpha = 0.5
            # alpha_channel = np.full((self.clusterMask.shape[0], self.clusterMask.shape[1], 1), alpha)
            # alpha_cluster = np.dstack((self.clusterMask, alpha_channel))
            output = dst.copy()
            cv2.addWeighted(self.clusterMask, alpha, output, 1 - alpha, 0, output)
            dst = output
        elif self.suggestedClusterMask is not None and self.showSuggestedCluster:  # and show suggested cluster only
            alpha = 0.5
            # alpha_channel = np.full((self.suggestedClusterMask.shape[0], self.suggestedClusterMask.shape[1], 1), alpha)
            # alpha_cluster = np.dstack((self.suggestedClusterMask, alpha_channel))
            print self.clusterMask.shape
            output = dst.copy()
            cv2.addWeighted(self.suggestedClusterMask, alpha, output, 1 - alpha, 0, output)
            dst = output


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
            if self.hideImg:
                output_origin = cv2.bitwise_and(self.outputMask, self.outputMask, mask=mask_inv)
                dst = cv2.add(output_origin, area_FloodFill)
            else:
                change_origin = cv2.bitwise_and(self.cvimage, self.cvimage, mask=mask_inv)
                dst = cv2.add(change_origin, area_FloodFill)

        return dst

    def undo(self):
        """Undo the last changes to the image"""
        old = self.historyStack.pop(-1)
        self.outputMask = old
        self.showImage()
        self.updateStatus("Undo")
        if len(self.historyStack) == 0:
            self.undoAction.setEnabled(False)

    def updateLabelNums(self):
        self.labelRedNum = self.labelRedSpinBox.value()
        self.labelGreenNum = self.labelGreenSpinBox.value()
        self.labelBlueNum = self.labelBlueSpinBox.value()

    def updateSPNum(self):
        self.spNum = self.spSpinBox.value()
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
        segments = slic(img, n_segments = self.spNum, sigma=1, compactness=30)
        path = config.outputFile(dir)
        pathSplit = path.split('.')
        np.savetxt(pathSplit[0] + ".csv", segments, delimiter=",", fmt="%d")        
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        cv2.imwrite(config.outputFile(dir).decode('utf-8').encode('gbk'), output)

    def runSuperpixelAlg(self):
        self.firstDone = False
        self.q.put([self.filename,0])
        self.spMassTotal = len(self.allImages)
        self.spMassComplete = 0
        self.spMassActive = True
        self.updateStatus("SP progress: %d/%d" %(self.spMassComplete, self.spMassTotal))

    def hideButtonClick(self):
        """Handle Hide button clicks"""
        if self.hideOriginalAction.isChecked():
            self.hideImg = True
        else:
            self.hideImg = False
            
        if self.hidespAction.isChecked():
            self.hideSP = True
        else: 
            self.hideSP = False

        if self.hideClusterAction.isChecked():
            self.hideCluster = True
        else:
            self.hideCluster = False

        if self.showSuggestedClusterAction.isChecked():
            self.showSuggestedCluster = True
        else:
            self.showSuggestedCluster = False

            
        if self.hideMaskAction.isChecked():
            self.hideMask = True
        else: 
            self.hideMask = False
            
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
        self.spSubAction.setEnabled(True)
        self.hidespAction.setEnabled(True)
        self.spAction.setEnabled(False)
        self.spAddAction.setChecked(True)

    def spDeactivate(self):
        self.spActive = False
        self.spMouseAction.setEnabled(False)
        self.spAddAction.setEnabled(False)
        self.spSubAction.setEnabled(False)
        self.hidespAction.setEnabled(False)
        self.spAction.setEnabled(True)
        self.spMouseAction.setChecked(True)

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
            self.imageLabel.mouseDoubleClickEvent = self.finishPoly

    def mouseReleasePoly(self, event):
        pass

    def startPoly(self, event):
        """Start labelling polygon"""
        self.imageLabel.setMouseTracking(True)
        self.lastSpinboxValue = self.zoomSpinBox.value()
        if self.finishChoosingArea:
            self.lines = []
            self.points = []
            self.finishChoosingArea = False
        self.notFinishAreaChoosing()
        self.isLaballing = True
        self.end = event.pos()
        if self.end != self.begin:
            self.points.append(self.begin)
        if len(self.points) > 1:
            self.lines.append(QLine(self.begin, self.end))
        self.begin = event.pos()
        self.imageLabel.update()

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

    def labelToFileAdd(self):
        imgRoot = os.path.dirname(self.filename)
        path = os.path.join(imgRoot, "label.txt")
        self.append_to_csv(path, [self.labelRedNum, self.labelGreenNum, self.labelBlueNum, self.labelName])
        self.updateToolBar()

    def editLabelFile(self):
        self.colourLabels = config.getLabelColor(self.filename)
        if self.colourLabels is None:
            imgRoot = os.path.dirname(self.filename)
            path = os.path.join(imgRoot, "label.txt")
            self.append_to_csv(path, [0, 0, 0, "unknown"])

        self.updateToolBar()


    def clusterActivate(self):
        self.clusterActive = True
        self.clusterMouseAction.setEnabled(True)
        self.clusterSubAction.setEnabled(True)
        self.clusterAddAction.setEnabled(True)
        self.hideClusterAction.setEnabled(True)
        self.showSuggestedClusterAction.setEnabled(True)
        self.clusterAction.setEnabled(False)
        self.clusterAddAction.setChecked(True)

    def clusterDeactivate(self):
        self.clusterActive = False
        self.clusterMouseAction.setEnabled(False)
        self.clusterAddAction.setEnabled(False)
        self.clusterSubAction.setEnabled(False)
        self.hideClusterAction.setEnabled(False)
        self.showSuggestedClusterAction.setEnabled(False)
        self.clusterAction.setEnabled(True)
        self.clusterMouseAction.setChecked(True)

    def openClusters(self):
        self.updateStatus("Open Cluster")
        split_file_dir = self.filename.split('.')
        avg_path = split_file_dir[0] + "_avg." + split_file_dir[1]
        pred_path = split_file_dir[0] + "_prediction." + split_file_dir[1]
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
        return sorted_labels[0]

    def avgToSemgentHelper(self):
        for y in range(0, self.segMaskAvg.shape[1]):
            for x in range(0, self.segMaskAvg.shape[0]):
                if x == 0 and y == 0:
                    self.segMask[x][y] = 1
                    continue
                if not x == self.segMaskAvg.shape[0] - 1:
                    self.segMask[x][y] = self.findLabel(x, y)

    def avgToSegments(self, avg):
        avg_2 = avg.reshape(avg.shape[0] * avg.shape[1], avg.shape[2])
        avg_3 = np.unique(avg_2, axis=0)
        self.segMask = np.zeros((avg.shape[0], avg.shape[1]))
        self.avgNextLabel = 2
        self.segMaskAvg = avg.copy()
        self.avgToSemgentHelper()
        self.updateStatus("avgToSegments complete")
        return self.segMask

    def labelClusterAdd(self):
        """Set mouse action when adding to superpixel segments"""
        if self.sender().isChecked():
            self.mouseAction.setChecked(True)
            self.setMouseAction()

            self.isLaballing = False
            self.notFinishAreaChoosing()
            self.showImage()
            self.imageLabel.mousePressEvent = self.startClusterAdd
            self.imageLabel.mouseMoveEvent = self.DragClusterADD
            self.imageLabel.mouseReleaseEvent = self.stopClusterAdd

    def addCluster(self):
        print self.clusterPosition.x(),self.clusterPosition.y()
        for i in range(len(self.cluster_queue)):
            if self.cluster_queue[i] == label:
                return
        self.cluster_queue.append(label)
        self.confirmEdit()
        self.showImage()

    def startClusterAdd(self, event):
        """Start labelling sp"""
        self.imageLabel.setMouseTracking(True)
        self.isLaballing = True
        self.clusterPosition = event.pos()
        self.addCluster()

    def DragClusterADD(self, event):
        self.clusterPosition = event.pos()
        self.addCluster()

    def stopClusterAdd(self, event):
        """Finish labelling sp"""
        self.imageLabel.setMouseTracking(False)
        self.cluster_queue = []
        #self.confirmEdit()
        #self.showImage()

    def getLabel(self, pos):
        factor = self.zoomSpinBox.value() * 1.0 / 100.0
        x = int(round(pos.x() / factor))
        y = int(round(pos.y() / factor))
        if self.spAddAction.isChecked() or self.spSubAction.isChecked():
            label = self.spSegments[y][x]
        elif self.clusterAddAction.isChecked() or self.clusterSubAction.isChecked():
            label = self.regionSegments[y][x]
        return label, x, y

    def addSP(self):
        label, x, y = self.getLabel(self.spPosition)
        for i in range(len(self.sp_queue)):
            if self.sp_queue[i] == label:
                return
        self.sp_queue.append(label)
        self.confirmEdit()
        self.showImage()

    def startSPAdd(self, event):
        """Start labelling sp"""
        self.imageLabel.setMouseTracking(True)
        self.lastSpinboxValue = self.zoomSpinBox.value()
        self.isLaballing = True
        self.spPosition = event.pos()
        self.addSP()

    def DragSPADD(self, event):
        self.spPosition = event.pos()
        self.addSP()

    def stopSPAdd(self, event):
        """Finish labelling sp"""
        self.imageLabel.setMouseTracking(False)
        self.sp_queue = []
        #self.confirmEdit()
        #self.showImage()

    def labelSPAdd(self):
        """Set mouse action when adding to superpixel segments"""
        if self.sender().isChecked():
            self.mouseAction.setChecked(True)
            self.setMouseAction()
            
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

        factor = self.lastSpinboxValue * 1.0 / 100
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
        elif self.spAddAction.isChecked():
            label, x, y = self.getLabel(self.spPosition)
            indices = np.argwhere(self.spSegments == label)
            for i in range(0, len(indices)):
                self.outputMask[indices[i][0]][indices[i][1]] = [self.currentColor.red(),
                        self.currentColor.green(), self.currentColor.blue()]
            self.updateStatus("Superpixel at x:%d y:%d added" % (x, y))
            #self.updateStatus("Superpixel at x:%d y:%d added, label:%d" % (x, y, label))
        elif self.spSubAction.isChecked():
            label, x, y = self.getLabel(self.spPosition)
            indices = np.argwhere(self.spSegments == label)
            for i in range(0, len(indices)):
                self.outputMask[indices[i][0]][indices[i][1]] = [self.backgroundColor.red(),
                                                                 self.backgroundColor.green(),
                                                                 self.backgroundColor.blue()]
            self.updateStatus("Superpixel at x:%d y:%d removed" % (x, y))
        elif self.clusterAddAction.isChecked():
            self.updateStatus("updating add cluster")
            label, x, y = self.getLabel(self.clusterPosition)
            indices = np.argwhere(self.regionSegments == label)
            for i in range(0, len(indices)):
                self.outputMask[indices[i][0]][indices[i][1]] = [self.currentColor.red(),
                                                                 self.currentColor.green(),
                                                                 self.currentColor.blue()]
        elif self.clusterSubAction.isChecked():
            self.updateStatus("updating sub cluster")
            label, x, y = self.getLabel(self.clusterPosition)
            indices = np.argwhere(self.regionSegments == label)
            for i in range(0, len(indices)):
                self.outputMask[indices[i][0]][indices[i][1]] = [self.backgroundColor.red(),
                                                                 self.backgroundColor.green(),
                                                                 self.backgroundColor.blue()]
        if not self.dirty:
            self.setDirty()
        self.undoAction.setEnabled(True)

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

    def chooseColor(self):
        """Use a palette to choose labelling color"""
        color = self.colorDialog.getColor(self.currentColor,
                                "Choose labelling color", config.DEFAULT_FILLING_COLOR)
        if color:
            self.currentColor = color
            self.updateStatus("Choose a new color")
            icon = QPixmap(50, 50)
            icon.fill(self.currentColor)
            self.paletteAction.setIcon(QIcon(icon))
            if not self.colorLabelBar.isHidden() and \
                            self.labelsGroup.checkedAction() is not None:
                self.labelsGroup.checkedAction().setChecked(False)
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

    def showLog(self):
        self.logDockWidget.show()

    def hideLog(self):
        self.logDockWidget.hide()

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
                                """<b>Image Annotation Tool</b> v %s
                                <p>This application can be used to annotate images,
                                especially for plantations.
                                <p>Python %s - Qt %s - PyQt %s on %s
                                <p>By Jingxiao Ma, School of Computer Science, 
                                University of Nottingham\nEmail: psyjm7@nottingham.ac.uk"""
                                %( __version__, platform.python_version(), QT_VERSION_STR,
                                PYQT_VERSION_STR, platform.system()))

    def helpHelp(self):
        QMessageBox.about(self, "Help -- Image Annotation Tool",
                          """<b>Usage:</b>
                          <p>1. Open a directory or an image.\n
                          2. Choose color.\n
                          3. Choose a shape and region.\n
                          4. Confirm, unlabel or perform flood-fill.\n
                          5. Save image when finish annotating.
                          <p>For details, please check user manual in docs folder.""")

    @staticmethod
    def normalise(norm_input, max_num=1):
        # with warnings.catch_warnings():
        #    warnings.simplefilter("ignore")
        norm_input = norm_input.astype(float)
        min_ = abs(np.min(norm_input))
        max_ = abs(np.max(norm_input))
        i_max = min_ if min_ > max_ else max_
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
