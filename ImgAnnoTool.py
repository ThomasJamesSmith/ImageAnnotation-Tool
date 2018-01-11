# -*- coding: utf-8 -*-
# This is Image Annotation Tool for annotating plantations.
# Created by Jingxiao Ma.
# Languages: python 2 (2.7)
# Sys requirement: Linux / Windows 7 / OS X 10.8 or later versions
# Package requirement: PyQt 4 / OpenCv 2 / numpy


import os
import platform
import sys
import qrc_resources
import numpy as np
import cv2
import config
from colorDialog import *
from FloodFillConfig import *
import sip
# RECOMMAND: Use PyQt4
try:
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *
except ImportError:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *


__version__ = '1.0'

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
        self.isHiding = False
        self.isLaballing = False
        self.finishChoosingArea = False

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

        self.hideOriginalAction = self.createAction("&Hide\nOriginal", self.hideOriginalImage, "Ctrl+H",
                                                    "hide", "Hide original image", True, "toggled(bool)")
        self.hideOriginalAction.setChecked(False)

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
                                 (self.mouseAction, True))

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
        self.addActions(viewMenu, (zoomInAction, zoomOutAction, self.hideOriginalAction,
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
        self.toolBarActions_2 = (zoomOutAction, self.hideOriginalAction, None,
                                 self.paletteAction, self.confirmAction, self.deleteAction,
                                 self.floodFillAction, None, self.mouseAction, self.rectLabelAction,
                                 self.ellipseLabelAction, self.polygonLabelAction)
        self.addActions(self.toolBar, self.toolBarActions_1)
        self.toolBar.addWidget(self.zoomSpinBox)
        self.addActions(self.toolBar, self.toolBarActions_2)

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


    def createAction(self, text, slot=None, shortcut=None,
                     icon=None, tip=None, checkable=False,
                     signal="triggered()"):
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
                            (i + 1, QFileInfo(fname).fileName()), self)
                action.setData(QVariant(fname))
                self.connect(action, SIGNAL("triggered()"),
                             self.loadImage)
                self.fileMenu.addAction(action)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.fileMenuActions[-1])


    def updateToolBar(self):
        """Update toolbar to show color labels"""
        labels = config.getLabelColor(self.filename)

        if labels is None:
            self.colorLabelBar.hide()
        else:
            self.colorLabelBar.clear()
            self.colorLabelBar.show()
            self.labelsGroup = QActionGroup(self)
            self.colorLabelDict= {}
            for label in labels.keys():
                action = self.createAction(label, self.chooseColor_2, None,
                                                       None, "Color the label with user specified color",
                                                       True, "toggled(bool)")
                icon = QPixmap(50, 50)
                icon.fill(labels[label])
                action.setIcon(QIcon(icon))
                self.colorLabelBar.addAction(action)
                self.labelsGroup.addAction(action)
                self.colorLabelDict[action] = labels[label]


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
        if self.filename is not None:
            self.setWindowTitle("Image Annotation Tool - %s[*]" % \
                                os.path.basename(self.filename))
        else:
            self.setWindowTitle("Image Annotation Tool[*]")
        self.setWindowModified(self.dirty)


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



    def dirOpen(self):
        """Open a directory and load the first image"""
        if not self.okToContinue():
            return

        dir = os.path.dirname(self.filename) \
            if self.filename is not None else "."
        dirname = unicode(QFileDialog.getExistingDirectory(self,
                                "Image Annotation Tool - Select Directory", dir))
        if dirname:
            self.updateStatus("Open directory: %s" % dirname)
            self.filepath = dirname
            self.allImages = self.scanAllImages(dirname)
            self.fileListWidget.clear()
            for imgPath in self.allImages:
                filename = os.path.basename(imgPath)
                item = QListWidgetItem(filename)
                self.fileListWidget.addItem(item)

            # Open first file
            if len(self.allImages) > 0:
                self.updateStatus("Open directory: %s" % dirname)
                self.loadImage(self.allImages[0])
                self.updateToolBar()
                self.colorListWidget(self.allImages[0])
            else:
                QMessageBox.warning(self, 'Error', "[ERROR]: No images in %s" % dirname)


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
				   
        vid_formats = [u'*.mp4']
        fname = unicode(QFileDialog.getOpenFileName(self,
                            "Image Annotation Tool - Choose Image", dir,
                            "Image files (%s) ;; Video files (%s)" % (" ".join(img_formats), " ".join(vid_formats))))
        if fname.endswith(".mp4"):
            self.loadVideo(fname)
        else:
            self.fileListWidget.clear()
            self.loadImage(fname)
            self.updateToolBar()

    def loadVideo(self, fname=None):
		fsplit = fname.split("/")
		nsplit = fsplit[len(fsplit)-1].split(".")
		fsplit[len(fsplit)-1]=nsplit[0]
		dirname="/".join(fsplit) + "/"
		
		msg = "Do you want to reverse the video?"
		reply = QMessageBox.question(self, 'Message',
						msg, QMessageBox.Yes, QMessageBox.No)

		reverse = True if reply == QMessageBox.Yes else False
		
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
			vidcap = cv2.VideoCapture(fname)
			success,image = vidcap.read()
			success = True
			frames = []
			while success:
				success,image = vidcap.read()
				if success:
					image2 = cv2.resize(image,(640,360), interpolation = cv2.INTER_CUBIC)
					frames.insert(len(frames),image2)
			
			start = 0
			end = len(frames)-1
			step = 1
			name = 0;
			
			if reverse:
				self.updateStatus("Video Reversed")
				start = end-1
				end = -1
				step = -1
			else:		
				self.updateStatus("Video Not Reversed")
			
			for i in range(start, end, step):
				cv2.imwrite(dirname + "%05d.jpg" % name, frames[i])
				name += 1
		else:
			self.updateStatus("Action stopped: Please rename file.")
			return
		
		if dirname:
			self.updateStatus("Open directory: %s" % dirname)
			self.filepath = dirname
			self.allImages = self.scanAllImages(dirname)
			self.fileListWidget.clear()
			for imgPath in self.allImages:
				filename = os.path.basename(imgPath)
				item = QListWidgetItem(filename)
				self.fileListWidget.addItem(item)

			if len(self.allImages) > 0:
				self.updateStatus("Open directory: %s" % dirname)
				self.loadImage(self.allImages[0])
				self.updateToolBar()
				self.colorListWidget(self.allImages[0])
			else:
				QMessageBox.warning(self, 'Error', "[ERROR]: No images in %s" % dirname)
		
	

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
                self.filename = fname
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
        grayImage = cv2.cvtColor(self.outputMask, cv2.COLOR_RGB2GRAY)
        ret, mask = cv2.threshold(grayImage, 2, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        labels1 = cv2.bitwise_and(self.outputMask, self.outputMask, mask = mask)
        labels2 = cv2.bitwise_and(self.cvimage, self.cvimage, mask = mask)
        origin = cv2.bitwise_and(self.cvimage, self.cvimage, mask = mask_inv)
        labels = cv2.addWeighted(labels1, 0.5, labels2, 0.5, 0)
        dst = cv2.add(labels, origin)
        # If trying to apply floodfill

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
            if self.hideOriginalAction.isChecked():
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


    def hideOriginalImage(self):
        """Hide original image and only show """
        if self.hideOriginalAction.isChecked():
            self.isHiding = True
        else:
            self.isHiding = False
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


    def labelRectOrEllipse(self):
        """Set mouse action when labelling rectangle or ellipse"""
        if self.sender().isChecked():
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
            reply = QMessageBox.question(self, "Exit", "You are going to leave the " +
                                        "Image Annotation Tool. Are you sure?",
                                        QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
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
