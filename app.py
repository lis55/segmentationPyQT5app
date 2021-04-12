# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'load_image_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QFileDialog)
from PyQt5.QtGui import QPixmap
from model import *
import numpy as np
import qimage2ndarray

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(700, 700)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(160, 590, 75, 21))
        self.pushButton.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(280, 590, 131, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(70, 30, 550, 550))
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.label_2 = QtWidgets.QLabel(self.tab)
        self.label_2.setGeometry(QtCore.QRect(10, 10, 512, 512))
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.label = QtWidgets.QLabel(self.tab_2)
        self.label.setGeometry(QtCore.QRect(10, 10, 512, 512))
        self.label.setText("")
        self.label.setObjectName("label")
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_7 = QtWidgets.QWidget()
        self.tab_7.setObjectName("tab_7")
        self.label_7 = QtWidgets.QLabel(self.tab_7)
        self.label_7.setGeometry(QtCore.QRect(10, 10, 512, 512))
        self.label_7.setText("")
        self.label_7.setObjectName("label_7")
        self.tabWidget.addTab(self.tab_7, "")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(450, 590, 75, 23))
        self.pushButton_3.setObjectName("pushButton_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 700, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        self.pushButton.clicked.connect(self.open_browser)
        self.pushButton_2.clicked.connect(self.run_segmentation)
        self.pushButton_3.clicked.connect(self.label_7.clear)
        self.pushButton_3.clicked.connect(self.label.clear)
        self.pushButton_3.clicked.connect(self.label_2.clear)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Lil Lis noob app"))
        self.pushButton.setText(_translate("MainWindow", "Insert files"))
        self.pushButton_2.setText(_translate("MainWindow", "Generate segmentation"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Image"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Mask"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_7), _translate("MainWindow", "Overlay"))
        self.pushButton_3.setText(_translate("MainWindow", "clear"))

    def open_browser(self):
        """Launch the browser dialog."""
        dlg = QFileDialog()
        fname = QFileDialog.getOpenFileName(dlg, 'Open file', 'c:\\', "Image files (*.jpg *.gif *.png *.dcm)")
        batch=load_dicom(fname[0])
        batch = batch/np.max(batch)
        self.batch = np.zeros((1,batch.shape[0],batch.shape[1],batch.shape[2]))
        self.batch[0,:,:,:] = batch
        yourQImage = qimage2ndarray.array2qimage(batch*255)
        self.label_2.setPixmap(QPixmap(yourQImage))
        #self.show_image(fname)
        #dlg.exec()

    def run_segmentation(self):
        """Launch the browser dialog."""
        model = unet()
        model.load_weights('C:/Users/lis/Desktop/overfitting/relu+dice/unet_ThighOuterSurfaceval.hdf5')
        mask = model.predict(self.batch, batch_size=1, verbose=0, steps=None, callbacks=None, max_queue_size=10,workers=1, use_multiprocessing=False)
        mask = mask[0, :, :, :]
        mask[mask <= 0.1] = 0
        yourQImage = qimage2ndarray.array2qimage(mask*255)
        self.label.setPixmap(QPixmap(yourQImage))
        pix = make_overlay(mask[:,:,0],self.batch[0,:,:,0])
        pix = np.array(pix)
        yourQImage = qimage2ndarray.array2qimage(pix)
        self.label_7.setPixmap(QPixmap(yourQImage))
        #self.show_image(fname)
        #dlg.exec()

    def show_image(self,fname):
        self.label.setText(fname[0])
        pixmap = QPixmap(fname[0])
        pixmap = pixmap.scaled(self.label.size(),QtCore.Qt.KeepAspectRatio)
        self.label.setPixmap(pixmap)
        # Optional, resize window to image size
        #self.resize(pixmap.width(), pixmap.height())

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)

    MainWindow.show()
    sys.exit(app.exec_())