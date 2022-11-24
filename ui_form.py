# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

import sys
import time

import numpy as np

from PySide2.QtCore import *
from PySide2.QtGui import QIntValidator, QDoubleValidator
from PySide2.QtWidgets import *

from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as
NavigationToolbar)
from matplotlib.figure import Figure

from cce import CrossEntropy as CCE
from linear import Linear
from model import Model
from sigmoid import Sigmoid
from softmax import Softmax
from train import run
import asyncio


class Ui_MainWindow(object):

    def __init__(self):
        self.steps = 10
        self.steps_done = 0
        self.alfa = 0.001
        self.net = None
        self.net_opt = None
        self.full = False
        self.started = False
        self.common_canvas: FigureCanvas = None
        self.dual_canvas1 = None
        self.dual_canvas2 = None
        self._line1 = None
        self._line2 = None
        self.ax1 = None
        self.ax2 = None
        self.xs = []
        self.ys = []
        self.ys_opt = []

    def on_click_reset(self):
        # self.stop()
        self.stepsLineEdit.setText("10")
        self.side_by_side_change()
        self.ys, self.ys_opt = [], []
        self.steps_done = 0
        self.started = False
        # self.clear_axes()
        # self.common_canvas.clear_canvas # TODO

    # def on_click_go(self):
    #     self._timer1.interval = self.stepsLineEdit.text()

    # def stop(self):
    #     self.started = False
    #     if self._timer1:
    #         self._timer1.stop()
    #     if self._timer2:
    #         self._timer2.stop()
    #     if self._timer_common:
    #         self._timer_common.stop()

    def init_model(self):
        self.steps = int(self.stepsLineEdit.text())
        self.alfa = float(self.alfaLineEdit.text())

        self.net = Model([Linear(784, 20), Sigmoid(), Linear(20, 10), Softmax()], opt=None,
                         cost=CCE())
        self.net_opt = Model([Linear(784, 20), Sigmoid(), Linear(20, 10), Softmax()],
                             opt="amsgrad",
                             cost=CCE())
        if self.full:
            self.data_train_file = ".data_train.pickle"
            self.data_test_file = ".data_test.pickle"
        else:
            self.data_train_file = ".data_train_small.pickle"
            self.data_test_file = ".data_test_small.pickle"

    def step(self):
        if self.steps_done == 0:
            self.init_model()
        self.side_by_side_change()
        if self.chb_side_by_side.isChecked():
            self.ax1 = self.dual_canvas1.figure.subplots()
            self.ax2 = self.dual_canvas2.figure.subplots()
            self.update_data(self.ys, self.ys_opt, self.ax1, self.ax2)
            y, y_opt, val, val_opt = run(self.net, self.net_opt, self.data_train_file,
                                         self.data_test_file, self.alfa, test=True)
            self.steps_done += 1
            self.ys.append(y)
            self.ys_opt.append(y_opt)
            xs = range(self.steps_done)
            self._line1, = self.ax1.plot(xs, self.ys)
            self._line2, = self.ax2.plot(xs, self.ys_opt)
            self._line1.figure.canvas.draw()
            self._line2.figure.canvas.draw()

        else:
            self.ax1 = self.common_canvas.figure.subplots()
            self.update_data(self.ys, self.ys_opt, self.ax1, self.ax1)
            y, y_opt, val, val_opt = run(self.net, self.net_opt, self.data_train_file,
                                         self.data_test_file, self.alfa, test=True)
            self.steps_done += 1
            self.ys.append(y)
            self.ys_opt.append(y_opt)
            xs = range(self.steps_done)
            self._line1, = self.ax1.plot(xs, self.ys)
            self._line2, = self.ax1.plot(xs, self.ys_opt)
            # self._line1.set_data(t, np.sin(t + time.time()))
            self._line1.figure.canvas.draw()
            self._line2.figure.canvas.draw()

    def start(self):
        if self.steps_done == 0:
            self.init_model()
        self.side_by_side_change()
        self.started = True
        if self.chb_side_by_side.isChecked():
            self.ax1 = self.dual_canvas1.figure.subplots()
            self.ax2 = self.dual_canvas2.figure.subplots()
            for i in range(1, self.steps):
                self.update_data(self.ys, self.ys_opt, self.ax1, self.ax2)
            y, y_opt, val, val_opt = run(self.net, self.net_opt, self.data_train_file,
                                         self.data_test_file, self.alfa, test=True)
            self.steps_done += 1
            self.ys.append(y)
            self.ys_opt.append(y_opt)
            xs = range(self.steps)
            self._line1, = self.ax1.plot(xs, self.ys)
            self._line2, = self.ax2.plot(xs, self.ys_opt)
            self._line1.figure.canvas.draw()
            self._line2.figure.canvas.draw()

        else:
            self.ax1 = self.common_canvas.figure.subplots()
            for i in range(1, self.steps):
                self.update_data(self.ys, self.ys_opt, self.ax1, self.ax1)
            y, y_opt, val, val_opt = run(self.net, self.net_opt, self.data_train_file,
                                         self.data_test_file, self.alfa, test=True)
            self.steps_done += 1
            self.ys.append(y)
            self.ys_opt.append(y_opt)
            xs = range(self.steps)
            self._line1, = self.ax1.plot(xs, self.ys)
            self._line2, = self.ax1.plot(xs, self.ys_opt)
            # self._line1.set_data(t, np.sin(t + time.time()))
            self._line1.figure.canvas.draw()
            self._line2.figure.canvas.draw()

    def update_data(self, ys, ys_opt, ax1, ax2):
        # self.side_by_side_change()

        y, y_opt, val, val_opt = run(self.net, self.net_opt, self.data_train_file, self.data_test_file, self.alfa)
        ys.append(y)
        ys_opt.append(y_opt)
        self.steps_done += 1
        xs = range(self.steps_done)
        self._line1, = ax1.plot(xs, ys)
        self._line2, = ax2.plot(xs, ys_opt)
        self._line1.figure.canvas.draw()
        self._line2.figure.canvas.draw()

    def create_graphs(self):
        self.side_by_side_change()

    def side_by_side_change(self):
        # self.stop()
        self.clear_axes()
        if not self.chb_side_by_side.isChecked():
            # Create single chart
            if self.common_canvas is None:
                self.common_canvas = FigureCanvas(Figure(figsize=(5, 3)))
            self.horizontalLayout.addWidget(self.common_canvas)
        # checked -> user wants two plots.
        else:
            if self.dual_canvas1 is None:
                self.dual_canvas1 = FigureCanvas(Figure(figsize=(5, 3)))
            if self.dual_canvas2 is None:
                self.dual_canvas2 = FigureCanvas(Figure(figsize=(5, 3)))
            self.horizontalLayout.addWidget(self.dual_canvas1)
            self.horizontalLayout.addWidget(self.dual_canvas2)

    def clear_axes(self):
        # Remove charts.
        if self.dual_canvas1:
            self.horizontalLayout.removeWidget(self.dual_canvas1)
            self.dual_canvas1.deleteLater()
            self.dual_canvas1 = None
        if self.dual_canvas2:
            self.horizontalLayout.removeWidget(self.dual_canvas2)
            self.dual_canvas2.deleteLater()
            self.dual_canvas2 = None
        if self.common_canvas:
            self.horizontalLayout.removeWidget(self.common_canvas)
            self.common_canvas.deleteLater()
            self.common_canvas = None

    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(800, 437)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")

        self.alfatext = QLabel(self.centralwidget)
        self.alfatext.setObjectName(u"alfaText")
        self.alfatext.setGeometry(QRect(10, 10, 20, 25))
        self.alfatext.setText("α:")

        self.alfaLineEdit = QLineEdit(self.centralwidget)
        self.alfaLineEdit.setObjectName(u"alfaLineEdit")
        self.alfaLineEdit.setGeometry(QRect(25, 10, 45, 25))
        alfa_validator = QDoubleValidator(0., 10., 5)
        alfa_validator.setLocale(QLocale("en_US"))
        self.alfaLineEdit.setValidator(alfa_validator)

        self.stepsLabel = QLabel(self.centralwidget)
        self.stepsLabel.setObjectName(u"stepsLabel")
        self.stepsLabel.setGeometry(QRect(75, 10, 60, 25))
        self.stepsLabel.setText("# of steps:")

        self.stepsLineEdit = QLineEdit(self.centralwidget)
        self.stepsLineEdit.setObjectName(u"stepsLineEdit")
        self.stepsLineEdit.setGeometry(QRect(138, 10, 32, 25))
        validator = QIntValidator(1, 100)
        self.stepsLineEdit.setValidator(validator)

        self.stepButton = QPushButton(self.centralwidget)
        self.stepButton.setObjectName(u"stepButton")
        self.stepButton.setGeometry(QRect(175, 10, 50, 25))
        self.stepButton.clicked.connect(self.step)

        self.startButton = QPushButton(self.centralwidget)
        self.startButton.setObjectName(u"startButton")
        self.startButton.setGeometry(QRect(230, 10, 60, 25))
        self.startButton.clicked.connect(self.start)

        # self.stopButton = QPushButton(text="Stop", parent=self.centralwidget)
        # self.stopButton.setObjectName(u"startButton")
        # self.stopButton.setGeometry(QRect(270, 10, 60, 25))
        # self.stopButton.clicked.connect(self.stop)

        self.horizontalLayoutWidget = QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setObjectName(u"horizontalLayoutWidget")
        self.horizontalLayoutWidget.setGeometry(QRect(10, 60, 771, 331))

        self.horizontalLayout = QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)

        self.common_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        self.dual_canvas1 = FigureCanvas(Figure(figsize=(5, 3)))
        self.dual_canvas2 = FigureCanvas(Figure(figsize=(5, 3)))

        self.btn_reset = QPushButton(self.centralwidget)
        self.btn_reset.setObjectName(u"pushButton_2")
        self.btn_reset.setGeometry(QRect(700, 10, 71, 25))
        self.btn_reset.clicked.connect(self.on_click_reset)

        self.chb_side_by_side = QCheckBox(self.centralwidget)
        self.chb_side_by_side.setObjectName(u"checkBox")
        self.chb_side_by_side.setGeometry(QRect(350, 10, 111, 23))
        self.chb_side_by_side.setCheckState(Qt.Unchecked)
        self.chb_side_by_side.toggled.connect(self.side_by_side_change)

        self.create_graphs()

        # self.horizontalLayout.addWidget(NavigationToolbar(dual_canvas1, self))

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
#if QT_CONFIG(tooltip)
        self.alfaLineEdit.setToolTip("Learning rate Alfa")
        self.stepsLineEdit.setToolTip("Number of iterations")
#endif // QT_CONFIG(tooltip)
        self.stepsLineEdit.setText("10")
        self.alfaLineEdit.setText("0.001")
        self.startButton.setText("All steps")
        self.stepButton.setText("1 step")
        self.btn_reset.setText(QCoreApplication.translate("MainWindow", u"Reset", None))
        self.chb_side_by_side.setText(QCoreApplication.translate("MainWindow", u"Side by side", None))
