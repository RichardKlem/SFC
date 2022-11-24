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
from PySide2.QtGui import QIntValidator
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


class Ui_MainWindow(object):

    def __init__(self):
        self.common_canvas = None
        self.dual_canvas1 = None
        self._line1 = None
        self._line2 = None
        self.dual_canvas2 = None
        self.started = False
        self.ax1 = None
        self.ax2 = None
        self._timer_common = None
        self._timer1 = None
        self._timer2 = None
        self.common_canvas: FigureCanvas
        self.xs = []
        self.ys = []

    def on_click_reset(self):
        # self.stop()
        self.itersLineEdit.setText("10")
        self.side_by_side_change()
        # self.clear_axes()
        # self.common_canvas.clear_canvas # TODO

    def on_click_go(self):
        self._timer1.interval = self.itersLineEdit.text()

    def stop(self):
        self.started = False
        if self._timer1:
            self._timer1.stop()
        if self._timer2:
            self._timer2.stop()
        if self._timer_common:
            self._timer_common.stop()

    def start(self):
        iters = int(self.itersLineEdit.text())
        full = False
        alfa = 0.001
        net = Model([Linear(784, 20), Sigmoid(), Linear(20, 10), Softmax()], opt=None, cost=CCE())
        net_opt = Model([Linear(784, 20), Sigmoid(), Linear(20, 10), Softmax()], opt="amsgrad",
                        cost=CCE())
        if full:
            data_train_file = ".data_train.pickle"
            data_test_file = ".data_train.pickle"
        else:
            data_train_file = ".data_train_small.pickle"
            data_test_file = ".data_train_small.pickle"

        ys, ys_opt = [], []

        # Set up a Line2D.
        # self._line1, = self.ax1.plot(np.linspace(0, 1, int(time.time())),
        #                                     np.sin(np.linspace(0, 1, int(time.time()))))
        self.started = True
        if self.chb_side_by_side.isChecked():
            self.ax1 = self.dual_canvas1.figure.subplots()
            self.ax2 = self.dual_canvas2.figure.subplots()
            for i in range(1, iters):
                xs = range(i)
                y, y_opt, val, val_opt = run(net, net_opt, data_train_file, data_test_file, alfa)
                ys.append(y)
                ys_opt.append(y_opt)
                self._line1, = self.ax1.plot(xs, ys)
                self._line2, = self.ax2.plot(xs, ys_opt)
                self._line1.figure.canvas.draw()
                self._line2.figure.canvas.draw()
            y, y_opt, val, val_opt = run(net, net_opt, data_train_file, data_test_file, alfa, test=True)
            ys.append(y)
            ys_opt.append(y_opt)
            xs = range(iters)
            self._line1, = self.ax1.plot(xs, ys)
            self._line2, = self.ax2.plot(xs, ys_opt)
            self._line1.figure.canvas.draw()
            self._line2.figure.canvas.draw()

        else:
            self.ax1 = self.common_canvas.figure.subplots()
            for i in range(1, iters):
                xs = range(i)
                y, y_opt, val, val_opt = run(net, net_opt, data_train_file, data_test_file, alfa)
                ys.append(y)
                ys_opt.append(y_opt)
                self._line1, = self.ax1.plot(xs, ys)
                self._line2, = self.ax1.plot(xs, ys_opt)
                self._line1.figure.canvas.draw()
                self._line2.figure.canvas.draw()
            y, y_opt, val, val_opt = run(net, net_opt, data_train_file, data_test_file, alfa, test=True)
            ys.append(y)
            ys_opt.append(y_opt)
            xs = range(iters)
            self._line1, = self.ax1.plot(xs, ys)
            self._line2, = self.ax1.plot(xs, ys_opt)
            # self._line1.set_data(t, np.sin(t + time.time()))
            self._line1.figure.canvas.draw()
            self._line2.figure.canvas.draw()

    def create_graphs(self):
        self.side_by_side_change()

    def side_by_side_change(self):
        self.stop()
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

        self.itersLineEdit = QLineEdit(self.centralwidget)
        self.itersLineEdit.setObjectName(u"itersLineEdit")
        self.itersLineEdit.setGeometry(QRect(10, 10, 60, 25))
        validator = QIntValidator(1, 1000)
        self.itersLineEdit.setValidator(validator)

        self.startButton = QPushButton(self.centralwidget)
        self.startButton.setObjectName(u"startButton")
        self.startButton.setGeometry(QRect(150, 10, 60, 25))
        self.startButton.clicked.connect(self.start)

        self.stopButton = QPushButton(text="Stop", parent=self.centralwidget)
        self.stopButton.setObjectName(u"startButton")
        self.stopButton.setGeometry(QRect(220, 10, 60, 25))
        self.stopButton.clicked.connect(self.stop)

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
        self.itersLineEdit.setToolTip(QCoreApplication.translate("MainWindow", u"Hello", None))
#endif // QT_CONFIG(tooltip)
        self.itersLineEdit.setText(QCoreApplication.translate("MainWindow", u"10", None))
        self.startButton.setText(QCoreApplication.translate("MainWindow", u"Start", None))
        self.btn_reset.setText(QCoreApplication.translate("MainWindow", u"Reset", None))
        self.chb_side_by_side.setText(QCoreApplication.translate("MainWindow", u"Side by side", None))
