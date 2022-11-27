from matplotlib.backends.backend_qt5agg import (FigureCanvas)
from matplotlib.figure import Figure
from PyQt5.QtCore import *
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtWidgets import *

from cce import CrossEntropy as CCE
from linear import Linear
from model import Model
from sigmoid import Sigmoid
from softmax import Softmax
from train import run


class UIMainWindow(object):

    def __init__(self):
        self.data_train_file = ""
        self.data_test_file = ""
        self.val1 = None
        self.val2 = None
        self.chb_show_val = None
        self.chb_side_by_side = None
        self.btn_reset = None
        self.opt_dd1 = None
        self.opt_dd2 = None
        self.opt1 = None
        self.opt2 = "amsgrad"
        self.opt_dd1: QComboBox
        self.opt_dd2: QComboBox
        self.showVal = False
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
        self.stepsLineEdit.setText("10")
        self.side_by_side_change()
        self.ys, self.ys_opt = [], []
        self.steps_done = 0
        self.started = False

    def init_model(self):
        self.steps = int(self.stepsLineEdit.text())
        self.alfa = float(self.alfaLineEdit.text())

        self.net = Model([Linear(784, 20), Sigmoid(), Linear(20, 10), Softmax()], opt=self.opt1,
                         cost=CCE())
        self.net_opt = Model([Linear(784, 20), Sigmoid(), Linear(20, 10), Softmax()],
                             opt=self.opt2, cost=CCE())
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
            if self.showVal:
                _, _, val, val_opt = run(self.net, self.net_opt, self.data_train_file,
                                         self.data_test_file, self.alfa, test=True)
                self.val1.setText("Test loss opt 1: " + str(val))
                self.val2.setText("Test loss opt 2: " + str(val_opt))
            xs = range(self.steps_done)
            self._line1, = self.ax1.plot(xs, self.ys, marker='.')
            self._line2, = self.ax2.plot(xs, self.ys_opt, marker='.')
            self._line1.figure.canvas.draw()
            self._line2.figure.canvas.draw()

            self.ax2.set_xticks([int(x) for x in xs])
            self.ax2.set_xlabel("Epochs")
            self.ax2.set_ylabel("Loss")

        else:
            self.ax1 = self.common_canvas.figure.subplots()
            self.update_data(self.ys, self.ys_opt, self.ax1, self.ax1)
            if self.showVal:
                _, _, val, val_opt = run(self.net, self.net_opt, self.data_train_file,
                                         self.data_test_file, self.alfa, test=True)
                self.val1.setText("Test loss opt 1: " + str(val))
                self.val2.setText("Test loss opt 2: " + str(val_opt))
            xs = range(self.steps_done)
            self._line1, = self.ax1.plot(xs, self.ys, marker='.')
            self._line2, = self.ax1.plot(xs, self.ys_opt, marker='.')
            # self._line1.set_data(t, np.sin(t + time.time()))
            self._line1.figure.canvas.draw()
            self._line2.figure.canvas.draw()
        self.ax1.set_xticks([int(x) for x in xs])
        self.ax1.set_xlabel("Epochs")
        self.ax1.set_ylabel("Loss")

    def start(self):
        if self.steps_done == 0:
            self.init_model()
        self.side_by_side_change()
        self.started = True
        if self.chb_side_by_side.isChecked():
            self.ax1 = self.dual_canvas1.figure.subplots()
            self.ax2 = self.dual_canvas2.figure.subplots()
            for i in range(0, self.steps):
                self.update_data(self.ys, self.ys_opt, self.ax1, self.ax2)
            if self.showVal:
                _, _, val, val_opt = run(self.net, self.net_opt, self.data_train_file,
                                         self.data_test_file, self.alfa, test=True)
                self.val1.setText("Test loss opt 1: " + str(val))
                self.val2.setText("Test loss opt 2: " + str(val_opt))
            xs = range(self.steps_done)
            self._line1, = self.ax1.plot(xs, self.ys, marker='.')
            self._line2, = self.ax2.plot(xs, self.ys_opt, marker='.')
            self._line1.figure.canvas.draw()
            self._line2.figure.canvas.draw()

        else:
            self.ax1 = self.common_canvas.figure.subplots()
            for i in range(0, self.steps):
                self.update_data(self.ys, self.ys_opt, self.ax1, self.ax1)
            if self.showVal:
                _, _, val, val_opt = run(self.net, self.net_opt, self.data_train_file,
                                         self.data_test_file, self.alfa, test=True)
                self.val1.setText("Test loss opt 1: " + str(val))
                self.val2.setText("Test loss opt 2: " + str(val_opt))
            xs = range(self.steps_done)
            self._line1, = self.ax1.plot(xs, self.ys, marker='.')
            self._line2, = self.ax1.plot(xs, self.ys_opt, marker='.')
            self._line1.figure.canvas.draw()
            self._line2.figure.canvas.draw()

    def update_data(self, ys, ys_opt, ax1, ax2):
        y, y_opt, val, val_opt = run(self.net, self.net_opt, self.data_train_file,
                                     self.data_test_file, self.alfa)
        ys.append(y)
        ys_opt.append(y_opt)
        self.steps_done += 1
        xs = range(self.steps_done)
        self._line1, = ax1.plot(xs, ys, marker='.')
        self._line2, = ax2.plot(xs, ys_opt, marker='.')
        self._line1.figure.canvas.draw()
        self._line2.figure.canvas.draw()

    def create_graphs(self):
        self.side_by_side_change()

    def side_by_side_change(self):
        self.clear_axes()
        if not self.chb_side_by_side.isChecked():
            # Create single chart
            if self.common_canvas is None:
                self.common_canvas = FigureCanvas(Figure(figsize=(5, 3), tight_layout=True))
            self.horizontalLayout.addWidget(self.common_canvas)
        # checked -> user wants two plots.
        else:
            if self.dual_canvas1 is None:
                self.dual_canvas1 = FigureCanvas(Figure(figsize=(5, 3), tight_layout=True))
            if self.dual_canvas2 is None:
                self.dual_canvas2 = FigureCanvas(Figure(figsize=(5, 3), tight_layout=True))
            self.horizontalLayout.addWidget(self.dual_canvas1)
            self.horizontalLayout.addWidget(self.dual_canvas2)

    def show_val_change(self):
        self.showVal = self.chb_show_val.isChecked()
        if not self.showVal:
            self.val1.setHidden(True)
            self.val2.setHidden(True)
        else:
            self.val1.setHidden(False)
            self.val2.setHidden(False)

    def opt_change1(self):
        if self.opt_dd1.currentText() == "AmsGrad":
            self.opt1 = 'amsgrad'
        elif self.opt_dd1.currentText() == "Adam":
            self.opt1 = "adam"
        else:
            self.opt1 = None

    def opt_change2(self):
        if self.opt_dd2.currentText() == "AmsGrad":
            self.opt2 = 'amsgrad'
        elif self.opt_dd2.currentText() == "Adam":
            self.opt2 = "adam"
        else:
            self.opt2 = None

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

    def setupUi(self, main_window):
        if not main_window.objectName():
            main_window.setObjectName(u"UIMainWindow")
        main_window.resize(800, 437)
        self.centralwidget = QWidget(main_window)
        self.centralwidget.setObjectName(u"centralwidget")

        self.alfatext = QLabel(self.centralwidget)
        self.alfatext.setObjectName(u"alfaText")
        self.alfatext.setGeometry(QRect(10, 10, 20, 25))
        self.alfatext.setText("α:")

        self.alfaLineEdit = QLineEdit(self.centralwidget)
        self.alfaLineEdit.setObjectName(u"alfaLineEdit")
        self.alfaLineEdit.setGeometry(QRect(25, 10, 45, 25))
        alfa_validator = QDoubleValidator(bottom=0., top=1., decimals=5)
        alfa_validator.setLocale(QLocale("en_US"))
        self.alfaLineEdit.setValidator(alfa_validator)

        self.stepsLabel = QLabel(self.centralwidget)
        self.stepsLabel.setObjectName(u"stepsLabel")
        self.stepsLabel.setGeometry(QRect(75, 10, 70, 25))
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

        self.horizontalLayoutWidget = QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setObjectName(u"horizontalLayoutWidget")
        self.horizontalLayoutWidget.setGeometry(QRect(10, 80, 771, 331))

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
        self.chb_side_by_side.setGeometry(QRect(320, 10, 111, 23))
        self.chb_side_by_side.setCheckState(Qt.Unchecked)
        self.chb_side_by_side.toggled.connect(self.side_by_side_change)

        self.chb_show_val = QCheckBox(self.centralwidget)
        self.chb_show_val.setObjectName(u"chb_show_val")
        self.chb_show_val.setGeometry(QRect(441, 10, 111, 23))
        self.chb_show_val.setCheckState(Qt.Unchecked)
        self.chb_show_val.toggled.connect(self.show_val_change)

        self.opt1Label = QLabel(self.centralwidget)
        self.opt1Label.setObjectName(u"opt1Label")
        self.opt1Label.setGeometry(QRect(10, 45, 75, 25))
        self.opt1Label.setText("Optimizer 1:")

        self.opt_dd1 = QComboBox(self.centralwidget)
        self.opt_dd1.setObjectName("opt_dd")
        self.opt_dd1.setGeometry(QRect(90, 45, 85, 25))
        self.opt_dd1.addItems(["None", "Adam", "AmsGrad"])
        self.opt_dd1.setEditable(False)
        self.opt_dd1.setCurrentIndex(0)
        self.opt_dd1.currentIndexChanged.connect(self.opt_change1)

        self.opt2Label = QLabel(self.centralwidget)
        self.opt2Label.setObjectName(u"opt2Label")
        self.opt2Label.setGeometry(QRect(185, 45, 75, 25))
        self.opt2Label.setText("Optimizer 2:")

        self.opt_dd2 = QComboBox(self.centralwidget)
        self.opt_dd2.setObjectName("opt_dd")
        self.opt_dd2.setGeometry(QRect(265, 45, 85, 25))
        self.opt_dd2.addItems(["None", "Adam", "AmsGrad"])
        self.opt_dd2.setEditable(False)
        self.opt_dd2.setCurrentIndex(2)
        self.opt_dd2.currentIndexChanged.connect(self.opt_change2)

        self.val1 = QLabel(self.centralwidget)
        self.val1.setObjectName(u"val1")
        self.val1.setGeometry(QRect(120, 410, 120, 25))

        self.val2 = QLabel(self.centralwidget)
        self.val2.setObjectName(u"val2")
        self.val2.setGeometry(QRect(520, 410, 120, 25))

        self.create_graphs()
        self.show_val_change()

        main_window.setCentralWidget(self.centralwidget)

        self.retranslate_ui(main_window)

        QMetaObject.connectSlotsByName(main_window)

    # setupUi

    def retranslate_ui(self, main_window):
        main_window.setWindowTitle("Backpropagation with optimizers")
        self.alfaLineEdit.setToolTip("Learning rate Alfa")
        self.stepsLineEdit.setToolTip("Number of iterations")
        self.stepsLineEdit.setText("10")
        self.alfaLineEdit.setText("0.001")
        self.startButton.setText("All steps")
        self.stepButton.setText("1 step")
        self.btn_reset.setText("Reset")
        self.chb_side_by_side.setText("Side by side")
        self.chb_show_val.setText("Show test loss")
        self.val1.setText("Test loss opt 1: XXX")
        self.val2.setText("Test loss opt 2: XXX")
