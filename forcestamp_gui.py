# -*- coding: utf-8 -*-

# Python 3 compatibility
from __future__ import print_function
try:
    input = raw_input
except NameError:
    pass

import sys
import numpy as np

from pyqtgraph.Qt import QtCore, QtGui
from PyQt5 import QtWidgets
import pyqtgraph as pg
# from pyqtgraph.ptime import time

from forcestamp_ui import Ui_MainWindow


class IDparameter:
    def __init__(self):
        self.posx_max = 100.0
        self.posx_min = 0.0
        self.posy_max = 100.0
        self.posy_min = 0.0
        self.force_max = 100.0
        self.force_min = 0.0
        self.vecx_max = 100.0
        self.vecx_min = -100.0
        self.vecy_max = 100.0
        self.vecy_min = -100.0

    def printParameters(self):
        print(self.posx_max, self.posx_min, self.posy_max, self.posy_min, self.force_max, self.force_min, self.vecx_max, self.vecx_min, self.vecy_max, self.vecy_min)


class ForceStamp (QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(ForceStamp, self).__init__(parent)
        self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.CustomizeWindowHint | QtCore.Qt.WindowMinimizeButtonHint | QtCore.Qt.WindowMaximizeButtonHint | QtCore.Qt.WindowCloseButtonHint)
        self.ui = Ui_MainWindow()  # import GUI layout
        self.ui.setupUi(self)  # initalization

        # param names
        self.param_names = ['posx_max', 'posx_min', 'posy_max', 'posy_min', 'force_max',
                            'force_min', 'vecx_max', 'vecx_min', 'vecy_max', 'vecy_min']

        # Initialize combobox items
        self.initComboBox()

        # initialize spin boxes
        self.initSpinBox()

        # initialize progress bars
        self.initProgressBar()

        # initailize ID scale parameters
        self.IDparam = [IDparameter() for count in range(12)]

        # initialize force image
        self.initViewBox()
        # TODO
        # draw vector representation of markers
        # display tooltip on marker position
        # make the progress bar more fancy

    def initViewBox(self):
        # Create random image
        self.img = pg.ImageItem()
        view = self.ui.graphicsView
        view.ci.layout.setContentsMargins(0, 0, 0, 0)
        view.ci.layout.setSpacing(0)
        self.viewBox = view.addViewBox()

        # self.ui.graphicsView.scale(50, 50)
        # view.setCentralWidget(viewBox)
        self.viewBox.addItem(self.img)
        self.viewBox.setRange(QtCore.QRectF(0, 0, 185 * 1, 105 * 1), padding=0)
        self.viewBox.setMouseEnabled(x=False, y=False)
        self.viewBox.setMenuEnabled(False)

        data = np.random.normal(size=(185 * 1, 105 * 1), loc=1024, scale=64).astype(np.uint16)
        self.img.setImage(data)

    def initComboBox(self):
        # insert IDs
        self.ui.comboBox.addItem('ID')
        self.ui.comboBox.addItem('1')
        self.ui.comboBox.addItem('2')
        self.ui.comboBox.addItem('3')
        self.ui.comboBox.addItem('4')
        self.ui.comboBox.addItem('5')
        self.ui.comboBox.addItem('6')
        self.ui.comboBox.addItem('7')
        self.ui.comboBox.addItem('8')
        self.ui.comboBox.addItem('9')
        self.ui.comboBox.addItem('10')
        self.ui.comboBox.addItem('11')
        self.currentID = 0

        # event on value change
        self.ui.comboBox.activated[str].connect(self.onComboBoxActivated)

    def onComboBoxActivated(self, text):
        # save current parameters
        self.IDparam[self.currentID].posx_max = self.ui.doubleSpinBox_posx_max.value()
        self.IDparam[self.currentID].posx_min = self.ui.doubleSpinBox_posx_min.value()
        self.IDparam[self.currentID].posy_max = self.ui.doubleSpinBox_posy_max.value()
        self.IDparam[self.currentID].posy_min = self.ui.doubleSpinBox_posy_min.value()
        self.IDparam[self.currentID].force_max = self.ui.doubleSpinBox_force_max.value()
        self.IDparam[self.currentID].force_min = self.ui.doubleSpinBox_force_min.value()
        self.IDparam[self.currentID].vecx_max = self.ui.doubleSpinBox_vecx_max.value()
        self.IDparam[self.currentID].vecx_min = self.ui.doubleSpinBox_vecx_min.value()
        self.IDparam[self.currentID].vecy_max = self.ui.doubleSpinBox_vecy_max.value()
        self.IDparam[self.currentID].vecy_min = self.ui.doubleSpinBox_vecy_min.value()

        # change current marker ID
        if text == 'ID':
            self.currentID = 0
        else:
            self.currentID = int(text)
        print(self.currentID)
        # print(self.IDparam[self.currentID].printParameters())

        # load ID parameters and apply to current controls
        for name in self.param_names:
            evaltext = 'self.ui.doubleSpinBox_' + name + '.setValue(self.IDparam[self.currentID].' + name + ')'
            eval(evaltext)

        self.ui.progressBar_posx.setRange(self.IDparam[self.currentID].posx_min, self.IDparam[self.currentID].posx_max)
        self.ui.progressBar_posy.setRange(self.IDparam[self.currentID].posy_min, self.IDparam[self.currentID].posy_max)
        self.ui.progressBar_force.setRange(self.IDparam[self.currentID].force_min, self.IDparam[self.currentID].force_max)
        self.ui.progressBar_vecx.setRange(self.IDparam[self.currentID].vecx_min, self.IDparam[self.currentID].vecx_max)
        self.ui.progressBar_vecy.setRange(self.IDparam[self.currentID].vecy_min, self.IDparam[self.currentID].vecy_max)

    def initSpinBox(self):

        for name in self.param_names:
            eval('self.ui.doubleSpinBox_' + name + '.setRange(-1000, 1000)')
            if 'max' in name:  # if there's 'max' string
                eval('self.ui.doubleSpinBox_' + name + '.setValue(100)')
            else:
                if 'vec' in name:  # if the box is vector
                    eval('self.ui.doubleSpinBox_' + name + '.setValue(-100)')
                else:
                    eval('self.ui.doubleSpinBox_' + name + '.setValue(0)')
            eval('self.ui.doubleSpinBox_' + name + '.setDecimals(1)')
            eval('self.ui.doubleSpinBox_' + name + '.setKeyboardTracking(False)')
            eval('self.ui.doubleSpinBox_' + name + '.valueChanged.connect(self.changeProgressBarRange)')

    def changeProgressBarRange(self, value):
        # sender is the latest signal
        sender = self.sender()
        name = sender.objectName()

        # Constrain the spinbox values
        if 'max' in name:  # if the spinbox is named 'max'
            # compare with min value
            compare = eval('self.ui.' + name[:-3] + 'min.value()')
            if compare > value:
                sender.setValue(compare)
        else:
            # compare with max value
            compare = eval('self.ui.' + name[:-3] + 'max.value()')
            if compare < value:
                sender.setValue(compare)

        # Change corresponding progress bar range
        self.ui.progressBar_posx.setRange(self.ui.doubleSpinBox_posx_min.value(), self.ui.doubleSpinBox_posx_max.value())
        self.ui.progressBar_posy.setRange(self.ui.doubleSpinBox_posy_min.value(), self.ui.doubleSpinBox_posy_max.value())
        self.ui.progressBar_force.setRange(self.ui.doubleSpinBox_force_min.value(), self.ui.doubleSpinBox_force_max.value())
        self.ui.progressBar_vecx.setRange(self.ui.doubleSpinBox_vecx_min.value(), self.ui.doubleSpinBox_vecx_max.value())
        self.ui.progressBar_vecy.setRange(self.ui.doubleSpinBox_vecy_min.value(), self.ui.doubleSpinBox_vecy_max.value())

    def initProgressBar(self):
        # Change corresponding progress bar range
        self.ui.progressBar_posx.setRange(self.ui.doubleSpinBox_posx_min.value(), self.ui.doubleSpinBox_posx_max.value())
        self.ui.progressBar_posy.setRange(self.ui.doubleSpinBox_posy_min.value(), self.ui.doubleSpinBox_posy_max.value())
        self.ui.progressBar_force.setRange(self.ui.doubleSpinBox_force_min.value(), self.ui.doubleSpinBox_force_max.value())
        self.ui.progressBar_vecx.setRange(self.ui.doubleSpinBox_vecx_min.value(), self.ui.doubleSpinBox_vecx_max.value())
        self.ui.progressBar_vecy.setRange(self.ui.doubleSpinBox_vecy_min.value(), self.ui.doubleSpinBox_vecy_max.value())


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = ForceStamp()
    window.show()
    app.exec_()
    print('Closed the window')
    sys.exit()
