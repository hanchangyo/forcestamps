# -*- coding: utf-8 -*-

# Python 3 compatibility
from __future__ import print_function
try:
    input = raw_input
except NameError:
    pass

import sys
import numpy as np
import cv2
from pythonosc import osc_message_builder
from pythonosc import udp_client
import argparse
from copy import deepcopy

# from pyqtgraph.Qt import QtCore, QtGui
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from pyqtgraph.ptime import time

import sensel_control as sc
import forcestamp

from forcestamp_ui import Ui_MainWindow


class MarkerPopupWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(MarkerPopupWidget, self).__init__(parent)
        self.parent = parent

        # make the window frameless
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        self.fillColor = QtGui.QColor(30, 30, 30, 120)
        self.penColor = QtGui.QColor("#333333")

        self.popup_fillColor = QtGui.QColor(240, 240, 240, 120)
        self.popup_penColor = QtGui.QColor(200, 200, 200, 255)

        self.marker_fillColor = QtGui.QColor(240, 240, 240, 120)

        self.rotation_fillColor = QtGui.QColor(240, 10, 10)

        self.vector_fillColor = QtGui.QColor(255, 240, 40)

        self.radius = 25 * 5
        # print(self.parent.markers)

    def makeTriangle(self, centerx, centery, r, theta, dr=15, phi=0.1):
        # p1: outer point
        # p2, p2: inner points
        triangle = QtGui.QPolygonF()
        # dr = 15
        # phi = 0.1

        p1x = centerx + r * np.cos(theta)
        p1y = centery + r * np.sin(theta)
        p1 = QtCore.QPointF(p1x, p1y)

        p2x = centerx + (r - dr) * np.cos(theta - phi)
        p2y = centery + (r - dr) * np.sin(theta - phi)
        p3x = centerx + (r - dr) * np.cos(theta + phi)
        p3y = centery + (r - dr) * np.sin(theta + phi)

        p2 = QtCore.QPointF(p2x, p2y)
        p3 = QtCore.QPointF(p3x, p3y)

        triangle.append(p1)
        triangle.append(p2)
        triangle.append(p3)

        return triangle

    def paintEvent(self, event):
        for mkr in self.parent.markers:
            # # get current window size
            # s = self.size()

            # make painter object
            qp = QtGui.QPainter()

            # start paint event
            qp.begin(self)

            # # Draw background
            # qp.setRenderHint(QtGui.QPainter.Antialiasing, True)
            # qp.setPen(self.penColor)
            # qp.setBrush(self.fillColor)
            # qp.drawRect(0, 0, s.width(), s.height())

            # Draw marker
            qp.setPen(QtCore.Qt.NoPen)
            # marker_fillColor = QtGui.QColor(240, 240, 240, forcestamp.constraint(int(mkr.force / 50), 0, 255))
            marker_fillColor = QtGui.QColor(240, 240, 240, 150)
            qp.setBrush(marker_fillColor)
            # radius = 25 * 5
            centerx = mkr.posX * 5
            centery = (104 - mkr.posY) * 5
            center = QtCore.QPoint(centerx, centery)
            qp.drawEllipse(center, self.radius, self.radius)
            qp.drawEllipse(center, forcestamp.constraint(self.radius * mkr.force / 10000, 0, self.radius),
                           forcestamp.constraint(self.radius * mkr.force / 10000, 0, self.radius)
                           )

            # Draw rotation
            qp.setBrush(self.rotation_fillColor)
            qp.drawPolygon(self.makeTriangle(centerx, centery, self.radius, mkr.rot))

            # Draw force vector
            # qp.setPen(self.vector_fillColor)
            qp.setPen(QtGui.QPen(self.vector_fillColor, 7, QtCore.Qt.SolidLine))
            qp.setBrush(self.vector_fillColor)
            # vec_len = mkr.force / 200
            vecX = mkr.vecX / 1000
            vecY = mkr.vecY / 1000
            qp.drawLine(centerx, centery, centerx + vecX, centery - vecY)
            qp.setPen(QtCore.Qt.NoPen)
            qp.drawEllipse(center, 7, 7)
            # qp.drawPolygon(self.makeTriangle(centerx, centery, vec_len, -mkr.vecY / mkr.vecX, 15, 0.3))

            font = QtGui.QFont()
            font.setPixelSize(36)
            font.setBold(False)
            qp.setFont(font)
            qp.setPen(QtGui.QColor(0, 0, 0))
            text = 'ID: ' + str(mkr.ID)
            qp.drawText(centerx - 35, centery - 20, text)

            qp.end()


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


class MarkerPopupSignals(QtCore.QObject):
    # SIGNALS
    OPEN = QtCore.pyqtSignal()
    CLOSE = QtCore.pyqtSignal()


class ForceStamp (QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(ForceStamp, self).__init__(parent)
        self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.CustomizeWindowHint | QtCore.Qt.WindowMinimizeButtonHint | QtCore.Qt.WindowMaximizeButtonHint | QtCore.Qt.WindowCloseButtonHint)
        self.ui = Ui_MainWindow()  # import GUI layout
        self.ui.setupUi(self)  # initalization

        # param names
        self.param_names = ['posx_max', 'posx_min', 'posy_max', 'posy_min', 'force_max',
                            'force_min', 'vecx_max', 'vecx_min', 'vecy_max', 'vecy_min']

        # Morph size
        self.rows = 185
        self.cols = 105
        self.markerRadius = 20

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

        # # Open Morph
        self.handle, self.info = sc.open_sensel()
        # Initalize frame
        self.frame = sc.init_frame(self.handle)

        # Setup OSC server
        self.setupOSC()

        # update interval
        self.interval = 0  # miliseconds

        self.lastTime = time()
        self.fps = None

        # update using timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updateData)
        self.timer.start(self.interval)

        # marker information popup
        self._popframe = None
        self._popflag = False
        self.SIGNALS = MarkerPopupSignals()
        self.SIGNALS.OPEN.connect(self.onPopup)
        self.SIGNALS.CLOSE.connect(self.closePopup)

    def updateData(self):
        # scan image from the device
        try:
            self.f_image = sc.scan_frames(self.handle, self.frame, self.info)
        except(UnboundLocalError):
            sc.close_sensel(self.handle, self.frame)
            # Open Morph
            self.handle, self.info = sc.open_sensel()
            # Initalize frame
            self.frame = sc.init_frame(self.handle)
            self.f_image = sc.scan_frames(self.handle, self.frame, self.info)

        # find local peaks
        self.f_image_peaks = forcestamp.findLocalPeaks(self.f_image)

        # find marker objects
        markerCenters, markerCenters_raw = forcestamp.findMarker(self.f_image_peaks, self.markerRadius)
        # forcestamp.findMarker(self.f_image_peaks, self.markerRadius)

        # retrieve marker parameters from marker coordinates
        self.markers = []
        for i in range(len(markerCenters)):
            # print(cnt)
            self.markers.append(forcestamp.marker(self.f_image, self.f_image_peaks, self.markerRadius, markerCenters[i]))
            # print('markerX: ' + str(self.markers[0].posX))
            # print('markerY: ' + str(self.markers[0].posY))
            # print('markerCode: ' + str(self.markers[0].code))
            self.markers[i].force = self.markers[i].sumForce()
            (self.markers[i].vecX, self.markers[i].vecY) = self.markers[i].vectorForce()
            self.markers[i].rot = self.markers[i].calculateAbsoluteRotation()

        if len(self.markers) > 0:
            # print('markerID: ' + str(self.markers[0].ID))
            # print('markerForce: ' + str(self.markers[0].sumForce()))
            # print('markerVectorForce: ' + str(self.markers[0].vectorForce()))
            if not self._popflag:
                self.SIGNALS.OPEN.emit()

        else:
            if self._popflag:
                self.SIGNALS.CLOSE.emit()

        # send marker parameters to GUI
        self.sendMarkerParameters()

        # retrieve peak coordinates from the peak image
        self.peaks = forcestamp.findPeakCoord(self.f_image_peaks)

        # prepare a image copy for display
        f_image_show = deepcopy(self.f_image)
        if np.max(f_image_show) > 0:
            f_image_show = f_image_show / np.max(f_image_show) * 255
        f_image_show = cv2.cvtColor(f_image_show.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        # draw peaks
        for cnt in self.peaks:
            cv2.circle(
                f_image_show,
                (np.int(cnt[::-1][0]), np.int(cnt[::-1][1])),
                0,
                (0, 255, 255)
            )

        # set image for display
        self.img.setImage(np.rot90(f_image_show, 3), autoLevels=True, levels=(0, 50))

        # self.calculateFPS()
        QtGui.QApplication.processEvents()

    def resizeEvent(self, event):
        if self._popflag:
            self._popframe.move(0, 0)
            self._popframe.resize(self.width(), self.height())

    def onPopup(self):
        self._popframe = MarkerPopupWidget(self)
        self._popframe.move(10, 10)
        # print(self.ui.graphicsView.width(), self.ui.graphicsView.height())
        self._popframe.resize(self.ui.graphicsView.width(), self.ui.graphicsView.height())
        self._popflag = True
        self._popframe.show()

    def closePopup(self):
        self._popframe.close()
        self._popflag = False

    def sendMarkerParameters(self):
        posx_min = 30
        posx_max = 184 - 30
        posy_min = 30
        posy_max = 104 - 30
        force_min = 0
        force_max = 10000
        vecx_min = -30000
        vecx_max = 30000
        vecy_min = -30000
        vecy_max = 30000

        for mkr in self.markers:
            mkr.posY = 104 - mkr.posY
            # mkr.force = mkr.sumForce()
            # (mkr.vecX, mkr.vecY) = mkr.vectorForce()
            mkr.vecY *= -1
            mkr.posX_scaled = (self.IDparam[mkr.ID].posx_max - self.IDparam[mkr.ID].posx_min) * ((forcestamp.constraint(mkr.posX, posx_min, posx_max) - posx_min) / (posx_max - posx_min)) + self.IDparam[mkr.ID].posx_min
            mkr.posY_scaled = (self.IDparam[mkr.ID].posy_max - self.IDparam[mkr.ID].posy_min) * ((forcestamp.constraint(mkr.posY, posy_min, posy_max) - posy_min) / (posy_max - posy_min)) + self.IDparam[mkr.ID].posy_min
            mkr.force_scaled = (self.IDparam[mkr.ID].force_max - self.IDparam[mkr.ID].force_min) * ((forcestamp.constraint(mkr.force, force_min, force_max) - force_min) / (force_max - force_min)) + self.IDparam[mkr.ID].force_min
            mkr.vecX_scaled = (self.IDparam[mkr.ID].vecx_max - self.IDparam[mkr.ID].vecx_min) * ((forcestamp.constraint(mkr.vecX, vecx_min, vecx_max) - vecx_min) / (vecx_max - vecx_min)) + self.IDparam[mkr.ID].vecx_min
            mkr.vecY_scaled = (self.IDparam[mkr.ID].vecy_max - self.IDparam[mkr.ID].vecy_min) * ((forcestamp.constraint(mkr.vecY, vecy_min, vecy_max) - vecy_min) / (vecy_max - vecy_min)) + self.IDparam[mkr.ID].vecy_min

            # send osc data
            self.sendOSC(mkr.posX_scaled, '/m' + str(mkr.ID) + '/posx')
            self.sendOSC(mkr.posY_scaled, '/m' + str(mkr.ID) + '/posy')
            self.sendOSC(mkr.force_scaled, '/m' + str(mkr.ID) + '/force')
            self.sendOSC(mkr.vecX_scaled, '/m' + str(mkr.ID) + '/vecx')
            self.sendOSC(mkr.vecY_scaled, '/m' + str(mkr.ID) + '/vecy')
            self.sendOSC(np.rad2deg(mkr.rot), '/m' + str(mkr.ID) + '/rot')
            self.sendOSC(mkr.ID, '/m' + str(mkr.ID) + '/id')

            if self.currentID == mkr.ID and mkr.ID is not 0:
                # send current parameters
                self.ui.progressBar_posx.setValue(mkr.posX_scaled * 10)
                self.ui.progressBar_posy.setValue(mkr.posY_scaled * 10)
                self.ui.progressBar_force.setValue(mkr.force_scaled * 10)
                self.ui.progressBar_vecx.setValue(mkr.vecX_scaled * 10)
                self.ui.progressBar_vecy.setValue(mkr.vecY_scaled * 10)
                self.ui.value_posx.setText(('%.01f' % mkr.posX_scaled))
                self.ui.value_posy.setText(('%.01f' % mkr.posY_scaled))
                self.ui.value_force.setText(('%.01f' % mkr.force_scaled))
                self.ui.value_vecx.setText(('%.01f' % mkr.vecX_scaled))
                self.ui.value_vecy.setText(('%.01f' % mkr.vecY_scaled))

    def calculateFPS(self):
        now = time()
        dt = now - self.lastTime
        self.lastTime = now

        if self.fps is None:
            self.fps = 1.0 / dt
        else:
            s = np.clip(dt * 3., 0, 1)
            self.fps = self.fps * (1 - s) + (1.0 / dt) * s

        print('%0.2f fps' % self.fps)

    def setupOSC(self):
        # setup OSC server
        ip = '127.0.0.1'
        port_num = 8002
        parser = argparse.ArgumentParser()
        parser.add_argument("--ip", default=ip, help="The ip of th OSC Server")
        parser.add_argument("--port", type=int, default=port_num, help="The port the OSC server is listening on")
        args = parser.parse_args()
        self.client = udp_client.UDPClient(args.ip, args.port)
        print('OSC server on: ' + ip + ':' + str(port_num))

    def sendOSC(self, msg, address):
        msgStruct = osc_message_builder.OscMessageBuilder(address=address)
        msgStruct.add_arg(msg)
        msgStruct = msgStruct.build()
        self.client.send(msgStruct)

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
        self.viewBox.setAspectLocked(True)
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
        # self.ui.comboBox.activated[str].connect(self.onPopup)

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
        # print(self.currentID)
        # print(self.IDparam[self.currentID].printParameters())

        # load ID parameters and apply to current controls
        for name in self.param_names:
            evaltext = 'self.ui.doubleSpinBox_' + name + '.setValue(self.IDparam[self.currentID].' + name + ')'
            eval(evaltext)

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
            eval('self.ui.doubleSpinBox_' + name + '.valueChanged.connect(self.onSpinBoxChanged)')

    def onSpinBoxChanged(self, value):
        # sender is the latest signal
        sender = self.sender()
        name = sender.objectName()

        # save current parameter
        exectext = 'self.IDparam[self.currentID].' + name[14:] + ' = self.ui.' + name + '.value()'
        exec(exectext)

        self.updateProgressBarRange()

    def updateProgressBarRange(self):
        # Change corresponding progress bar range
        if self.ui.doubleSpinBox_posx_max.value() >= self.ui.doubleSpinBox_posx_min.value():
            self.ui.progressBar_posx.setRange(10 * self.ui.doubleSpinBox_posx_min.value(), 10 * self.ui.doubleSpinBox_posx_max.value())
        else:
            self.ui.progressBar_posx.setRange(10 * self.ui.doubleSpinBox_posx_max.value(), 10 * self.ui.doubleSpinBox_posx_min.value())

        if self.ui.doubleSpinBox_posy_max.value() >= self.ui.doubleSpinBox_posy_min.value():
            self.ui.progressBar_posy.setRange(10 * self.ui.doubleSpinBox_posy_min.value(), 10 * self.ui.doubleSpinBox_posy_max.value())
        else:
            self.ui.progressBar_posy.setRange(10 * self.ui.doubleSpinBox_posy_max.value(), 10 * self.ui.doubleSpinBox_posy_min.value())

        if self.ui.doubleSpinBox_force_max.value() >= self.ui.doubleSpinBox_force_min.value():
            self.ui.progressBar_force.setRange(10 * self.ui.doubleSpinBox_force_min.value(), 10 * self.ui.doubleSpinBox_force_max.value())
        else:
            self.ui.progressBar_force.setRange(10 * self.ui.doubleSpinBox_force_max.value(), 10 * self.ui.doubleSpinBox_force_min.value())

        if self.ui.doubleSpinBox_vecx_max.value() >= self.ui.doubleSpinBox_vecx_min.value():
            self.ui.progressBar_vecx.setRange(10 * self.ui.doubleSpinBox_vecx_min.value(), 10 * self.ui.doubleSpinBox_vecx_max.value())
        else:
            self.ui.progressBar_vecx.setRange(10 * self.ui.doubleSpinBox_vecx_max.value(), 10 * self.ui.doubleSpinBox_vecx_min.value())

        if self.ui.doubleSpinBox_vecy_max.value() >= self.ui.doubleSpinBox_vecy_min.value():
            self.ui.progressBar_vecy.setRange(10 * self.ui.doubleSpinBox_vecy_min.value(), 10 * self.ui.doubleSpinBox_vecy_max.value())
        else:
            self.ui.progressBar_vecy.setRange(10 * self.ui.doubleSpinBox_vecy_max.value(), 10 * self.ui.doubleSpinBox_vecy_min.value())

    def initProgressBar(self):
        # Change corresponding progress bar range
        self.ui.progressBar_posx.setRange(10 * self.ui.doubleSpinBox_posx_min.value(), 10 * self.ui.doubleSpinBox_posx_max.value())
        self.ui.progressBar_posy.setRange(10 * self.ui.doubleSpinBox_posy_min.value(), 10 * self.ui.doubleSpinBox_posy_max.value())
        self.ui.progressBar_force.setRange(10 * self.ui.doubleSpinBox_force_min.value(), 10 * self.ui.doubleSpinBox_force_max.value())
        self.ui.progressBar_vecx.setRange(10 * self.ui.doubleSpinBox_vecx_min.value(), 10 * self.ui.doubleSpinBox_vecx_max.value())
        self.ui.progressBar_vecy.setRange(10 * self.ui.doubleSpinBox_vecy_min.value(), 10 * self.ui.doubleSpinBox_vecy_max.value())

    def closeEvent(self, event):
        print('Quitting application')
        sc.close_sensel(self.handle, self.frame)
        sys.exit()


if __name__ == '__main__':
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        app = QtGui.QApplication(sys.argv)
        window = ForceStamp()
        window.show()
        sys.exit(app.exec_())
