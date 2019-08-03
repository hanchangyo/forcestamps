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
from oscpy.client import OSCClient

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

        self.cof_fillColor = QtGui.QColor(255, 240, 40)

        # self.radius = 25 * 5
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
        for mkr in self.parent.MarkerTracker.markers:
            if mkr.ID is not 0:
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
                centerx = mkr.pos_x * 5
                centery = mkr.pos_y * 5
                radius = mkr.radius * 1.25 * 5
                center = QtCore.QPoint(centerx, centery)
                qp.drawEllipse(center, radius, radius)
                qp.drawEllipse(center, forcestamp.constraint(radius * mkr.force / self.parent.force_sensitivity, 0, radius),
                               forcestamp.constraint(radius * mkr.force / self.parent.force_sensitivity, 0, radius)
                               )

                # Draw rotation
                qp.setBrush(self.rotation_fillColor)
                qp.drawPolygon(self.makeTriangle(centerx, centery, radius, -mkr.rot + np.pi / 2))

                # Draw force vector
                # qp.setPen(self.cof_fillColor)
                qp.setPen(QtGui.QPen(self.cof_fillColor, 7, QtCore.Qt.SolidLine))
                qp.setBrush(self.cof_fillColor)
                # vec_len = mkr.force / 200
                cof_x = mkr.cof_x / self.parent.cof_sensitivity * 20
                cof_y = mkr.cof_y / self.parent.cof_sensitivity * 20
                # print(cof_y)
                qp.drawLine(centerx, centery, centerx + cof_x, centery - cof_y)
                qp.setPen(QtCore.Qt.NoPen)
                vecCenter = QtCore.QPoint(centerx + cof_x, centery - cof_y)
                qp.drawEllipse(vecCenter, 7, 7)
                # qp.drawPolygon(self.makeTriangle(centerx, centery, vec_len, -mkr.cof_y / mkr.cof_x, 15, 0.3))

                font = QtGui.QFont()
                font.setPixelSize(36)
                font.setBold(False)
                qp.setFont(font)
                qp.setPen(QtGui.QColor(0, 0, 0))
                # text = 'ID: ' + str(mkr.ID)
                text = str(mkr.ID)
                # qp.drawText(centerx - 35, centery - 20, text)
                qp.drawText(centerx - 7, centery - 20, text)

                qp.end()


class IDparameter:
    def __init__(self):
        self.posx_max = 100.0
        self.posx_min = 0.0
        self.posy_max = 100.0
        self.posy_min = 0.0
        self.force_max = 100.0
        self.force_min = 0.0
        self.cof_x_max = 100.0
        self.cof_x_min = -100.0
        self.cof_y_max = 100.0
        self.cof_y_min = -100.0

    def printParameters(self):
        print(self.posx_max, self.posx_min, self.posy_max, self.posy_min, self.force_max, self.force_min, self.cof_x_max, self.cof_x_min, self.cof_y_max, self.cof_y_min)


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
                            'force_min', 'cof_x_max', 'cof_x_min', 'cof_y_max', 'cof_y_min']

        # Morph size
        self.rows = 185
        self.cols = 105
        # self.radius = [12.8, 20]
        self.marker_radii = [55 / 2 / 1.25, 17 / 1.25, 20.0, 16 / 1.25]
        self.num_ID = 102

        # Initialize combobox items
        self.initComboBox()

        # initialize spin boxes
        self.initSpinBox()

        # initialize progress bars
        self.initProgressBar()

        # initailize ID scale parameters
        self.IDs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 51, 80, 101]
        self.IDparam = [IDparameter() for count in range(self.num_ID)]

        # initialize force image
        self.initViewBox()

        self.ui.pushButton.clicked.connect(self.onStartButton)
        self._buttonFlag = False
        # Open Morph
        # try:
        # self.handle, self.info = sc.open_sensel()
        # except OSError:
        #     sc.close_sensel(self.handle, self.frame)
        #     # Open Morph
        #     self.handle, self.info = sc.open_sensel()

        # # Initalize frame
        # self.frame = sc.init_frame(self.handle)

        # initalize marker storage
        self.markers = []

        # set marker sensitivity
        self.force_sensitivity = 3000
        self.cof_sensitivity = 10

        # store previous state
        self.prevState = [0] * self.num_ID

        # Setup OSC server
        # self.setupOSC()
        address = '127.0.0.1'
        # address = '192.168.0.3'
        port = 12000
        self.osc = OSCClient(address, port)


        # update interval
        self.interval = 0  # miliseconds

        self.lastTime = time()
        self.fps = None

        # update using timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updateData)
        # self.timer.start(self.interval)

        # marker information popup
        self._popframe = None
        self._popflag = False
        self.SIGNALS = MarkerPopupSignals()
        self.SIGNALS.OPEN.connect(self.onPopup)
        self.SIGNALS.CLOSE.connect(self.closePopup)

        self.BlobTracker = forcestamp.TrackBlobs()
        self.MarkerTracker = forcestamp.TrackMarkers(radii=self.marker_radii)

    def updateData(self):
        # scan image from the device
        try:
            self.f_image = sc.scan_frames(self.handle, self.frame, self.info)
        except(UnboundLocalError):
            try:
                sc.close_sensel(self.handle, self.frame)
                # Open Morph
                self.handle, self.i = sc.open_sensel()
                # Initalize frame
                self.frame = sc.init_frame(self.handle, baseline=0)
                self.f_image = sc.scan_frames(self.handle, self.frame, self.info)
            except(UnboundLocalError):
                self.f_image = np.zeros((self.rows, self.cols))

        # update blob information
        self.blobs = self.BlobTracker.update(self.f_image)

        # update marker information
        self.MarkerTracker.update(self.f_image, self.blobs)
        if len(self.MarkerTracker.markers) > 0:
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

        '''
        # retrieve peak coordinates from the peak image
        self.peaks = forcestamp.findPeakCoord(self.f_image_peaks)

        # exclude marker areas from the peak coords
        self.f_image_peaks_excluded = deepcopy(self.f_image_peaks)
        for mkr in self.markers:
            self.f_image_peaks_excluded = forcestamp.excludeMarkerPeaks(self.f_image_peaks_excluded, (mkr.pos_y, mkr.pos_x), mkr.radius)
        self.peaks_excluded = forcestamp.findPeakCoord(self.f_image_peaks_excluded)
        self.peaks_excluded = forcestamp.findSubpixelPeaks(self.peaks_excluded, self.f_image)
        self.peaks_force = []
        for pk in self.peaks_excluded:
            self.peaks_force.append(np.sum(forcestamp.cropImage(self.f_image, pk, radius=3, margin=1)))
        # print(self.peaks_excluded)
        # print(self.peaks_force)
        # print(self.peaks_excluded)
        if len(self.peaks_excluded) > 0:
            self.sendOSC_coords(self.peaks_excluded, self.peaks_force)
        '''

        # prepare a image copy for display
        f_image_show = deepcopy(self.f_image)
        if np.max(f_image_show) > 0:
            f_image_show = f_image_show / np.max(f_image_show) * 255
        f_image_show = cv2.cvtColor(f_image_show.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        # draw peaks
        for b in self.blobs:
            cv2.circle(
                f_image_show,
                (np.int(b.cx), np.int(b.cy)),
                0,
                (0, 255, 255)
            )

        # set image for display
        self.img.setImage(np.rot90(f_image_show, 3), autoLevels=True, levels=(0, 50))

        # self.calculateFPS()
        QtGui.QApplication.processEvents()

    def onStartButton(self):
        if self._buttonFlag:
            self._buttonFlag = True
            # self.ui.pushButton.setText('Start')
        else:
            self._buttonFlag = True
            # self.ui.pushButton.setText('Stop')
            try:
                self.handle, self.info = sc.open_sensel()
            except OSError:
                sc.close_sensel(self.handle, self.frame)
                # Open Morph
                self.handle, self.info = sc.open_sensel()
            # Initalize frame
            self.frame = sc.init_frame(self.handle, baseline=0)

            # update using timer
            # self.timer = QtCore.QTimer()
            # self.timer.timeout.connect(self.updateData)
            self.timer.start(self.interval)

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
        posx_max = self.rows - 30
        posy_min = 30
        posy_max = self.cols - 30
        force_min = 0
        # force_max = 8000
        # cof_x_min = -30000
        # cof_x_max = 30000
        # cof_y_min = -30000
        # cof_y_max = 30000
        cof_x_min = cof_y_min = -self.cof_sensitivity
        cof_x_max = cof_y_max = self.cof_sensitivity

        currentState = [0] * self.num_ID

        # self.sendOSC(len(self.MarkerTracker.markers), '/num')
        self.osc.send_message(b'/num', [len(self.MarkerTracker.markers)])
        # self.sendOSC([0.0, 0.0, 0.1], 'pos_x')
        pos_x = []
        pos_y = []
        force = []
        cof_x = []
        cof_y = []
        angle = []
        id_list = []
        radius = []

        for mkr in self.MarkerTracker.markers:
            # mkr.pos_y = 104 - mkr.pos_y
            # mkr.force = mkr.sumForce()
            # (mkr.cof_x, mkr.cof_y) = mkr.vectorForce()
            # mkr.cof_y *= -1
            mkr.pos_x_scaled = (self.IDparam[mkr.ID].posx_max - self.IDparam[mkr.ID].posx_min) * ((forcestamp.constraint(mkr.pos_x, posx_min, posx_max) - posx_min) / (posx_max - posx_min)) + self.IDparam[mkr.ID].posx_min
            mkr.pos_y_scaled = (self.IDparam[mkr.ID].posy_max - self.IDparam[mkr.ID].posy_min) * ((forcestamp.constraint(self.cols - mkr.pos_y, posy_min, posy_max) - posy_min) / (posy_max - posy_min)) + self.IDparam[mkr.ID].posy_min
            mkr.force_scaled = (self.IDparam[mkr.ID].force_max - self.IDparam[mkr.ID].force_min) * ((forcestamp.constraint(mkr.force, force_min, self.force_sensitivity) - force_min) / (self.force_sensitivity - force_min)) + self.IDparam[mkr.ID].force_min
            mkr.cof_x_scaled = (self.IDparam[mkr.ID].cof_x_max - self.IDparam[mkr.ID].cof_x_min) * ((forcestamp.constraint(mkr.cof_x, cof_x_min, cof_x_max) - cof_x_min) / (cof_x_max - cof_x_min)) + self.IDparam[mkr.ID].cof_x_min
            mkr.cof_y_scaled = (self.IDparam[mkr.ID].cof_y_max - self.IDparam[mkr.ID].cof_y_min) * ((forcestamp.constraint(mkr.cof_y, cof_y_min, cof_y_max) - cof_y_min) / (cof_y_max - cof_y_min)) + self.IDparam[mkr.ID].cof_y_min

            # store current state
            currentState[mkr.ID] = 1

            # send osc data
            # self.sendOSC(mkr.pos_x_scaled, '/m' + str(mkr.ID) + '/pos_x')
            # self.sendOSC(mkr.d_pos_x, '/m' + str(mkr.ID) + '/d_pos_x')
            # self.sendOSC(mkr.pos_y_scaled, '/m' + str(mkr.ID) + '/pos_y')
            # self.sendOSC(mkr.d_pos_y, '/m' + str(mkr.ID) + '/d_pos_y')
            # self.sendOSC(mkr.force_scaled, '/m' + str(mkr.ID) + '/force')
            # self.sendOSC(mkr.d_force, '/m' + str(mkr.ID) + '/d_force')
            # self.sendOSC(mkr.cof_x_scaled, '/m' + str(mkr.ID) + '/cof_x')
            # self.sendOSC(mkr.cof_y_scaled, '/m' + str(mkr.ID) + '/cof_y')
            # self.sendOSC(mkr.d_cof_x, '/m' + str(mkr.ID) + '/d_cof_x')
            # self.sendOSC(mkr.d_cof_y, '/m' + str(mkr.ID) + '/d_cof_y')
            # self.sendOSC(np.rad2deg(mkr.rot), '/m' + str(mkr.ID) + '/rot')
            # self.sendOSC(np.rad2deg(mkr.d_rot), '/m' + str(mkr.ID) + '/d_rot')
            # self.sendOSC(mkr.ID, '/m' + str(mkr.ID) + '/id')
            # osc.send_message(b'/ids', id_list)
            pos_x.append(mkr.pos_x)
            pos_y.append(mkr.pos_y)
            force.append(mkr.force)
            cof_x.append(mkr.cof_x)
            cof_y.append(mkr.cof_y)
            # angle.append(np.rad2deg(mkr.rot))
            angle.append(mkr.rot)
            id_list.append(mkr.ID)
            radius.append(mkr.radius)

            if self.currentID == mkr.ID and mkr.ID is not 0:
                # send current parameters
                self.ui.progressBar_posx.setValue(mkr.pos_x_scaled * 10)
                self.ui.progressBar_posy.setValue(mkr.pos_y_scaled * 10)
                self.ui.progressBar_force.setValue(mkr.force_scaled * 10)
                self.ui.progressBar_cof_x.setValue(mkr.cof_x_scaled * 10)
                self.ui.progressBar_cof_y.setValue(mkr.cof_y_scaled * 10)
                self.ui.value_posx.setText(('%.01f' % mkr.pos_x_scaled))
                self.ui.value_posy.setText(('%.01f' % mkr.pos_y_scaled))
                self.ui.value_force.setText(('%.01f' % mkr.force_scaled))
                self.ui.value_cof_x.setText(('%.01f' % mkr.cof_x_scaled))
                self.ui.value_cof_y.setText(('%.01f' % mkr.cof_y_scaled))

        # self.sendOSC(pos_x, '/pos_x')
        # self.sendOSC(pos_y, '/pos_y')
        # self.sendOSC(force, '/force')
        # self.sendOSC(cof_x, '/cof_x')
        # self.sendOSC(cof_y, '/cof_y')
        # self.sendOSC(angle, '/angle')
        # self.sendOSC(id_list, '/id')
        # self.sendOSC(radius, '/radius')
        self.osc.send_message(b'/pos_x', pos_x)
        self.osc.send_message(b'/pos_y', pos_y)
        self.osc.send_message(b'/force', force)
        self.osc.send_message(b'/cof_x', cof_x)
        self.osc.send_message(b'/cof_y', cof_y)
        self.osc.send_message(b'/angle', angle)
        self.osc.send_message(b'/id', id_list)
        self.osc.send_message(b'/radius', radius)

        # # send OSC messages when markers disappear
        # index = np.where((np.asarray(self.prevState, dtype=np.int8) - np.asarray(currentState, dtype=np.int8)) == 1)
        # # print(index[0])
        # for i in index[0]:
        #     # print('send message to %d' % i)
        #     # print('/m' + str(i) + '/force')
        #     self.sendOSC(0.0, '/m' + str(i) + '/force')
        #     # self.sendOSC(0.0, '/m' + str(i) + '/rot')
        #     self.sendOSC(0.0, '/m' + str(i) + '/cof_x')
        #     self.sendOSC(0.0, '/m' + str(i) + '/cof_y')

        # update state
        self.prevState = currentState

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

    def sendOSC_coords(self, coords, force):
        msgStructX = osc_message_builder.OscMessageBuilder(address='/coords/x')
        msgStructY = osc_message_builder.OscMessageBuilder(address='/coords/y')
        msgStructForce = osc_message_builder.OscMessageBuilder(address='/coords/force')
        for m in coords:
            msgStructX.add_arg(m[1])
            msgStructY.add_arg(m[0])
        for f in force:
            msgStructForce.add_arg(f)
        msgStructX = msgStructX.build()
        msgStructY = msgStructY.build()
        msgStructForce = msgStructForce.build()
        self.client.send(msgStructX)
        self.client.send(msgStructY)
        self.client.send(msgStructForce)

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
        self.ui.comboBox.addItem('12')
        self.ui.comboBox.addItem('51')
        self.ui.comboBox.addItem('80')
        self.ui.comboBox.addItem('101')

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
        self.IDparam[self.currentID].cof_x_max = self.ui.doubleSpinBox_cof_x_max.value()
        self.IDparam[self.currentID].cof_x_min = self.ui.doubleSpinBox_cof_x_min.value()
        self.IDparam[self.currentID].cof_y_max = self.ui.doubleSpinBox_cof_y_max.value()
        self.IDparam[self.currentID].cof_y_min = self.ui.doubleSpinBox_cof_y_min.value()

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

        self.ui.doubleSpinBox_force_sens.setRange(0, 20000)
        self.ui.doubleSpinBox_force_sens.setValue(3000)
        self.ui.doubleSpinBox_force_sens.setDecimals(0)
        self.ui.doubleSpinBox_force_sens.valueChanged.connect(self.onSensivityChanged)
        self.ui.doubleSpinBox_cof_sens.setRange(0, 50000)
        self.ui.doubleSpinBox_cof_sens.setValue(5)
        self.ui.doubleSpinBox_cof_sens.setDecimals(0)
        self.ui.doubleSpinBox_cof_sens.valueChanged.connect(self.onSensivityChanged)

    def onSensivityChanged(self, value):
        self.force_sensitivity = self.ui.doubleSpinBox_force_sens.value()
        self.cof_sensitivity = self.ui.doubleSpinBox_cof_sens.value()

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

        if self.ui.doubleSpinBox_cof_x_max.value() >= self.ui.doubleSpinBox_cof_x_min.value():
            self.ui.progressBar_cof_x.setRange(10 * self.ui.doubleSpinBox_cof_x_min.value(), 10 * self.ui.doubleSpinBox_cof_x_max.value())
        else:
            self.ui.progressBar_cof_x.setRange(10 * self.ui.doubleSpinBox_cof_x_max.value(), 10 * self.ui.doubleSpinBox_cof_x_min.value())

        if self.ui.doubleSpinBox_cof_y_max.value() >= self.ui.doubleSpinBox_cof_y_min.value():
            self.ui.progressBar_cof_y.setRange(10 * self.ui.doubleSpinBox_cof_y_min.value(), 10 * self.ui.doubleSpinBox_cof_y_max.value())
        else:
            self.ui.progressBar_cof_y.setRange(10 * self.ui.doubleSpinBox_cof_y_max.value(), 10 * self.ui.doubleSpinBox_cof_y_min.value())

    def initProgressBar(self):
        # Change corresponding progress bar range
        self.ui.progressBar_posx.setRange(10 * self.ui.doubleSpinBox_posx_min.value(), 10 * self.ui.doubleSpinBox_posx_max.value())
        self.ui.progressBar_posy.setRange(10 * self.ui.doubleSpinBox_posy_min.value(), 10 * self.ui.doubleSpinBox_posy_max.value())
        self.ui.progressBar_force.setRange(10 * self.ui.doubleSpinBox_force_min.value(), 10 * self.ui.doubleSpinBox_force_max.value())
        self.ui.progressBar_cof_x.setRange(10 * self.ui.doubleSpinBox_cof_x_min.value(), 10 * self.ui.doubleSpinBox_cof_x_max.value())
        self.ui.progressBar_cof_y.setRange(10 * self.ui.doubleSpinBox_cof_y_min.value(), 10 * self.ui.doubleSpinBox_cof_y_max.value())

    def closeEvent(self, event):
        print('Exit application')
        if self._buttonFlag:
            sc.close_sensel(self.handle, self.frame)
        sys.exit()


if __name__ == '__main__':
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        app = QtGui.QApplication(sys.argv)
        window = ForceStamp()
        window.show()
        sys.exit(app.exec_())
