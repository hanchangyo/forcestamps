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
import pyqtgraph as pg
from pyqtgraph.ptime import time

import cv2

import copy

from pythonosc import osc_message_builder
from pythonosc import udp_client
import argparse

import sensel_control as sc

import forcestamp


def sendOSC(msg, address):
        msgStruct = osc_message_builder.OscMessageBuilder(address=address)
        msgStruct.add_arg(msg)
        msgStruct = msgStruct.build()
        client.send(msgStruct)        


rows = 185
cols = 105
zoom = 3

# create Qt Application window
app = QtGui.QApplication([])
app.quitOnLastWindowClosed()

# Define a top-level widget to hold everything
win = QtGui.QWidget()
layout = QtGui.QGridLayout()
win.setLayout(layout)

# Create window with GraphicsView widget
grview = pg.GraphicsLayoutWidget()
# grview.show()  ## show widget alone in its own window
win.show()
win.setWindowTitle('Sensel test')
win.resize(rows * 5, cols * 5)

# set views
view1 = grview.addViewBox(row=1, col=1)  # show frame
view2 = grview.addViewBox(row=1, col=2)  # cropped marker
view3 = grview.addViewBox(row=2, col=2)  # cropped marker peak

# lock the aspect ratio so pixels are always square
view1.setAspectLocked(True)
view2.setAspectLocked(True)
view3.setAspectLocked(True)

# Create image item
img1 = pg.ImageItem(border='y')
img2 = pg.ImageItem(border='r')
img3 = pg.ImageItem(border='g')

# add image item
view1.addItem(img1)
view2.addItem(img2)
view3.addItem(img3)

# Set initial view bounds
view1.setRange(QtCore.QRectF(0, 0, rows, cols))
view2.setRange(QtCore.QRectF(0, 0, 49, 49))
view3.setRange(QtCore.QRectF(0, 0, 49, 49))
layout.addWidget(grview, 0, 0)

# Sensel initialization
handle, info = sc.open_sensel()

# Initalize frame
frame = sc.init_frame(handle)

# update interval
interval = 0  # miliseconds

lastTime = time()
fps = None

# setup OSC server
port_num = 8002
parser = argparse.ArgumentParser()
parser.add_argument("--ip", default="127.0.0.1", help="The ip of th OSC Server")
parser.add_argument("--port", type=int, default=port_num, help="The port the OSC server is listening on")
args = parser.parse_args()
client = udp_client.UDPClient(args.ip, args.port)


def update():
    global lastTime, fps
    f_image = sc.scan_frames(handle, frame, info)

    # find local peaks
    peakThreshold = 0.5
    f_image_peaks = forcestamp.findLocalPeaks(f_image, peakThreshold)

    # extract marker centers
    markerRadius = 20
    # markerRadius = 20 / 1.25
    
    # find marker objects
    markerCenters, markerCenters_raw = forcestamp.findMarker(f_image_peaks, markerRadius, 1)
    # print(markerCenters)

    markers = []
    for i in range(len(markerCenters)):
        # print(cnt)
        markers.append(forcestamp.marker(f_image, f_image_peaks, markerRadius, markerCenters[i]))
        print('markerX: ' + str(markers[0].posX))
        print('markerY: ' + str(markers[0].posY))
        # print('markerCode: ' + str(markers[0].code))

    if len(markers) > 0:
        print('markerID: ' + str(markers[0].ID))
        print('markerForce: ' + str(markers[0].sumForce()))
        print('markerVectorForce: ' + str(markers[0].vectorForce()))
        print('markerAbsRot: ' + str(np.rad2deg(markers[0].calculateAbsoluteRotation())))
        img2.setImage(np.rot90(markers[0].markerImg, 3), autoLevels=False, levels=(0, 100))
        img3.setImage(np.rot90(markers[0].markerImgPeak, 3), autoLevels=True, levels=(0, 50))

    # print('------------------------')

    # send MIDI & OSC messages
    '''
    if len(markers) == 0:
        for i in range(1, 12):
            sendOSC(0, '/m' + str(i) + '/is')
    else:
        for mkr in markers:
            # xMap = int((forcestamp.constraint(mkr.posX, 30, 155) - 30) / 125 * 127)
            # yMap = int((forcestamp.constraint(mkr.posY, 25, 75) - 25) / 50 * 127)
            # vel = int((forcestamp.constraint(mkr.sumForce(), 1000, 10000) / 9000 * 127))


            # OSC message
            # print(str(mkr.ID))
            sendOSC(1, '/m' + str(mkr.ID) + '/is')
            tilt_h = (forcestamp.constraint(mkr.vectorForce()[0], -20000, 20000) + 20000) / 40000 * 0.25
            tilt_v = (forcestamp.constraint(mkr.vectorForce()[1], -20000, 20000) + 20000) / 40000 * 0.25
            sendOSC(mkr.posX, '/m' + str(mkr.ID) + '/posX')
            sendOSC(mkr.posY, '/m' + str(mkr.ID) + '/posY')
            # sendOSC(mkr.sumForce(), '/m' + str(mkr.ID) + '/force')
            # print(mkr.sumForce())
            sendOSC(tilt_h, '/m' + str(mkr.ID) + '/tilt_h')
            sendOSC(tilt_v, '/m' + str(mkr.ID) + '/tilt_v')
    '''

    # print(markers)


    peaks = forcestamp.findPeakCoord(f_image_peaks)

    f_image_show = copy.deepcopy(f_image)
    if np.max(f_image_show) > 0:
        f_image_show = f_image_show / np.max(f_image_show) * 255
    f_image_show = cv2.cvtColor(f_image_show.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # for i in dotRegions:
    #     for cnt in i:
    #         cv2.circle(
    #             f_image_show,
    #             (np.int(cnt[::-1][0]), np.int(cnt[::-1][1])),
    #             3,
    #             (255, 255, 0)
    #         )

    for cnt in peaks:
        cv2.circle(
            f_image_show,
            (np.int(cnt[::-1][0]), np.int(cnt[::-1][1])),
            0,
            (0, 255, 255)
        )

    for cnt in markerCenters_raw:
        cv2.circle(
            f_image_show,
            (np.int(cnt[::-1][0]), np.int(cnt[::-1][1])),
            0,
            (0, 255, 0)
        )

    # # display ID
    # if len(markers) > 0:
    #     for mkr in markers:
    #         # print(codes)
    #         ID = mkr.ID
    #         font = cv2.FONT_HERSHEY_SIMPLEX
    #         cv2.putText(f_image_show, str(ID), (int(mkr.posX), int(mkr.posY)), font, 1, (255, 255, 255), 2, cv2.LINE_AA)


    # img1.setImage(np.rot90(f_image, 3), autoLevels=True, levels=(0, 50))
    # img1.setImage(np.rot90(f_image_peaks, 3), autoLevels=True, levels=(0, 50))
    img1.setImage(np.rot90(f_image_show, 3), autoLevels=True, levels=(0, 50))

    now = time()
    dt = now - lastTime
    lastTime = now
    if fps is None:
        fps = 1.0 / dt
    else:
        s = np.clip(dt * 3., 0, 1)
        fps = fps * (1 - s) + (1.0 / dt) * s

    # print('%0.2f fps' % fps)
    QtGui.QApplication.processEvents()


# update using timer
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(interval)

if __name__ == "__main__":
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
        print('Closed the window')
        sc.close_sensel(handle, frame)
        
