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

import sensel_control as sc

import forcestamp

# def sendOSC(msg, address):
#         msgStruct = osc_message_builder.OscMessageBuilder(address=address)
#         msgStruct.add_arg(msg)
#         msgStruct = msgStruct.build()
#         client.send(msgStruct)        


rows = 185
cols = 105
zoom = 4

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
win.resize(rows * 7, cols * 7)

# set views
view1 = grview.addViewBox(row=1, col=1)  # raw frame
view2 = grview.addViewBox(row=1, col=2)  # annotations
view3 = grview.addViewBox(row=2, col=2)  # threshold
view4 = grview.addViewBox(row=2, col=1)  # threshold

# lock the aspect ratio so pixels are always square
view1.setAspectLocked(True)
view2.setAspectLocked(True)
view3.setAspectLocked(True)
view4.setAspectLocked(True)

# Create image item
img1 = pg.ImageItem(border='y')
img2 = pg.ImageItem(border='r')
img3 = pg.ImageItem(border='g')
img4 = pg.ImageItem(border='b')

# add image item
view1.addItem(img1)
view2.addItem(img2)
view3.addItem(img3)
view4.addItem(img4)

# Set initial view bounds
view1.setRange(QtCore.QRectF(0, 0, rows, cols))
view2.setRange(QtCore.QRectF(0, 0, rows, cols))
view3.setRange(QtCore.QRectF(0, 0, rows, cols))
view4.setRange(QtCore.QRectF(0, 0, 60, 60))
layout.addWidget(grview, 0, 0)

# Sensel initialization
handle, info = sc.open_sensel()

# Initalize frame
frame = sc.init_frame(handle, detail=0, baseline=0)

# update interval
interval = 0  # miliseconds

lastTime = time()
fps = None

BlobTracker = forcestamp.TrackBlobs()
marker_radii = [55 / 2 / 1.25, 17 / 1.25, 20]
MarkerTracker = forcestamp.TrackMarkers(radii=marker_radii)


def update():
    global lastTime, fps, info, handle, frame
    try:
        f_image = sc.scan_frames(handle, frame, info)
    except UnboundLocalError:
        sc.close_sensel(handle, frame)
        # Sensel initialization
        handle, info = sc.open_sensel()
        # Initalize frame
        frame = sc.init_frame(handle, detail=0, baseline=0)
        f_image = sc.scan_frames(handle, frame, info)

    # print(np.max(f_image))

    # find blobs from the image
    blobs, contours, hierarchy, areas, cx, cy, forces, f_image_thre = forcestamp.detectBlobs(f_image, areaThreshold=1000)
    # print(contours)

    # update blob information
    blobs = BlobTracker.update(f_image)

    # update marker information
    MarkerTracker.update(f_image, blobs)

    # prepare image to show
    f_image_show = copy.deepcopy(f_image)
    if np.max(f_image_show) > 0:
        f_image_show = f_image_show / np.max(f_image_show) * 255
    f_image_show = cv2.cvtColor(f_image_show.astype(np.uint8),
                                cv2.COLOR_GRAY2RGB
                                )

    # f_image_thre = np.zeros((cols, rows), dtype=np.uint8)
    # if np.max(f_image) > 0:
    #     f_image_thre = forcestamp.findLocalPeaks(f_image, threshold=0.2).astype(np.uint8)
    # peak_coords = forcestamp.findPeakCoord(f_image_peaks)
    # f_image_thre = forcestamp.findSubpixelPeaks(peak_coords, f_image)

    # # draw contours
    # if len(blobs) > 0:
    #     for b in blobs:
    #         cv2.drawContours(
    #             f_image_show,
    #             [b.contour],
    #             0,
    #             (0, 255, 0),
    #             1
    #         )

    # display force
    # if len(blobs) > 0:
    #     for b in blobs:
    #         font = cv2.FONT_HERSHEY_SIMPLEX
    #         cv2.putText(
    #             f_image_show,
    #             '%0.1f' % b.force,
    #             (int(b.cx) + 5, int(b.cy) + 10),
    #             font,
    #             0.3,  # font size
    #             (0, 255, 0),
    #             1,
    #             cv2.LINE_AA
    #         )
    # # display appeared time
    # if len(blobs) > 0:
    #     for b in blobs:
    #         font = cv2.FONT_HERSHEY_SIMPLEX
    #         cv2.putText(
    #             f_image_show,
    #             '%5.2f' % b.t_appeared,
    #             (int(b.cx), int(b.cy) + 10),
    #             font,
    #             0.3,  # font size
    #             (0, 255, 255),
    #             1,
    #             cv2.LINE_AA
    #         )

    # display ID
    # if len(blobs) > 0:
    #     for b in blobs:
    #         font = cv2.FONT_HERSHEY_SIMPLEX
    #         cv2.putText(
    #             f_image_show,
    #             'ID: %d' % b.ID,
    #             (int(b.cx) + 5, int(b.cy) + 0),
    #             font,
    #             0.3,  # font size
    #             (255, 255, 255),
    #             1,
    #             cv2.LINE_AA
    #         )

    # show peaks
    if len(blobs) > 0:
        for b in blobs:
            cv2.circle(
                f_image_show,
                (np.int(b.cx), np.int(b.cy)),
                0,
                (255, 0, 0)
            )

    # show marker center
    for mkr in MarkerTracker.markers:
        cv2.circle(
            f_image_show,
            (np.int(mkr.pos_x), np.int(mkr.pos_y)),
            3,
            (0, 255, 0)
        )
        cv2.line(
            f_image_show,
            (np.int(mkr.pos_x), np.int(mkr.pos_y)),
            (np.int(mkr.pos_x + mkr.radius * np.sin(mkr.rot)), np.int(mkr.pos_y + mkr.radius * np.cos(mkr.rot))),
            (255, 0, 0),
            1
        )
        cv2.line(
            f_image_show,
            (np.int(mkr.pos_x), np.int(mkr.pos_y)),
            (np.int(mkr.pos_x + 1 * mkr.cof_x), np.int(mkr.pos_y - 1 * mkr.cof_y)),
            (255, 255, 0),
            1
        )

    # show marker blobs
    for mkr in MarkerTracker.markers:
        for b in mkr.blobs:
            cv2.circle(
                f_image_show,
                (np.int(b.cx), np.int(b.cy)),
                0,
                (0, 255, 0)
            )

    # display marker radius
    for mkr in MarkerTracker.markers:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            f_image_show,
            'ID: %d' % mkr.ID,
            (int(mkr.pos_x) + 5, int(mkr.pos_y) + 0),
            font,
            0.3,  # font size
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        cv2.putText(
            f_image_show,
            '%d' % mkr.force,
            (int(mkr.pos_x) + 5, int(mkr.pos_y) + 10),
            font,
            0.3,  # font size
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )
        for b in mkr.blobs:
            cv2.putText(
                f_image_show,
                '%d' % b.slot,
                (int(b.c[0]) + 0, int(b.c[1]) + 3),
                font,
                0.3,  # font size
                (0, 255, 255),
                1,
                cv2.LINE_AA
            )
            # cv2.putText(
            #     f_image_show,
            #     '%d' % b.ID,
            #     (int(b.c[0]) + 0, int(b.c[1]) - 3),
            #     font,
            #     0.2,  # font size
            #     (255, 255, 255),
            #     1,
            #     cv2.LINE_AA
            # )

    for mkr in MarkerTracker.markers:
        # print(mkr.cof_x, mkr.cof_y)
        # print(mkr.code)
        # print(mkr.phaseError)
        # IDs = []
        # for b in mkr.blobs:
        #     IDs.append(b.ID)
        # print(IDs)
        # print('------------------')
        # print(mkr.phaseError)
        # print(mkr.code)
        # print(mkr.codeword)
        # print(np.rad2deg([b.phase for b in mkr.blobs]))
        # print(np.rad2deg(mkr.rot))
        print(np.sqrt(mkr.cof_x ** 2 + mkr.cof_y ** 2))
        img4.setImage(np.rot90(mkr.markerImg, 3), autoLevels=True, levels=(0, 80))

    # img1.setImage(np.rot90(f_image, 3), autoLevels=True, levels=(0, 50))
    # img1.setImage(np.rot90(f_image_peaks, 3), autoLevels=True, levels=(0, 50))
    img1.setImage(np.rot90(f_image, 3), autoLevels=False, levels=(0, 80))
    img2.setImage(np.rot90(f_image_show, 3), autoLevels=True, levels=(0, 80))
    img3.setImage(np.rot90(f_image_thre, 3), autoLevels=True, levels=(0, 80))
    # img3.setImage(np.rot90(unknown, 3), autoLevels=True, levels=(0, 80))

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
