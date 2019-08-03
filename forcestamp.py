import numpy as np
from scipy.ndimage.filters import maximum_filter, minimum_filter
import itertools
import bch
import ellipses
import forcestamp_c
import time
import cv2
from scipy.spatial import distance as dist
import copy


def findLocalPeaks(img, threshold=0.5, kernal=3):
    # apply the local maximum filter; all pixel of maximum value
    # in their neighborhood are set to 1
    local_max_g = maximum_filter(img, kernal)
    local_min_g = minimum_filter(img, kernal)

    # store local maxima
    local_max = (local_max_g == img)

    # difference between local maxima and minima
    diff = ((local_max_g - local_min_g) > threshold)
    # insert 0 where maxima do not exceed threshold
    local_max[diff == 0] = 0

    return local_max


def isDotIncluded(dot, rows=185, cols=105):
    # check if the dot is in the pad area
    if dot[0] > 0 and dot[0] < rows and dot[1] > 0 and dot[1] < cols:
        return True
    else:
        return False


def findCircles(dots, radius):
    # find circle center candidates from two dots on the circle

    # extract coordinates from dots
    x1 = dots[0][0]
    x2 = dots[1][0]
    y1 = dots[0][1]
    y2 = dots[1][1]

    # distance between pt1 and pt2
    q = np.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

    # middle point
    x3 = (x1 + x2) / 2
    y3 = (y1 + y2) / 2

    # print(radius ** 2 - (q / 2) ** 2)
    a = radius ** 2 - (q / 2) ** 2
    if a < 0:
        a = 0

    cnt1_x = x3 + np.sqrt(a) * (y1 - y2) / q
    cnt1_y = y3 + np.sqrt(a) * (x2 - x1) / q
    cnt1 = (cnt1_x, cnt1_y)

    cnt2_x = x3 - np.sqrt(a) * (y1 - y2) / q
    cnt2_y = y3 - np.sqrt(a) * (x2 - x1) / q
    cnt2 = (cnt2_x, cnt2_y)

    return cnt1, cnt2


def distance(pt1, pt2):
    # calculate distance between two points
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def findPeakCoord(img):
    # return peak coordinates from input peak image
    peaks = [tuple(coords) for coords in zip(*np.where(img == True))]
    # print(peaks)
    return peaks


def findSubpixelPeaks(peaks, img, n=7):
    # prepare an empty array of kernal size
    # n = 7
    subPeaks = []
    for pk in peaks:
        r = np.floor(n / 2)
        # crop region around peak coord with the kernal size
        cropped = cropImage(img, pk, radius=r, margin=0)

        # project values to x axis and y axis each
        x = np.sum(cropped, axis=0)
        y = np.sum(cropped, axis=1)

        # perform CoM peak detection
        # x_CoM7 = (3 * x[6] + 2 * x[5] + x[4] - x[2] - 2 * x[1] - 3 * x[0]) / (np.sum(x))
        # y_CoM7 = (3 * y[6] + 2 * y[5] + y[4] - y[2] - 2 * y[1] - 3 * y[0]) / (np.sum(y))
        if n == 7:
            x_CoM = (3 * x[6] + 2 * x[5] + x[4] - x[2] - 2 * x[1] - 3 * x[0]) / (np.sum(x))
            y_CoM = (3 * y[6] + 2 * y[5] + y[4] - y[2] - 2 * y[1] - 3 * y[0]) / (np.sum(y))
        elif n == 5:
            x_CoM = (2 * x[4] + 1 * x[3] - 1 * x[1] - 2 * x[0]) / (np.sum(x))
            y_CoM = (2 * y[4] + 1 * y[3] - 1 * y[1] - 2 * y[0]) / (np.sum(y))

        # print(x_CoM, y_CoM)
        subPeaks.append((pk[0] + x_CoM, pk[1] + y_CoM))

    return subPeaks

'''
def findMarker(peaks, markerRadius=20, distanceTolerance=1, cMode=True):
    # find marker pin peaks from the peak image
    # distanceTolerance: tolerance when finding marker center candidates
    # peaks = findPeakCoord(img)

    circleCenters = []
    # make combination of two to find circle peaks
    for peaks_comb in itertools.combinations(peaks, 2):
        if distance(peaks_comb[0], peaks_comb[1]) < markerRadius * 2.0:
            circleCenter1, circleCenter2 = findCircles(
                peaks_comb, markerRadius)
            # parse the obtained center candidates
            if isDotIncluded(circleCenter1):
                circleCenters.append(circleCenter1)
            if isDotIncluded(circleCenter2):
                circleCenters.append(circleCenter2)
    # print(len(circleCenters))
    # print(circleCenters)
    # print(peaks)

    if cMode:
        markerCentersFiltered = forcestamp_c.findMarkerCenters(circleCenters, peaks, markerRadius, distanceTolerance)
    else:
        # find marker center candidates by given radius
        markerCenters = []
        for cnt in circleCenters:
            distanceCount = 0
            inboundPeakCount = 0
            for peak in peaks:
                if distance(cnt, peak) < markerRadius + distanceTolerance and \
                   distance(cnt, peak) > markerRadius - distanceTolerance:
                    distanceCount += 1
                if distance(cnt, peak) < markerRadius - distanceTolerance:
                    inboundPeakCount += 1
            # if there are less than 3 dots on the circle
            # and less than 3 dots inside the circle
            # print(distanceCount)
            if distanceCount > 4 and inboundPeakCount < 4:
                markerCenters.append(cnt)

        # cluster and average marker center candidates to find accurate centers
        markerCentersClustered = []
        numCluster = 0
        for cnt in markerCenters:
            # for first marker, just add a new cluster
            if numCluster == 0:
                markerCentersClustered.append([cnt])
                numCluster += 1
            else:
                currentCluster = 0
                isNew = True
                minDist = 1000
                for i in range(len(markerCentersClustered)):
                    # check distance from existing cluster centers
                    dist = distance(cnt, markerCentersClustered[i][0])
                    minDist = np.minimum(minDist, dist)
                    if dist < markerRadius / 2:
                        # add the center point to current cluster
                        isNew = False
                        currentCluster = i
                        break
                if isNew:
                    # if the point does not belong to any existing clusters
                    if minDist > markerRadius * 2.1:  # do not overlap with others
                        numCluster += 1
                        markerCentersClustered.append([])
                        markerCentersClustered[numCluster - 1].append(cnt)
                else:
                    markerCentersClustered[currentCluster].append(cnt)

        # average marker center candidates to get accurate centers
        markerCentersFiltered = []
        for cluster in markerCentersClustered:
            averageCoordX = 0
            averageCoordY = 0
            for coord in cluster:
                averageCoordX += coord[0]
                averageCoordY += coord[1]
            averageCoordX /= len(cluster)
            averageCoordY /= len(cluster)
            markerCentersFiltered.append((averageCoordX, averageCoordY))

    return markerCentersFiltered
'''


def findMarkerCenter(blobs, markerRadii, distanceTolerance):
    for blobs_two in itertools.combinations(blobs, 2):
        dist = distance(blobs_two[0].c, blobs_two[1].c)
        # print(dist)
        for radius in markerRadii:
            if dist < radius * 2.0:
                center1, center2 = findCircles((blobs_two[0].c, blobs_two[1].c), radius)
                centers = [center1, center2]
                # print('centers:', centers)
                for cnt in centers:
                    if isDotIncluded(cnt):
                        temp_marker = marker(radius)
                        innerBlobCount = 0
                        # print('counting blobs')
                        for b in blobs:
                            dist = distance(center1, b.c)
                            # print(dist)
                            if dist < radius + distanceTolerance and \
                               dist > radius - distanceTolerance:
                                temp_marker.addBlob(b)
                            elif dist < radius - distanceTolerance:
                                innerBlobCount += 1
                        # print(len(temp_marker.blobs))
                        if len(temp_marker.blobs) > 6 and innerBlobCount < 3:
                            blobs_unused = [blob for blob in blobs if blob not in temp_marker.blobs]
                            # print(blobs_unused)
                            temp_marker.pos = tuple(cnt)
                            return temp_marker, blobs_unused

    return None, blobs


def findMarker(blobs, markerRadii=[20], distanceTolerance=1):
    # distanceTolerance: tolerance when finding marker center candidates

    # for combination of two blobs, find circle center
    # for the circle center, calculate distance from any other blobs
    # if there are at least 7 blobs with matching distance, confirm it as a center

    markers = []

    while len(blobs) > 1:  # while there are more than 2 blobs
        marker, blobs = findMarkerCenter(blobs, markerRadii, distanceTolerance)
        # print(marker, blobs)
        if marker is None:
            break
        else:
            markers.append(marker)

    return markers, blobs


'''
    for blobs_comb in itertools.combinations(blobs, 2):
        if distance(blobs_comb[0].c, blobs_comb[1].c) < markerRadius * 2.0:
            center1, center2 = findCircles((blobs_comb[0].c, blobs_comb[0].c), markerRadius)
            # check for coordinate validness
            if isDotIncluded(center1):
                # calculate distance from other blobs
                distanceCount = 0
                innerBlobCount = 0
                for b in blobs:
                    dist = distance(center1, b.c)
                    if dist < markerRadius + distanceTolerance and \
                       dist > markerRadius - distanceTolerance:
                       distanceCount += 1
                    elif dist < markerRadius - distanceTolerance:
                        innerBlobCount += 1
                if distanceCount > 7 and innerBlobCount < 2:
                    markers.append(marker(img, imgpeak, radius))





    circleCenters = []
    # make combination of two to find circle peaks
    for peaks_comb in itertools.combinations(peaks, 2):
        if distance(peaks_comb[0], peaks_comb[1]) < markerRadius * 2.0:
            circleCenter1, circleCenter2 = findCircles(
                peaks_comb, markerRadius)
            # parse the obtained center candidates
            if isDotIncluded(circleCenter1):
                circleCenters.append(circleCenter1)
            if isDotIncluded(circleCenter2):
                circleCenters.append(circleCenter2)
    # print(len(circleCenters))
    # print(circleCenters)
    # print(peaks)

    if cMode:
        markerCentersFiltered = forcestamp_c.findMarkerCenters(circleCenters, peaks, markerRadius, distanceTolerance)
    else:
        # find marker center candidates by given radius
        markerCenters = []
        for cnt in circleCenters:
            distanceCount = 0
            inboundPeakCount = 0
            for peak in peaks:
                if distance(cnt, peak) < markerRadius + distanceTolerance and \
                   distance(cnt, peak) > markerRadius - distanceTolerance:
                    distanceCount += 1
                if distance(cnt, peak) < markerRadius - distanceTolerance:
                    inboundPeakCount += 1
            # if there are less than 3 dots on the circle
            # and less than 3 dots inside the circle
            # print(distanceCount)
            if distanceCount > 4 and inboundPeakCount < 4:
                markerCenters.append(cnt)

        # cluster and average marker center candidates to find accurate centers
        markerCentersClustered = []
        numCluster = 0
        for cnt in markerCenters:
            # for first marker, just add a new cluster
            if numCluster == 0:
                markerCentersClustered.append([cnt])
                numCluster += 1
            else:
                currentCluster = 0
                isNew = True
                minDist = 1000
                for i in range(len(markerCentersClustered)):
                    # check distance from existing cluster centers
                    dist = distance(cnt, markerCentersClustered[i][0])
                    minDist = np.minimum(minDist, dist)
                    if dist < markerRadius / 2:
                        # add the center point to current cluster
                        isNew = False
                        currentCluster = i
                        break
                if isNew:
                    # if the point does not belong to any existing clusters
                    if minDist > markerRadius * 2.1:  # do not overlap with others
                        numCluster += 1
                        markerCentersClustered.append([])
                        markerCentersClustered[numCluster - 1].append(cnt)
                else:
                    markerCentersClustered[currentCluster].append(cnt)

        # average marker center candidates to get accurate centers
        markerCentersFiltered = []
        for cluster in markerCentersClustered:
            averageCoordX = 0
            averageCoordY = 0
            for coord in cluster:
                averageCoordX += coord[0]
                averageCoordY += coord[1]
            averageCoordX /= len(cluster)
            averageCoordY /= len(cluster)
            markerCentersFiltered.append((averageCoordX, averageCoordY))

    return markerCentersFiltered
'''

def constraint(input, const_floor, const_ceil):
    # make input to be
    # const_floor < input < const_ceil
    if input < const_floor:
        input = const_floor
    elif input > const_ceil:
        input = const_ceil
    else:
        input = input
    return input


def detectDots(img, coords, area=4):
    # detect if there are peaks in dot candidates with kernals
    # for i in range(area):
        # for j in range(area):
    x_start = coords[0] - int(area / 2)
    y_start = coords[1] - int(area / 2)
    x_end = coords[0] + int(area / 2) + 1
    y_end = coords[1] + int(area / 2) + 1
    x_start = constraint(x_start, 0, np.shape(img)[0])
    y_end = constraint(y_end, 0, np.shape(img)[1])
    x_start = constraint(x_start, 0, np.shape(img)[0])
    y_end = constraint(y_end, 0, np.shape(img)[1])
    kernal = img[x_start:x_end, y_start:y_end]
    # print(kernal)
    # print(np.sum(kernal))
    if np.sum(kernal) >= 1:
        return 1
    else:
        return 0


def extractCode(img, markerRadius, distTolerance=3):
    n = 15
    # find marker pin peaks from the peak image
    peakImg = findLocalPeaks(img, threshold=0.3)
    peaks = findPeakCoord(peakImg)
    peaks = findSubpixelPeaks(peaks, img)
    # print(peaks)

    # calculate temporary marker center
    centerPX = np.floor(np.shape(img)[0] / 2)
    centerPY = np.floor(np.shape(img)[1] / 2)
    markerCenter = (centerPX, centerPY)

    # find dots which are included in the marker
    trueDots = []
    for peak in peaks:
        dist = distance(markerCenter, peak)
        if dist < markerRadius + distTolerance and \
           dist > markerRadius - distTolerance:
            trueDots.append(peak)

    # print(trueDots)

    # find marker center from the true dots
    data = [[], []]
    for i in trueDots:
        data[0].append(i[0])
        data[1].append(i[1])

    try:
        lsqe = ellipses.LSqEllipse()
        lsqe.fit(data)
        center, width, height, phi = lsqe.parameters()
    except(IndexError):
        center = markerCenter
    except(np.linalg.linalg.LinAlgError):
        center = markerCenter
    # print('circle center: ', tuple(center))

    # make the dots as vectors from center point
    vecDots = []
    for dot in trueDots:
        vec = (dot[0] - center[0], dot[1] - center[1])
        vecMag = distance((0, 0), vec)
        # print(vec)
        try:
            vecPhs = np.arctan2(vec[1], vec[0])
        except TypeError:
            vecPhs = np.arctan2(np.real(vec[1]), np.real(vec[0]))
        vecDots.append((vecMag, vecPhs))
    # print(np.rad2deg(vecPhs) % (360 / 15))

    # 15th power to remove phase shifts
    # print(vecDots[1])
    phaseError = 0
    phaseErrorArray = []
    for i in vecDots:
        # i_error = i[1] % (2 * np.pi / n)
        # if i_error < (2 * np.pi / n) / 2:
            # i_error += (2 * np.pi / n)
        # phaseError += i_error
        phaseErrorArray.append(i[1] % (2 * np.pi / n))

    # unwrap phase
    thre = 0.005
    var = np.var(phaseErrorArray)
    phaseErrorArray_fixed = []
    for i in phaseErrorArray:
        if var > thre:
            if i < 2 * np.pi / 4 / n:
                i += (2 * np.pi / n)
        phaseErrorArray_fixed.append(i)

    # phaseError = phaseError / len(vecDots)
    phaseError = np.average(phaseErrorArray_fixed)
    # print('phase error: ' + str(np.rad2deg(phaseError)))
    # print('distribution: ', np.var(phaseErrorArray))
    # print(phaseErrorArray)
    # print(phaseErrorArray_fixed)

    # print(vecDots[0])
    vecDotsFixed = []
    for i in vecDots:
        vecDotsFixed.append((i[0], (i[1] - phaseError) % (2 * np.pi)))

    # print(vecDotsFixed)

    # fix vector representation to coordinates
    cartDotsFixed = []
    for vector in vecDotsFixed:
        x = center[0] + vector[0] * np.sin(vector[1])
        y = center[1] + vector[0] * np.cos(vector[1])
        cartDotsFixed.append((x, y))

    initDot = (center[0] + markerRadius, center[1])
    # print(initDot)

    codes = []
    dotRegions = []
    # specify dot regions
    initX = initDot[0] - center[0]
    initY = initDot[1] - center[1]
    try:
        initRad = np.arctan2(initX, initY)
    except TypeError:
        initRad = np.arctan2(np.real(initX), np.real(initY))
    for j in range(n):
        destX = center[0] + \
            markerRadius * np.sin(initRad + j * 2 * np.pi / n)
        destY = center[1] + \
            markerRadius * np.cos(initRad + j * 2 * np.pi / n)
        dotRegions.append((destX, destY))

    codes = []
    for dot in dotRegions:
        isDot = 0
        for dotData in cartDotsFixed:
            if distance(dot, dotData) < 3:
                isDot = 1
        codes.append(isDot)

    # detect dots and extract codes from marker grid
    # for dots in dotRegions:
    #     result = detectDots(img, dots)
    #     codes.append(result)

    # codes = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # print(codes)
    # print(phaseError)
    return codes, dotRegions, phaseError


def recognizeID(msg, full=False):
    # print(msg)
    # recognize ID by decoded codeword
    codeList = [
        [0],
        [32, 1, 34, 3, 68, 7, 8, 64, 14, 16, 81, 116, 104, 58, 29],
        [96, 65, 2, 115, 4, 102, 39, 9, 76, 78, 48, 19, 24, 57, 28],
        [97, 66, 5, 38, 41, 10, 77, 110, 112, 83, 20, 55, 56, 27, 92],
        [33, 67, 37, 6, 40, 105, 42, 13, 80, 82, 52, 21, 84, 26, 74],
        [98, 69, 119, 95, 11, 46, 111, 113, 23, 120, 59, 124, 93, 126, 63],
        [99, 70, 103, 108, 12, 79, 49, 51, 118, 88, 25, 123, 125, 62, 31],
        [35, 100, 71, 72, 107, 44, 15, 17, 50, 117, 86, 89, 122, 61, 30],
        [73, 18, 36],
        [75, 101, 106, 43, 45, 47, 114, 53, 22, 87, 121, 90, 60, 94, 85],
        [91, 109, 54],
        [127]
    ]

    # Recognize ID by full codeword.
    uniqueCodes = [
        np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int),
        np.array([1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0], dtype=np.int),
        np.array([1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0], dtype=np.int),
        np.array([1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0], dtype=np.int),
        np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int),
        np.array([1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0], dtype=np.int),
        np.array([1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0], dtype=np.int),
        np.array([1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0], dtype=np.int),
        np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], dtype=np.int),
        np.array([1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0], dtype=np.int),
        np.array([1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0], dtype=np.int),
        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int)
    ]

    n = 15

    IDout = 0
    if full:
        for ID in range(len(uniqueCodes)):
            code = uniqueCodes[ID]
            for i in range(n):
                recCode_shift = np.roll(msg, i)
                # print(recCode_shift)
                if recCode_shift.tolist() == code.tolist():
                    # print('found ID')
                    IDout = ID
                    break

    else:
        for ID in range(len(codeList)):
            for code in codeList[ID]:
                if msg == code:
                    IDout = ID
                    break

    return IDout


def cropImage(img, pos, radius, margin=4):
    # crop image region surrounded by circle

    # img size
    width = np.shape(img)[0]
    height = np.shape(img)[1]

    posX = int(pos[0])
    posY = int(pos[1])

    # crop size
    crop = int(round((radius + margin)) * 2 + 1)
    crop_half = int(round(radius) + margin)

    imgCropped = np.zeros((crop, crop))

    xMinFixed = 0
    xMaxFixed = crop
    yMinFixed = 0
    yMaxFixed = crop

    xMin = posX - crop_half
    xMax = posX + crop_half
    yMin = posY - crop_half
    yMax = posY + crop_half

    if xMin < 0:
        xMinFixed = -xMin
    if xMax >= width:
        xMaxFixed = crop + (width - xMax) - 1
    if yMin < 0:
        yMinFixed = -yMin
    if yMax >= height:
        yMaxFixed = crop + (height - yMax) - 1

    pxMin = constraint(posX - crop_half, 0, width)
    pxMax = constraint(posX + crop_half, 0, width)
    pyMin = constraint(posY - crop_half, 0, height)
    pyMax = constraint(posY + crop_half, 0, height)
    imgCropped[xMinFixed:xMaxFixed, yMinFixed:yMaxFixed] = \
        img[pxMin:pxMax + 1, pyMin:pyMax + 1]

    return imgCropped


def excludeMarkerPeaks(img, pos, radius, margin=2):
    # crop image region surrounded by circle
    # img size
    width = np.shape(img)[0]
    height = np.shape(img)[1]

    posX = int(pos[0])
    posY = int(pos[1])

    # crop size
    # crop = (radius + margin) * 2 + 1
    crop_half = radius + margin

    # imgCropped = np.zeros((crop, crop))

    # xMinFixed = 0
    # xMaxFixed = crop
    # yMinFixed = 0
    # yMaxFixed = crop

    # xMin = posX - crop_half
    # xMax = posX + crop_half
    # yMin = posY - crop_half
    # yMax = posY + crop_half

    # if xMin < 0:
    #     xMinFixed = -xMin
    # if xMax >= width:
    #     xMaxFixed = crop + (width - xMax) - 1
    # if yMin < 0:
    #     yMinFixed = -yMin
    # if yMax >= height:
    #     yMaxFixed = crop + (height - yMax) - 1

    pxMin = round(constraint(posX - crop_half, 0, width))
    pxMax = round(constraint(posX + crop_half, 0, width))
    pyMin = round(constraint(posY - crop_half, 0, height))
    pyMax = round(constraint(posY + crop_half, 0, height))
    # imgCropped[xMinFixed:xMaxFixed, yMinFixed:yMaxFixed] = \
    img[pxMin:pxMax + 1, pyMin:pyMax + 1] = False

    return img


def calculateForceVector(img):
    # calcuate vector of the applied force
    width = np.shape(img)[0]
    height = np.shape(img)[1]
    centerP = np.floor(np.shape(img)[0] / 2)

    vecX = 0
    vecY = 0

    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)

    xv, yv = np.meshgrid(x, y)

    vecMag = np.sqrt(np.power(xv - centerP, 2) + np.power(yv - centerP, 2))
    vecRad = np.arctan2(xv - centerP, yv - centerP)
    # print(np.sum(img))
    img_sum = np.sum(img)

    if img_sum > 0:
        vecX = np.sum(img * vecMag * np.sin(vecRad)) / img_sum
        vecY = -np.sum(img * vecMag * np.cos(vecRad)) / img_sum
    else:
        vecX = 0
        vecY = 0
    # print(vecX, vecY)

    return (vecX, vecY)


def detectBlobs(img, areaThreshold=1000, forceThreshold=6, binThreshold=2):

    contours = []
    hierarchy = []
    # moments = []
    areas = []
    cxs = []
    cys = []
    # pixelpoints = []
    forces = []
    blobs = []
    img_thre = np.zeros_like(img)

    if np.max(img) > 0:
        # img_uint8 = np.zeros_like(img, dtype=np.uint8)
        # img_uint8 = (img / np.max(img) * 255).astype(np.uint8)
        img_thre = copy.deepcopy(img) * 2
        img_thre[img_thre >= 255] = 255
        img_thre = img_thre.astype(np.uint8)

        # Binary threshold
        img_thre = cv2.threshold(img_thre,
                                 binThreshold,
                                 255,
                                 cv2.THRESH_BINARY)[1]
        # find contours
        _, contours, hierarchy = cv2.findContours(
            img_thre,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # find peaks
        img_peaks = findLocalPeaks(img, threshold=0.2).astype(np.uint8)

        # remove peaks which are included in large blobs
        # make masks for blobs over area threshold
        mask = np.zeros(img.shape, dtype=np.uint8)
        for cnt in contours:
            # print(cnt)
            area = cv2.contourArea(cnt)
            # print(area)
            if area > areaThreshold:
                cv2.drawContours(mask, [cnt], 0, 255, -1)
        img_peaks = cv2.subtract(img_peaks, mask)
        img_thre = mask

        # extract coordinates from peak image
        peaks = findPeakCoord(img_peaks)
        sub_peaks = findSubpixelPeaks(peaks, img, n=5)

        for peak in zip(peaks, sub_peaks):

            # peak coordinates
            cxs.append(peak[1][0])
            cys.append(peak[1][1])

            # force calculation
            cropped = cropImage(img, (peak[0][0], peak[0][1]), 1, margin=0)
            if np.shape(cropped)[0] == 0 or np.shape(cropped)[0] == 0:
                force = 0
            else:
                # calculate force from the raw input image
                force = np.sum(cropped)
            forces.append(force)

            # create blob objects
            b = Blob(peak[0][1], peak[0][0], 3 * 3, force, [], [])
            # if b.area < areaThreshold:
            if b.force > forceThreshold:
                blobs.append(b)
        '''
        for cnt in contours:
            # moment
            M = cv2.moments(cnt)
            moments.append(M)

            # area
            area = cv2.contourArea(cnt)
            areas.append(area)

            # centroid
            try:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
            except ZeroDivisionError:
                # print('zero division error!')
                # calculate center by four extreme points
                leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
                rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
                topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
                bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
                cx = (leftmost[0] + rightmost[0] + topmost[0] + bottommost[0]) / 2
                cy = (leftmost[1] + rightmost[1] + topmost[1] + bottommost[1]) / 2
            cxs.append(cx)
            cys.append(cy)

            # extract blob mask for exact force calculation
            mask = np.zeros(img.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], 0, 255, -1)
            # pixelpoint = np.transpose(np.nonzero(mask))
            pixelpoint = np.nonzero(mask)
            pixelpoints.append(pixelpoint)

            # calculate force from the raw input image
            force = np.sum(img[pixelpoint])
            forces.append(force)

            # create blob objects
            b = Blob(cx, cy, area, force, pixelpoint, cnt)
            if b.area < areaThreshold:
                if b.force > forceThreshold:
                    blobs.append(b)
        '''

        # print(pixelpoints)

    # print(contours)
    return blobs, contours, hierarchy, areas, cxs, cys, forces, img_thre


class Blob:
    # posX: x coordinate (0-184)
    # posY: y coordinate (0-104)
    # ID: blob ID (0-11)
    # force: force applied to the blob (0-?)
    # area: area of the blob
    # t_appeared: timestamp of the appeared time
    # points: coordinates of blob pixels

    def __init__(self, cx, cy, area, force, points, contour):
        self.cx = cx
        self.cy = cy
        self.c = (cx, cy)

        # self.ID = ID
        self.force = force
        self.area = area

        self.t_appeared = time.time()
        self.points = points
        self.contour = contour
        self.lifetime = 0

        self.slot = -1
        self.phase = 0

    def update(self, blob):
        self.cx = blob.cx
        self.cy = blob.cy
        self.c = (blob.cx, blob.cy)

        # self.ID = ID
        self.force = blob.force
        self.area = blob.area

        self.points = blob.points
        self.contour = blob.contour
        self.lifetime = 0

    def attributeID(self, ID):
        self.ID = ID
        # print('attributed ID: %d' % ID)

    def succeedTime(self, time):
        self.t_appeared = time


class TrackBlobs():
    def __init__(self):
        # set initial parameters
        self.nextID = 0
        self.prevBlobs = []
        # self.IDTable = [False] * 1000

    # def registerID(self, blob):
    #     # .index returns the index of the first item appears in the list
    #     availableID = self.IDTable.index(False)
    #     # print(self.IDTable)
    #     print(availableID)
    #     blob.attributeID(availableID)
    #     self.IDTable[availableID] = True

    #     return blob

    def update(self, img):
        # find blobs in current frame
        self.currentBlobs = detectBlobs(img, areaThreshold=1000)[0]

        # no blobs in the image
        if len(self.currentBlobs) == 0:
            # reset next ID
            self.nextID = 0
            # self.IDTable = [False] * 1000
            self.prevBlobs = []
            return self.currentBlobs

        # prepare distance matrix
        # current centroids
        currentCentroids = np.zeros((len(self.currentBlobs), 2), dtype=np.float)
        previousCentroids = np.zeros((len(self.prevBlobs), 2), dtype=np.float)

        # store blob coordinates
        for i in range(len(self.currentBlobs)):
            currentCentroids[i] = self.currentBlobs[i].c
        for i in range(len(self.prevBlobs)):
            previousCentroids[i] = self.prevBlobs[i].c

        # if there are no blobs being tracked, register all current blobs
        if len(self.prevBlobs) == 0:
            for b in self.currentBlobs:
                # print(b.c)
                # b = self.registerID(b)
                b.attributeID(self.nextID)
                # print('no prev blobs!')
                self.nextID += 1
        else:
            # calculate distance of all pairs of current and previous blobs
            distMat = dist.cdist(
                previousCentroids,
                currentCentroids,
                metric='euclidean'
            )

            # print(distMat)

            # sort the matrix by element's min values
            rows = distMat.min(axis=1).argsort()
            cols = distMat.argmin(axis=1)[rows]
            # print(rows)
            # print(cols)

            # check if the combination is already used
            usedRows = set()
            usedCols = set()

            # iterate over row, columns
            for (row, col) in zip(rows, cols):
                # ignore already examined rows, cols.
                if row in usedRows or col in usedCols:
                    continue

                # otherwise, update ID of the current blobs with
                # previous blob IDs, and maintain appeared time.
                blobID = self.prevBlobs[row].ID
                blobTime = self.prevBlobs[row].t_appeared
                self.currentBlobs[col].attributeID(blobID)
                # print('ID updated!')
                self.currentBlobs[col].succeedTime(blobTime)

                # check that we have examined the row, col
                usedRows.add(row)
                usedCols.add(col)

            # extract unchecked rows, cols
            # unusedRows = set(range(0, distMat.shape[0])).difference(usedRows)
            unusedCols = set(range(0, distMat.shape[1])).difference(usedCols)

            # print('unused rows: ', unusedRows)
            # print('unused cols: ', unusedCols)
            # if the number of prev blobs are greater than or equal to
            # current blobs, check their liftime
            for col in unusedCols:
                # self.register(inputCentroids[col])
                self.currentBlobs[col].attributeID(self.nextID)
                self.nextID += 1

        # toss the current blob information to prev
        self.prevBlobs = self.currentBlobs

        return self.currentBlobs


class marker:
    # pos_x: x coordinate (0-184)
    # delta pos_x
    # pos_y: y coordinate (0-104)
    # delta pos_y
    # ID: marker ID (1-90)
    # slot: each slot has one blob slot[0-14]
    # timestamp: time when the marker first created
    # force: force applied to marker (0-?)
    # delta force: delta force
    # cof_x: center of force for x coord
    # delta cof_x
    # cof_y: center of force for y coord
    # delta cof_y
    # rotation: orientation of marker (0-2pi)
    # delta rotation

    def __init__(self, radius):
        self.n = 15

        self.pos = (0, 0)
        self.pos_x = 0
        self.pos_y = 0
        self.d_pos_x = 0
        self.d_pos_y = 0

        self.radius = radius

        self.timestamp = time.time()

        self.blobs = []

        self.slots = [None] * self.n

        self.force = 0
        self.d_force = 0

        self.cof_x, self.cof_y = (0, 0)
        self.d_cof_x, self.d_cof_y = (0, 0)

        # make mask to exclude non marker touches
        size = int(round(self.radius)) * 2 + 1 + 5 * 2
        self.kernal_f = np.zeros((size, size))
        y, x = np.ogrid[-self.radius - 5:self.radius + 6, -self.radius - 5:self.radius + 6]
        # print(y, x)
        mask_f = x ** 2 + y ** 2 <= (self.radius + 4) ** 2
        self.kernal_f[mask_f] = 1

        # kernal for calculating center of force
        self.kernal_cof = np.zeros((size, size))
        mask_cof_outer = x ** 2 + y ** 2 <= (self.radius + 4) ** 2
        mask_cof_inner = x ** 2 + y ** 2 <= (self.radius - 8) ** 2
        self.kernal_cof[mask_cof_outer] = 1
        self.kernal_cof[mask_cof_inner] = 0

        self.uniqueCodes = [
            # np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int),
            # np.array([1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0], dtype=np.int),
            # np.array([1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.int),
            # np.array([1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0], dtype=np.int),
            # np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0], dtype=np.int),
            # np.array([1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0], dtype=np.int),
            # np.array([1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0], dtype=np.int),
            # np.array([1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0], dtype=np.int),
            # np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], dtype=np.int),
            # np.array([1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0], dtype=np.int),  # ID:9
            # np.array([1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0], dtype=np.int),
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int),
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1], dtype=np.int),  # 8, 10923 check ID:12
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1], dtype=np.int),  # 9, 10927
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1], dtype=np.int),  # 9, 10935
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1], dtype=np.int),  # 9, 10939
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1], dtype=np.int),  # 10, 10943
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1], dtype=np.int),  # 9, 10967
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1], dtype=np.int),  # 9, 10971
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1], dtype=np.int),  # 10, 10975
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1], dtype=np.int),  # 9, 10987
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1], dtype=np.int),  # 10, 10991
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1], dtype=np.int),  # 10, 10999
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1], dtype=np.int),  # 10, 11003
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int),  # 11, 11007
            np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1], dtype=np.int),  # 9, 11095
            np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1], dtype=np.int),  # 9, 11099 check?
            np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1], dtype=np.int),  # 10, 11103
            np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1], dtype=np.int),  # 9, 11115
            np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1], dtype=np.int),  # 10, 11119
            np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1], dtype=np.int),  # 10, 11127
            np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1], dtype=np.int),  # 10, 11131
            np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1], dtype=np.int),  # 11, 11135
            np.array([0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1], dtype=np.int),  # 10, 11183
            np.array([0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1], dtype=np.int),  # 10, 11191
            np.array([0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1], dtype=np.int),  # 10, 11195
            np.array([0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], dtype=np.int),  # 11, 11199
            np.array([0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1], dtype=np.int),  # 10, 11223
            np.array([0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1], dtype=np.int),  # 10, 11227
            np.array([0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1], dtype=np.int),  # 11, 11231
            np.array([0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1], dtype=np.int),  # 10, 11243
            np.array([0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1], dtype=np.int),  # 11, 11247
            np.array([0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1], dtype=np.int),  # 11, 11255
            np.array([0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1], dtype=np.int),  # 11, 11259
            np.array([0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int),  # 12, 11263
            np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1], dtype=np.int),  # 10, 11631
            np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1], dtype=np.int),  # 10, 11639
            np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1], dtype=np.int),  # 10, 11643
            np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1], dtype=np.int),  # 11, 11647
            np.array([0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1], dtype=np.int),  # 10, 11695
            np.array([0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1], dtype=np.int),  # 10, 11703
            np.array([0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1], dtype=np.int),  # 10, 11707 check ID: 51
            np.array([0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1], dtype=np.int),  # 11, 11711
            np.array([0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1], dtype=np.int),  # 10, 11735
            np.array([0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1], dtype=np.int),  # 10, 11739
            np.array([0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1], dtype=np.int),  # 11, 11743
            np.array([0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1], dtype=np.int),  # 11, 11759
            np.array([0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1], dtype=np.int),  # 11, 11767
            np.array([0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1], dtype=np.int),  # 11, 11771
            np.array([0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int),  # 12, 11775
            np.array([0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1], dtype=np.int),  # 10, 11963
            np.array([0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1], dtype=np.int),  # 11, 11967
            np.array([0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1], dtype=np.int),  # 10, 11995
            np.array([0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1], dtype=np.int),  # 11, 11999
            np.array([0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1], dtype=np.int),  # 11, 12015
            np.array([0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1], dtype=np.int),  # 11, 12023
            np.array([0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1], dtype=np.int),  # 11, 12027
            np.array([0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int),  # 12, 12031
            np.array([0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1], dtype=np.int),  # 11, 12127
            np.array([0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1], dtype=np.int),  # 11, 12143
            np.array([0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1], dtype=np.int),  # 11, 12151
            np.array([0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1], dtype=np.int),  # 11, 12155
            np.array([0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1], dtype=np.int),  # 12, 12159
            np.array([0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1], dtype=np.int),  # 11, 12215
            np.array([0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1], dtype=np.int),  # 11, 12219
            np.array([0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], dtype=np.int),  # 12, 12223
            np.array([0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1], dtype=np.int),  # 11, 12251
            np.array([0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1], dtype=np.int),  # 12, 12255
            np.array([0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1], dtype=np.int),  # 12, 12271
            np.array([0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1], dtype=np.int),  # 12, 12279
            np.array([0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1], dtype=np.int),  # 12, 12283 check ID:80, 69
            np.array([0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int),  # 13, 12287
            np.array([0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1], dtype=np.int),  # 11, 14047
            np.array([0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1], dtype=np.int),  # 11, 14063
            np.array([0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1], dtype=np.int),  # 11, 14071
            np.array([0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int),  # 12, 14079
            np.array([0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1], dtype=np.int),  # 11, 14191
            np.array([0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1], dtype=np.int),  # 11, 14199
            np.array([0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1], dtype=np.int),  # 12, 14207
            np.array([0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], dtype=np.int),  # 12, 14271
            np.array([0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1], dtype=np.int),  # 12, 14303
            np.array([0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1], dtype=np.int),  # 12, 14319
            np.array([0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1], dtype=np.int),  # 12, 14327
            np.array([0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int),  # 13, 14335
            np.array([0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], dtype=np.int),  # 12, 15295
            np.array([0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1], dtype=np.int),  # 12, 15327
            np.array([0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1], dtype=np.int),  # 12, 15343
            np.array([0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int),  # 13, 15359
            np.array([0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int),  # 13, 15871
            np.array([0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int),  # 13, 16127
            np.array([0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1], dtype=np.int),  # 13, 16255
            np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int)  # 14, 16383 check
        ]

        self.ID_fixed = False
        self.ID = 0  # do not determine ID at the first time
        self.code = np.zeros(self.n, dtype=np.int)  # raw code
        self.codeword = np.zeros(self.n, dtype=np.int)  # non-shifted 'true' codeword

        self.rot = 0
        self.d_rot = 0

        self.lifetime = 0

    def addBlob(self, blob):
        self.blobs.append(blob)

    def calculateMarkerCenter(self):
        temp_centers = []
        for blobs_two in itertools.combinations(self.blobs, 2):
            center1, center2 = findCircles((blobs_two[0].c, blobs_two[1].c), self.radius)
            centers = [center1, center2]
            for cnt in centers:
                if distance(self.pos, cnt) < self.radius:
                    # return tuple(cnt)
                    temp_centers.append(cnt)

        # print(temp_centers)
        if len(temp_centers) > 0:
            pos = np.sum(temp_centers, axis=0) / len(temp_centers)
            return pos
        else:
            return self.pos

    def update(self, blobs, img):
        # print([b.slot for b in self.blobs])
        # update blob positions
        temp_blobs = []
        for b_exist in self.blobs:
            # find blobs by ID
            for b in blobs:
                if b_exist.ID is b.ID:
                    b.slot = b_exist.slot  # succeed slot index
                    temp_blobs.append(b)

        # update center coordinate
        prev_pos = self.pos
        self.pos = self.calculateMarkerCenter()
        self.pos_x = self.pos[0]
        self.pos_y = self.pos[1]

        # update deltas
        self.d_pos_x = self.pos_x - prev_pos[0]
        self.d_pos_y = self.pos_y - prev_pos[1]
        self.d_pos = (self.d_pos_x, self.d_pos_y)

        self.blobs = temp_blobs

        self.markerImg = cropImage(img, self.pos[::-1], self.radius, margin=5)

        self.code, self.phaseError = self.extractCode()

        prev_force = self.force
        self.force = self.sumForce()

        self.d_force = self.force - prev_force

        prev_cof_x, prev_cof_y = self.cof_x, self.cof_y
        self.cof_x, self.cof_y = self.calculateCOF()
        self.d_cof_x = self.cof_x - prev_cof_x
        self.d_cof_y = self.cof_y - prev_cof_y

        # print(self.ID_fixed)
        if not self.ID_fixed:
            # print('recog ID')
            if self.checkIDConfidence():
                ID_out, shift = self.recognizeID()
                # print(shift)
                # print(ID_out)
                if ID_out == 0:
                    self.ID = ID_out
                else:
                    self.ID = ID_out
                    self.attributeSlots(shift)
                    self.ID_fixed = True
                    # print('fixed ID!')
        else:
            prev_rot = self.rot
            self.calculateRotation()
            self.d_rot = self.rot - prev_rot
            self.findSlots()

        # self.lifetime = 0
    def extractCode(self):
        # make the dots as vectors from center point
        blobPhases = []
        for b in self.blobs:
            vector = (b.c[0] - self.pos[0], b.c[1] - self.pos[1])
            try:
                vectorPhase = np.arctan2(vector[0], vector[1]) % (2 * np.pi)
            except TypeError:
                vectorPhase = np.arctan2(np.real(vector[0]), np.real(vector[1]))
            b.phase = vectorPhase
            blobPhases.append(vectorPhase)
        # print(np.rad2deg(vectorPhase) % (360 / 15))
        blobPhases = np.array(blobPhases, dtype=np.float)
        # print(blobPhases)
        # print(np.argsort(blobPhases, axis=0))
        # print('before sort')
        # for b in self.blobs:
        #     print(b.c)
        # self.blobs = [x for _, x in sorted(zip(blobPhases.tolist(), self.blobs))]
        self.blobs = np.array(self.blobs)[np.argsort(blobPhases)].tolist()
        # for i in range(len(self.blobs)):
        #     self.blobs[i].slot = i
        # print('after sort')
        # for b in self.blobs:
        #     print(b.c)
        blobPhases = np.sort(blobPhases)

        phaseError = 0
        if len(blobPhases) > 0:
            phaseError = blobPhases[0]
            blobPhases = blobPhases - phaseError
            # slots = np.linspace(0, (self.n - 1) * np.pi / self.n, num=self.n)
            # print(slots)
            # phsErrors = slots - np.array(blobPhases)
            # print(phsErrors)

        phaseErrors = []
        code = []
        for i in range(self.n):
            if i == 0:
                code.append(1)
                phaseErrors.append(0)
            else:
                isDot = False
                for phs in blobPhases:
                    if phs > np.pi / self.n * (2 * i - 1) and \
                       phs < np.pi / self.n * (2 * i + 1):
                        isDot = True
                        code.append(1)
                        phaseErrors.append(i * 2 * np.pi / self.n - phs)
                        break
                if not isDot:
                    code.append(0)
                    phaseErrors.append(0)

        phaseError = phaseError + np.sum(phaseErrors) / self.n

        return np.array(code, dtype=np.int), phaseError

    def checkIDConfidence(self):
        # check for the marker retrieval condition
        thre_force = 800
        thre_cof = 5

        if self.force > thre_force and np.sqrt(self.cof_x ** 2 + self.cof_y ** 2) < thre_cof:
            return True
        else:
            return False

    def sumForce(self):
        # print(np.shape(mask))
        img_masked = self.markerImg * self.kernal_f

        return np.sum(img_masked)

    def calculateCOF(self):
        # mask to exclude misc blobs
        img_masked = self.markerImg * self.kernal_cof
        return calculateForceVector(img_masked)

    def recognizeID(self):
        IDout = 0
        # codeword = np.zeros(self.n, dtype=np.int)
        shift = 0
        for ID in range(len(self.uniqueCodes)):
            code = self.uniqueCodes[ID]
            for i in range(self.n):
                recCode_shift = np.roll(code, i)
                # print(recCode_shift)
                if recCode_shift.tolist() == self.code.tolist():
                    # print('found ID')
                    IDout = ID
                    self.codeword = code
                    shift = i
                    # print(shift)
                    break

        return IDout, shift

    def attributeSlots(self, shift):
        # slots = np.linspace(0, self.n - 1, self.n, dtype=np.int)
        # slots = np.roll(slots, shift)
        # print(self.codeword)
        codeword_shift = np.roll(self.codeword, shift)
        # print(codeword_shift)
        # print(shift)
        shift = np.sum(codeword_shift[0:shift])  # remove zeros to obtain accurate shift
        # slot_index = np.argwhere(codeword_shift > 0)
        slot_index = np.roll(np.where(self.codeword == 1), shift)[0]
        # print(slot_index)
        # print(len(slot_index), len(self.blobs))

        # blobs are sorted in order of shifted codeword
        if len(self.blobs) is len(slot_index):
            for i in range(len(self.blobs)):
                self.blobs[i].slot = slot_index[i]
        else:
            for i in range(len(self.blobs)):
                self.blobs[i].slot = i
        # for b in self.blobs:
            # print(b.slot)

    def findSlots(self):
        for b in self.blobs:
            if b.slot < 0:
                # print(b.phase, self.rot)
                try:
                    b.slot = np.round((b.phase - self.rot) / (2 * np.pi / self.n)) % self.n
                except ValueError:
                    b.slot = -1

    def calculateRotation(self):
        vector_x = []
        vector_y = []
        # print(self.blobs)
        if len(self.blobs) > 0:
            for b in self.blobs:
                if b.slot > -1:
                    phase_shift = (b.phase - b.slot * 2 * np.pi / self.n) % (2 * np.pi)
                    vector_x.append(np.cos(phase_shift))
                    vector_y.append(np.sin(phase_shift))

            vector_x_avg = np.average(vector_x)
            vector_y_avg = np.average(vector_y)
            # print(vector_x_avg, vector_y_avg)

            if np.isnan(vector_x_avg) or np.isnan(vector_y_avg):
                self.rot = self.rot
                return True
                # vector_x_avg = 0
            # if np.isnan(vector_y_avg):
                # vector_y_avg = 0

            try:
                rot = (np.arctan2(vector_y_avg, vector_x_avg)) % (2 * np.pi)
            except RuntimeWarning:
                self.rot = self.rot
            else:
                self.rot = rot
        else:
            self.rot = self.rot


class TrackMarkers():
    def __init__(self, radii):
        # set initial parameters
        self.markers = []
        self.radii = radii
        self.distanceTolerance = 1

        self.t_threshold = 0.5

    def update(self, img, blobs):
        # for existing markers, update their information and exclude the marker's blobs from current blobs
        blobs_mkr = []
        for mkr in self.markers:
            mkr.update(blobs, img)
            for b in mkr.blobs:
                blobs_mkr.append(b)
        self.blobs_unused = [blob for blob in blobs if blob not in blobs_mkr]
        # print(self.blobs_unused)

        # for unused blobs, determine if they belong to any markers
        blob_mkr = []
        for b in self.blobs_unused:
            for mkr in self.markers:
                dist = distance(b.c, mkr.pos)
                if dist > mkr.radius - self.distanceTolerance * 1.5 and \
                   dist < mkr.radius + self.distanceTolerance * 1.5:
                    mkr.addBlob(b)
                    blob_mkr.append(b)
        self.blobs_unused = [blob for blob in self.blobs_unused if blob not in blobs_mkr]

        # for recent blobs, find marker centers
        self.t_current = time.time()

        # filter recent blobs
        # self.blobs = blobs
        self.recent_blobs = []
        for b in self.blobs_unused:
            if self.t_current - b.t_appeared < self.t_threshold:
                self.recent_blobs.append(b)

        # print(self.recent_blobs)

        # find markers
        temp_markers = []
        new_markers = []
        if len(self.recent_blobs) > 3:
            new_markers, blobs_unused = findMarker(self.recent_blobs, markerRadii=self.radii, distanceTolerance=self.distanceTolerance)
            for mkr in new_markers:
                mkr.update(blobs, img)
                # print('pos:', mkr.pos)

        # print('new markers:', new_markers)
        # check for existing markers
        if len(self.markers) is 0:
            # print('initial marker!')
            for mkr in new_markers:
                temp_markers.append(mkr)
                break
        else:
            # print('check existing markers!')
            # check existence of marker
            for mkr_exist in self.markers:
                if len(mkr_exist.blobs) < 1:
                    mkr_exist.lifetime += 1
                    if mkr_exist.lifetime < 20:
                        temp_markers.append(mkr_exist)
                else:
                    mkr_exist.lifetime = 0
                    temp_markers.append(mkr_exist)

            for mkr in new_markers:
                isExist = False
                for mkr_exist in self.markers:
                    if distance(mkr.pos, mkr_exist.pos) < 15:
                        # print('existing marker!')
                        # mkr_exist.update(img)
                        # temp_markers.append(mkr_exist)
                        isExist = True
                        break

                if not isExist:
                    # print('new marker!')
                    temp_markers.append(mkr)

        self.markers = temp_markers


'''
class marker:
    # posX: x coordinate (0-184)
    # posY: y coordinate (0-104)
    # ID: marker ID (0-11)
    # sumForce: force applied to relevant marker (0-?)
    # vecForce: vector of the applied force (0-1 vector? (dx, dy))
    # rotation (for assymetric markers)
    def __init__(self, img, imgPeak, markerRadius, pos):
        self.posX = pos[1]
        self.posY = pos[0]

        self.markerImg = cropImage(img, pos, markerRadius)

        # self.markerImgPeak = cropImage(imgPeak, pos, markerRadius)

        self.markerRadius = markerRadius

        self.code, self.dotRegions, self.phaseError = extractCode(
            self.markerImg,
            self.markerRadius
        )

        self.force = self.sumForce()

        self.vecX, self.vecY = self.vectorForce()

        self.ID = recognizeID(
            bch.bchDecode15_7(self.code)
        )
        # self.ID = 0  # do not determine ID at contact
        self.ID_fixed = False

        self.prevrot = self.calculateAbsoluteRotation()

        if self.ID is 8 or self.ID is 10 or self.ID is 11:
            self.rot = 0
        else:
            self.rot = self.calculateAbsoluteRotation()

        self.lifetime = 0

    def updateProperties(self, img, imgPeak, markerRadius, pos):
        n = 15
        self.posX = pos[1]
        self.posY = pos[0]

        self.markerImg = cropImage(img, pos, markerRadius)

        # self.markerImgPeak = cropImage(imgPeak, pos, markerRadius)

        self.code, self.dotRegions, self.phaseError = extractCode(
            self.markerImg,
            self.markerRadius
        )

        self.force = self.sumForce()

        self.vecX, self.vecY = self.vectorForce()
        # print(self.ID_fixed)
        if not self.ID_fixed:
            # print('recog ID')
            ID_out = recognizeID(self.code, full=True)
            # print(ID_out)
            if ID_out == 0:
                self.ID = recognizeID(bch.bchDecode15_7(self.code))
            else:
                self.ID = ID_out
                self.ID_fixed = True

        # if ID is not self.ID:
        #     if self.checkIDConfidence():
        #         self.ID = ID
        #         self.rot = self.calculateAbsoluteRotation()
        #     # do not update rotation
        #     self.rot = self.rot
        # else:
        if self.ID is 8 or self.ID is 10 or self.ID is 11:
            d_rot = self.prevrot - self.calculateAbsoluteRotation()
            # print(self.prevrot)
            # self.rot = self.calculateAbsoluteRotation()
            # print(d_rot)
            if self.ID is 10 and d_rot > 2 * np.pi * 13 / n and d_rot < 2 * np.pi * 1.1:
                self.rot -= d_rot - 2 * np.pi
            elif self.ID is 10 and d_rot > 2 * np.pi * 10 / n and d_rot < 2 * np.pi * 13 / n:
                self.rot -= d_rot - 2 * np.pi * 12 / n
            elif self.ID is 10 and d_rot < -2 * np.pi * 13 / n and d_rot > -2 * np.pi * 1.1:
                self.rot -= d_rot + 2 * np.pi
            elif self.ID is 10 and d_rot < -2 * np.pi * 10 / n and d_rot > -2 * np.pi * 13 / n:
                self.rot -= d_rot + 2 * np.pi * 12 / n
            elif self.ID is 8 and d_rot > 2 * np.pi * 13 / n and d_rot < 2 * np.pi * 1.1:
                self.rot -= d_rot - 2 * np.pi
            elif self.ID is 8 and d_rot > 2 * np.pi * 10 / n and d_rot < 2 * np.pi * 13 / n:
                self.rot -= d_rot - 2 * np.pi * 12 / n
            elif self.ID is 8 and d_rot < -2 * np.pi * 13 / n and d_rot > -2 * np.pi * 1.1:
                self.rot -= d_rot + 2 * np.pi
            elif self.ID is 8 and d_rot < -2 * np.pi * 10 / n and d_rot > -2 * np.pi * 13 / n:
                self.rot -= d_rot + 2 * np.pi * 12 / n
            elif self.ID is 11 and d_rot > 2 * np.pi / n * 0.8:
                self.rot -= d_rot - 2 * np.pi / n
            elif self.ID is 11 and d_rot < -2 * np.pi / n * 0.8:
                self.rot -= d_rot + 2 * np.pi / n
            else:
                self.rot -= d_rot
            self.prevrot = self.calculateAbsoluteRotation()
        else:
            # check if the ID is not equal
            ID = recognizeID(bch.bchDecode15_7(self.code))
            if ID is self.ID:
                self.rot = self.calculateAbsoluteRotation()

        self.lifetime = 0

    def checkIDConfidence(self):
        # check for the marker retrieval condition
        # print('ID change!')
        thre_force = 1500
        thre_vecx = thre_vecy = 12000
        # print(self.vecX, self.vecY)
        if self.force > thre_force and np.abs(self.vecX) < thre_vecx and np.abs(self.vecY) < thre_vecy:
            # print('accept changed ID')
            return True
        else:
            # print('maintain current ID')
            return False

    def vectorForce(self):
        return calculateForceVector(self.markerImg)

    def sumForce(self):
        return np.sum(self.markerImg)

    def calculateAbsoluteRotation(self):
        # define initial position of code with ID
        # Recognize ID by full codeword.
        uniqueCodes = [
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int),
            np.array([1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=np.int),
            np.array([1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.int),
            np.array([1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0], dtype=np.int),
            np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0], dtype=np.int),
            np.array([1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0], dtype=np.int),
            np.array([1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0], dtype=np.int),
            np.array([1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0], dtype=np.int),
            np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], dtype=np.int),
            np.array([1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0], dtype=np.int),
            np.array([1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0], dtype=np.int),
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int)
        ]

        # filter out the code by ID
        initalCode = uniqueCodes[self.ID].tolist()

        # correct the error in the received codeword
        code = bch.calculateSyndrome(self.code)

        # count the number of shifts from the initial position
        nShift = 0
        for i in range(len(code)):
            codeShift = np.roll(code, i).tolist()
            if initalCode == codeShift:
                nShift = i
                break

        n = 15
        rotation = (2 * np.pi / n * nShift - self.phaseError) % (2 * np.pi)
        # print(self.phaseError)

        return rotation
'''
