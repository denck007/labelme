
from qtpy import QtCore

import numpy as np
import cv2

class Point(object):
    '''
    Mock for the QCoreQPoint object
    '''
    def __init__(self):
        self._x = None
        self._y = None
    def setx(self,value):
        self._x = value
    def sety(self,value):
        self._y = value
    def x(self):
        return self._x
    def y(self):
        return self._y

def move_box_to_close_edges(image,original_points):
    '''
    Run hough lines at various parameters.
    Determine which points would result in the minimual adjustment
        to each side
    If the IoU of the new and original regions that have sides going through the points
        but extended to the top and bottom of the image is greater than _config['auto_detect_edges_from_previous_min_overlap']
        then update the points and return the modified values
        Otherwise return the original points
    '''
    points = original_points.copy()
    img = convertQImageToMat(image)
    print("move_box_to_close_edges is not yet implemented")
    for point in points:
        print("{},{}".format(point.x(),point.y()))
    
    return points


def convertQImageToMat(incomingImage):
    '''
    Converts a QImage into an opencv MAT format  
    source: https://stackoverflow.com/questions/19902183/qimage-to-numpy-array-using-pyside
        we are using pyqt5 not pyside, the poster was using code for pyqt5 not pyside so they had issue
    '''

    incomingImage = incomingImage.convertToFormat(4)

    width = incomingImage.width()
    height = incomingImage.height()

    ptr = incomingImage.bits()
    ptr.setsize(incomingImage.byteCount())
    arr = np.array(ptr).reshape(height, width, 4)  #  Copies the data
    return arr