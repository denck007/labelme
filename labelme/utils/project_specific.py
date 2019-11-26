
from qtpy import QtCore

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

def move_box_to_close_edges(image,points):
    '''
    Run hough lines at various parameters.
    Determine which points would result in the minimual adjustment
        to each side
    If the IoU of the new and original regions that have sides going through the points
        but extended to the top and bottom of the image is greater than _config['auto_detect_edges_from_previous_min_overlap']
        then update the points and return the modified values
        Otherwise return the original points

    '''
    original_points = points.copy()
    print("move_box_to_close_edges is not yet implemented")
    for point in points:
        print("{},{}".format(point.x(),point.y()))
    
    return original_points
