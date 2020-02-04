'''
Development for moving the edge location adjustment from one image to the next

The general idea is that between 2 images that are adjacent in the real world will be
    have edges in about the same location. This will try to adjust the edges defined in a 
    previous image and alter them to be directly on the edge
'''
from qtpy import QtCore
import cv2
import numpy as np
from itertools import permutations
import matplotlib.pyplot as plt

from labelme.utils.project_specific_exceptions import EdgeLabelingError,EdgeNotFound,PointsIncorrectOrder

class Point(object):
    '''
    Mock for the QCoreQPoint object
    '''
    _x = None
    _y = None
    def __init__(self,x=None,y=None):
        if (x is not None) and (y is not None):
            self._x = float(x)
            self._y = float(y)
    def setx(self,value):
        self._x = value
    def sety(self,value):
        self._y = value
    def x(self,value=None):
        if value is not None:
            self._x = value
        return self._x
    def y(self,value=None):
        if value is not None:
            self._y = value
        return self._y

class Edge:
    '''
    Hold all values to define an edge in an image
    '''

    def __init__(self,pt1,pt2,image_height,image_width):
        '''
        Given 2 points determine equation for line
        '''
        self.pt1 = pt1
        self.pt2 = pt2
        self.image_height = image_height
        self.image_width = image_width

        self.m = (pt1.y()-pt2.y())/(pt1.x()-pt2.x() + 1e-6)
        self.b = pt1.y() - self.m*pt1.x()
        self.angle = np.arctan2(pt1.y()-pt2.y(), pt1.x()-pt2.x() + 1e-6)
        
        self.x_top = self.x_at_height(0)
        self.x_bot = self.x_at_height(self.image_height)

    def x_at_height(self,y):
        '''
        return the x location
        '''
        return (y - self.b) / (self.m + 1e-6)
    
    def get_window(self,search_width):
        '''
        Given a search_width return a bounding box [(top_left.x,top_left.y),(bottom_right.x,bottom_right.y)
        search_width is the +/- %/100 to add around the bounds of the previous edge
        '''
        x_left = min(self.x_top,self.x_bot)
        x_right = max(self.x_top,self.x_bot)

        added = self.image_width* search_width

        x_left = int(x_left - added)
        x_right = int(x_right + added)

        return [(x_left,0),(x_right,self.image_height)]

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

def adjust_edges(image_source,points,search_width=0.05):
    '''
    image is a QImage image, is the entire railscope image
    points is a list of QCoreQPoint objects (or mocks using Point object)
    search_width: float [0,1] how wide of an area to search for the edge in %/100.
    adjusts locations of points based on the edges in image

    Enforces that points are ordered starting in top left and go clockwise
    '''

    if not verify_points_order(points):
        raise PointsIncorrectOrder("Points are not in correct order, must be clockwise from top left")

    image = convertQImageToMat(image_source)
    shape = image.shape
    if len(shape) == 3:
        height,width,_ = shape
    elif len(shape) == 2:
        height,width = shape
    else:
        raise TypeError("Invalid image dimensions {}".format(image.shape))

    # Determine equations for each edge and x location at top and bottom for both
    ref_edge_left = Edge(points[0],points[3],height,width)
    edge_left = process_edge(image,ref_edge_left)
    
    ref_edge_right = Edge(points[1],points[2],height,width)
    edge_right = process_edge(image,ref_edge_right)
    
    points = [edge_left.pt1,edge_right.pt1,edge_right.pt2,edge_left.pt2]
    return points

def process_edge(image,ref_edge,search_width=.05):
    '''
    Given an image and an edge from a previous image, attempt to find another edge close to the ref_edge

    Processs is:
        1) Extract bounding box
        2) Histogram equalization
        3) Gaussian blur with kernel size 7
        4) Sobel to find vertical edges
        5) Canny edge detector
        6) HoughLines Probabilistic
        7) Iterate over lines. Compute the x location at to and bottom of image.
            Find the line that minimizes the distance to the reference edge

    If a valid edge is found return that edge, otherwise throw EdgeNotFound
    '''
    DEBUG = False # Set to True to show in process plots
    image_height,image_width,_ = image.shape
    b_box = ref_edge.get_window(search_width)
    roi = image[b_box[0][1]:b_box[1][1],b_box[0][0]:b_box[1][0],:]
    roi_eq = cv2.equalizeHist(roi[:,:,0])
    roi_gaus = cv2.blur(roi_eq,(11,11))

    # Using an alternate to a Canny detector
    # Take the vertical sobel, scale it to [0,255] then threshold it a
    roi_edges = cv2.Sobel(roi_gaus,cv2.CV_32F,1,0,ksize=7)
    roi_edges = np.abs(roi_edges)
    roi_edges = roi_edges/roi_edges.max()*255
    roi_edges[roi_edges<75] = 0 # threshold
    roi_edges[roi_edges>=75] = 255 # threshold
    roi_edges = roi_edges.astype(np.uint8)

    #                         image,  rho, theta,           threshold[, lines[, minLineLength[, maxLineGap]]])
    linesP = cv2.HoughLinesP(roi_edges, 1, 0.25*np.pi / 180, 50, None,        250,         50)
    
    if DEBUG:
        output = roi.copy()

    if linesP is None:
        raise EdgeNotFound("No edges were found")
    else:
        best_edge = None
        best_x_offset = 2*image_width
        for line in linesP:
            l = line[0]
            pt1 = QtCore.QPoint(l[0],l[1])
            pt2 = QtCore.QPoint(l[2],l[3])
            prop = Edge(pt1,pt2,image_width,image_height)
            if best_edge is None:
                best_edge = prop
            x_offset_top = abs(b_box[0][0]+prop.x_at_height(0)-ref_edge.x_at_height(0))
            x_offset_bottom = abs(b_box[0][0]+prop.x_at_height(image_height)-ref_edge.x_at_height(image_height))
            x_offset_total = x_offset_top + x_offset_bottom
            if DEBUG:
                cv2.line(output, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

            if x_offset_total < best_x_offset:
                best_edge = prop
                best_x_offset = x_offset_top
                if DEBUG:
                    print("ref: {:4.0f}, {:4.0f} prop: {:4.0f}, {:4.0f} errors top: {:5.1f} bot: {:5.1f} total: {:5.1f}".format(ref_edge.x_at_height(0),
                                                                                                                            ref_edge.x_at_height(image_height),
                                                                                                                            b_box[0][0]+prop.x_at_height(0),
                                                                                                                            b_box[0][0]+prop.x_at_height(image_height),
                                                                                                                            x_offset_top,
                                                                                                                            x_offset_bottom,
                                                                                                                            x_offset_total))
        # plotting for debug
        if DEBUG:
            plt.subplot(1,6,1)
            plt.title("ROI source")
            plt.imshow(roi,cmap="gray")
            plt.subplot(1,6,2)
            plt.title("Equalized")
            plt.imshow(roi_eq,cmap="gray")
            plt.subplot(1,6,3)
            plt.title("Gaussian Blur")
            plt.imshow(roi_gaus,cmap="gray")
            plt.subplot(1,6,4)
            plt.title("Edges")
            plt.imshow(roi_edges,cmap="gray")
            plt.subplot(1,6,5)
            plt.title("All Edges")
            cv2.line(output,(int(best_edge.x_at_height(0)),0),(int(best_edge.x_at_height(image_height)),image_height),(0,255,0),3,cv2.LINE_AA)
            plt.imshow(output[:,:,:3])
            plt.show()

        pt1 = QtCore.QPoint(best_edge.x_at_height(ref_edge.pt1.y())+b_box[0][0],ref_edge.pt1.y())
        pt2 = QtCore.QPoint(best_edge.x_at_height(ref_edge.pt2.y())+b_box[0][0],ref_edge.pt2.y())
        return Edge(pt1,pt2,image_height,image_width)

def verify_points_order(points):
    '''
    Ensures a list of QCoreQPoint objects (or mocks) are in order
        starting at top left and going clock-wise

    Note coordinate system is +x is right, +y is down
        +y
        /\
        |   
        pt0 -------> pt1 ---->+x
        ^             |
        |             |
        |            \|/
        pt3 <------- pt2

    Returns True if they are in order
    Returns Fale if the points need to be reordered
    '''
    def print_error(message,points):
        print(message)
        print("\tPoint Locations:")
        for idx, point in enumerate(points):
            print("\t\tpt{}: ({:6.1f},{:6.1f})".format(idx,point.x(),point.y()))

    # pt0 is top left if it is above pt3 and left of pt1
    if (points[0].y() >= points[3].y()) or (points[0].x() >= points[1].x()):
        msg = "pt0 is not in top left position"
        print_error(msg,points)
        return False
    
    # pt1 is top right if it is above pt2 and right of pt0
    if (points[1].y() >= points[2].y()) or (points[1].x() <= points[0].x()):
        msg = "pt1 is not in top right position"
        print_error(msg,points)
        return False

    # pt2 is in bottom right if it is below pt1 and right of pt3
    if (points[2].y() <= points[1].y()) or (points[2].x() <= points[3].x()):
        msg = "pt2 is not in bottom right position"
        print_error(msg,points)
        return False

    # pt3 is bottom left if it is below pt0 and left of pt2
    if (points[3].y() <= points[0].y()) or (points[3].x() >= points[2].x()):
        msg = "pt3 is not in bottom right position"
        print_error(msg,points)
        return False
    
    return True

def plot_list_points(p):
    '''
    Utility used to debug ordering errors
    '''
    for idx,c in enumerate(['r','g','b']):
        plt.plot([p[idx].x(),p[idx+1].x()],[p[idx].y(),p[idx+1].y()],c)
    plt.show()
