'''
Development for moving the edge location adjustment from one image to the next

The general idea is that between 2 images that are adjacent in the real world will be
    have edges in about the same location. This will try to adjust the edges defined in a 
    previous image and alter them to be directly on the edge
'''

import cv2
import numpy as np
from itertools import permutations
import matplotlib.pyplot as plt

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

        self.m = (pt1.y-pt2.y)/(pt1.x-pt2.x + 1e-6)
        self.b = pt1.y - self.m*pt1.x
        self.angle = np.arctan2(pt1.y-pt2.y, pt1.x-pt2.x + 1e-6)
        
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






def adjust_edges(image,points,search_width=0.05,search_angle=5.):
    '''
    image is a cv2 image
    points is a list of QCoreQPoint objects (or mocks using Point object)
    search_width: float [0,1] how wide of an area to search for the edge in %/100.
    search_angle: float degrees, +/- angle  to search for angle
    adjusts locations of points based on the edges in image

    Enforces that points are ordered starting in top left and go clockwise
    '''

    if not verify_points_order(points):
        raise ValueError("Points are not in correct order, must be clockwise from top left")

    height,width = image.shape
    # Determine equations for each edge and x location at top and bottom for both
    edge_left = Edge(points[0],points[3],height,width)
    
    b_box = edge_left.get_window(search_width)
    cv2.HoughLinesP()
    
    
    edge_right = Edge(points[1],points[2],height,width)



    

    
    
    
    

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
            print("\t\tpt{}: ({:6.1f},{:6.1f})".format(idx,point.x,point.y))

    # pt0 is top left if it is above pt3 and left of pt1
    if (points[0].y >= points[3].y) or (points[0].x >= points[1].x):
        msg = "pt0 is not in top left position"
        print_error(msg,points)
        return False
    
    # pt1 is top right if it is above pt2 and right of pt0
    if (points[1].y >= points[2].y) or (points[1].x <= points[0].x):
        msg = "pt1 is not in top right position"
        print_error(msg,points)
        return False

    # pt2 is in bottom right if it is below pt1 and right of pt3
    if (points[2].y <= points[1].y) or (points[2].x <= points[3].x):
        msg = "pt2 is not in bottom right position"
        print_error(msg,points)
        return False

    # pt3 is bottom left if it is below pt0 and left of pt2
    if (points[3].y <= points[0].y) or (points[3].x >= points[2].x):
        msg = "pt3 is not in bottom right position"
        print_error(msg,points)
        return False
    
    return True

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
    @property
    def x(self):
        return self._x
    @property
    def y(self):
        return self._y

def plot_list_points(p):
    for idx,c in enumerate(['r','g','b']):
        plt.plot([p[idx].x,p[idx+1].x],[p[idx].y,p[idx+1].y],c)
    plt.show()

def test_verify_points_order():
    # Start with a simple square and do exhaustive testing
    points = [Point(x=0 ,y=0),
              Point(x=10,y=0),
              Point(x=10,y=10),
              Point(x=0 ,y=10)]
    response = verify_points_order(points)
    assert response == True, 'Case points in correct order failed'

    for idx,p in enumerate(permutations(points)):
        response = verify_points_order(p)
        if idx == 0: # first one are in correct order 
            assert response == True, 'Points are in correct order'
        else:
            msg = "idx: {:}".format(idx)
            for idx2, point in enumerate(p):
                msg += "\n\tpt{}: ({:6.0f},{:6.0f})".format(idx2,point.x,point.y)
            assert response == False, 'Points not in correct order but did not verify_points_order did not return False, points:\n'+msg
    
    points = [Point(x=1 ,y=1),
              Point(x=11,y=0),
              Point(x=10,y=10),
              Point(x=0 ,y=10)]
    response = verify_points_order(points)
    assert response == True, 'Points not in correct order'

    print("verify_points_order passed")

if __name__ == '__main__':
    test_verify_points_order()
