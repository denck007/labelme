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

    def scale(self,factor):
        '''
        Returns an edge object that is scaled by factor
        A factor of 2 will double all values
        A factor of 0.5 will half all values
        '''
        pt1 = QtCore.QPoint(int(self.pt1.x()*factor), int(self.pt1.y()*factor))
        pt2 = QtCore.QPoint(int(self.pt2.x()*factor), int(self.pt2.y()*factor))
        height = int(self.image_height * factor)
        width = int(self.image_width * factor)
        e = Edge(pt1,pt2,height,width)
        return e

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

#@profile
def adjust_edges_correlation(image,previous_image,previous_points,max_delta=.015):
    '''
    Given the previous image and the points selected on the previous image,
        adjust the edges for best fit on the new image
    
    Uses template matching over the image and RANSAC to determine the best guess for the rail edge 
        based on patches from the previous image

    image: QImage, Image to predict the edge locations on
    previous_image: QImage, Image to use as source for template matching
    previous_points: List of QPoints in clockwise order starting at top left
                        for the previous image. Used to create the templates 
                        and limit the search space 
    max_delta: float [0,1], The max % of image width the updated edge can move across the image width
                        at the image top and bottom
    Process for each side of the rail:
    1) Create N template patchs from previous_image centered on the line defined by
        previous_points.
    2) Preform template matching on image using template patches from step 1 over the entire image
    3) Multilpy the result from matching each template together
    4) Get x location of best match for each row of pixels
    5) RANSAC on these locations to get the best fit line that is the rail edge
    6) Enforce the bounds given by search_width    

    '''

    img = convertQImageToMat(image)
    prev_img = convertQImageToMat(previous_image)
    # Error checking on image shapes
    shape_img = img.shape
    shape_prev = prev_img.shape
    if (len(shape_img) == 3) and (len(shape_prev) == 3):
        height,width,_ = shape_img
    elif (len(shape_img) == 2) and (len(shape_prev) == 2):
        height,width,_ = shape_img
    else:
        raise TypeError("Invalid image dimensions {} for image and {} for previous_image. These must match!".format(shape_img,shape_prev))

    # Ensure proper dtypes
    if prev_img.dtype is np.dtype(np.uint8):
        prev_img = prev_img.astype(np.float32)/255.
    if img.dtype is np.dtype(np.uint8):
        img = img.astype(np.float32)/255.
    
    if not verify_points_order(previous_points):
        raise PointsIncorrectOrder("Points are not in correct order, must be clockwise from top left")

    #parameters for bluring image and the patches
    kernel = (5,5)
    sigmaX = 2

    img = cv2.resize(img,(img.shape[1]//2,img.shape[0]//2))
    prev_img = cv2.resize(prev_img,(prev_img.shape[1]//2,prev_img.shape[0]//2))

    img = cv2.GaussianBlur(img,kernel,sigmaX)
    prev_edge_left = Edge(previous_points[0],previous_points[3],height,width)
    prev_edge_left = prev_edge_left.scale(0.5)
    edge_left = process_edge_correlation(img,
                                        prev_img,
                                        prev_edge_left,
                                        max_delta=max_delta,
                                        blur_kernel=(3,3),
                                        blur_sigma=1,
                                        padding=8,
                                        num_per_side=3,
                                        patch_size=(8,450),
                                        patch_offset=150,
                                        side="left")
    
    prev_edge_right = Edge(previous_points[1],previous_points[2],height,width)
    prev_edge_right = prev_edge_right.scale(0.5)
    edge_right = process_edge_correlation(img,
                                        prev_img,
                                        prev_edge_right,
                                        max_delta=max_delta,
                                        blur_kernel=(3,3),
                                        blur_sigma=1,
                                        padding=8,
                                        num_per_side=3,
                                        patch_size=(8,450),
                                        patch_offset=-150,
                                        side="right")
    
    edge_left = edge_left.scale(2)
    edge_right = edge_right.scale(2)
    points = [edge_left.pt1,edge_right.pt1,edge_right.pt2,edge_left.pt2]
    return points

#@profile
def process_edge_correlation(img,
                                prev_img,
                                prev_edge,
                                max_delta=.015,
                                blur_kernel=(3,3),
                                blur_sigma=1,
                                padding=8,
                                num_per_side=3,
                                patch_size=(8,600),
                                patch_offset=0,
                                side=""):
    '''
    Calculate the best estimate of the rail edge using correlation of patches from the
        previous image
    img: ndarray shape: (w,h,1) dtype: np.float32, Image to search in
    prev_img: ndarray shape: (w,h,1) dtype: np.float32, Image to use for templates
    prev_edge: Edge object, describe the edge in prev_img
    max_delta: float [0,1], Updates that move the edge by more than this % of the image width
        (when x location of edge is measured at top and bottom of image) will be rejected
        and prev_edge is returned
    blur_kernel: tuple of length 2 of ints, the size of the blur kernel to be used on the patches
    blur_sigma: integer, the sigma to use in the bluring of the patches

    padding: int, number of pixels to keep template patches from the edge of the image
            Is used to removed edge effects on the image
    num_per_side: int, Number of templates to create and match
    patch_size: tuple of ints (patch_height, patch_width), How tall and wide each template patch is 
    '''
    if side != "":
        side = "-"+side

    # coordinates of patch to be taken from previous image
    y_top = np.linspace(padding,img.shape[0]-patch_size[0]-padding,num_per_side,dtype=np.int)
    y_mid = (y_top + patch_size[0]/2).astype(np.int)
    y_bot = (y_top + patch_size[0]).astype(np.int)

    x_mid = ((y_mid-prev_edge.b)/prev_edge.m+patch_offset).astype(np.int)
    x_left = (x_mid - patch_size[1]/2).astype(np.int)
    x_right = (x_mid + patch_size[1]/2).astype(np.int)

    # limit the search area to viable solution locations in the 
    img_xlim_left = max(0,int(x_left.min() - max_delta*img.shape[1]))
    img_xlim_right = min(img.shape[1]-1,int(x_right.max() + max_delta*img.shape[0]))

    # for debugging
    #img_out = img.copy()
    #img_out = (img_out*255).astype(np.uint8)

    # pre crop the image so it is not done each iteration
    img_cropped = np.ascontiguousarray(img[:,img_xlim_left:img_xlim_right,:])
    # create image indicating correlation using all the patches
    res_total = None
    for ii in range(num_per_side):
        patch = prev_img[y_top[ii]:y_bot[ii],x_left[ii]:x_right[ii],:]
        
        patch = cv2.GaussianBlur(patch,blur_kernel,blur_sigma)
        res = cv2.matchTemplate(img_cropped,patch,cv2.TM_CCORR) #TM_CCORR works just as well as TM_SQDIFF, but is 25% faster
        res = (res-res.min())/(res.max()-res.min())
        if res_total is None:
            res_total = res.copy()
        else:
            res_total += res

        # Draw on debug image
        #cv2.rectangle(img_out,(x_left[ii],y_top[ii]),(x_right[ii],y_bot[ii]),(0,0,1),thickness=3)

    
    # for debugging
    #res_total = (res_total-res_total.min())/(res_total.max()-res_total.min()) # only need to normalize to save image
    #cv2.imwrite("/home/nedenckl/labelme/correlation{}.jpeg".format(side),(res_total*255).astype(np.uint8))

    raw_edge = np.zeros((res_total.shape[0],2),dtype=np.float32)
    raw_edge[:,0] = res_total.argmax(axis=1).flatten()+patch_size[1]/2-patch_offset+img_xlim_left
    raw_edge[:,1] = np.arange(padding,padding+res_total.shape[0],1)

    # draw the max values on the edge
    #img_out[raw_edge[:-1,1].astype(np.int),raw_edge[:-1,0].astype(np.int)] = [0,0,255,0]
    

    eqn_pred, _ = linest_ransac(raw_edge,
                                n_sample=10,
                                n_iters=200,
                                inlier_thresh=1.5,
                                inlier_ratio_min=.15,
                                inlier_ratio_max=.95)
    if eqn_pred is None:
        print("linest_ransac failed to find a valid solution")
        #cv2.imwrite("/home/nedenckl/labelme/test_output{}-ransac_failed.jpeg".format(side),img_out)
        return prev_edge

    y = np.array([prev_edge.pt1.y(), prev_edge.pt2.y()])
    x = (y-eqn_pred[1])/eqn_pred[0]

    pt1 = QtCore.QPoint(x[0],y[0])
    pt2 = QtCore.QPoint(x[1],y[1])
    edge = Edge(pt1,pt2,img.shape[0],img.shape[1])


    # for debugging
    #cv2.line(img_out,
    #        (int(edge.x_at_height(0)),0),
    #        (int(edge.x_at_height(1199)),1199),
    #        (0,255,0),1)
    #cv2.line(img_out,
    #        (int(prev_edge.pt1.x()),int(prev_edge.pt1.y())),
    #        (int(prev_edge.pt2.x()),int(prev_edge.pt2.y())),
    #        (255,0,0),1)
    #cv2.imwrite("/home/nedenckl/labelme/test_output{}-success.jpeg".format(side),img_out)



    # validate 
    if abs(prev_edge.x_at_height(0) - edge.x_at_height(0)) > img.shape[1]*max_delta:
        print("Was unable to update edge due to edge moving too much at top of image")
        return prev_edge
    elif abs(prev_edge.x_at_height(img.shape[0]) - edge.x_at_height(img.shape[0])) > img.shape[1]*max_delta:
        print("Was unable to update edge due to edge moving too much at bottom of image")
        return prev_edge
    else:
        return edge

#@profile
def linest_ransac(data,n_sample=3,n_iters=50,inlier_thresh=10.,inlier_ratio_min=.15,inlier_ratio_max=1.0):
    '''
    data: ndarray with shape (n,2) dtype=np.float32
    n_sample: int, number of points to use in the fit    
    n_iters: int, maximum number of iterations
    inlier_thresh: float, distance in pixels from model to datapoint to indicate an inlier
    inlier_ratio_min: float [0,1], % of the datapoints that must be inliers
    inlier_ratio_max: float [0,1], must be > inlier_ratio_min, If this % of datapoints are inliers,
        consider it a good fit and return early. Can be set to 1 to disable. 
    '''
    
    n_datapoints = data.shape[0]
    
    best_inlier_count = 0
    best_l1 = 1e10
    best_l1_inlier = 1e10
    best_eqn = np.zeros((2),dtype=np.float32)
    iterations_since_change = 0
    
    for idx in range(n_iters):
        iterations_since_change += 1
        sample_idx = np.random.choice(n_datapoints,n_sample,replace=False)
        #sample_idx = np.sort(sample_idx)
        sample = data[sample_idx,:]
        try:
            eqn = np.polyfit(sample[:,0],sample[:,1],deg=1)
            poly = np.poly1d(eqn)
            prediction = poly(data[:,0])
        except:
            print("Error running poly fit on idx {:.0f}".format(idx))
            print(sample)
        
        x1 = data[0,0]
        x2 = data[-1,0]
        y1 = prediction[0]
        y2 = prediction[-1]
        x0 = data[:,0]
        y0 = data[:,1]

        num = np.abs( (y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1 )
        den = ( (y2-y1)**2 + (x2-x1)**2 )**0.5
        errors = num/den
        l1 = np.mean(errors)
        #print("min: {:7.2f} max: {:7.2f} mean: {:7.2f} l1: {:7.2f}".format(errors.min(),errors.max(),errors.mean(),l1))
        inlier_idx = np.where(errors<inlier_thresh)
        inlier_count = inlier_idx[0].shape[0]
        l1_inlier = np.mean(errors[inlier_idx])
        
        #print("\titer {:04d} inlier_count: {:04d} best_inlier_count: {:04d} min_inlier_count:{:04.0f} l1: {:.5f}".format(idx,inlier_count,best_inlier_count,n_datapoints*inlier_ratio_min,l1))
        if (inlier_count > best_inlier_count) and (inlier_count >= n_datapoints*inlier_ratio_min):
            #print("New best inlier count by having more inliers")
            #print("\titer {:04d} inlier_count: {:04d} best_inlier_count: {:04d} min_inlier_count:{:04.0f} best_l1: {:.5f} l1: {:.5f} best_l1_inlier: {:.5f} l1_inlier: {:.5f}".format(idx,inlier_count,best_inlier_count,n_datapoints*inlier_ratio_min,best_l1,l1,best_l1_inlier,l1_inlier))
            best_inlier_count = inlier_count
            best_eqn = eqn
            best_l1 = l1
            best_l1_inlier = l1_inlier
            best_sample_idx = sample_idx
            iterations_since_change = 0
            
        elif (inlier_count == best_inlier_count) and (l1 < best_l1) and (l1_inlier < best_l1_inlier):
            #print("Matched best liner count with better L1 error for inlier")
            #print("\titer {:04d} inlier_count: {:04d} best_inlier_count: {:04d} min_inlier_count:{:04.0f} best_l1: {:.5f} l1: {:.5f} best_l1_inlier: {:.5f} l1_inlier: {:.5f}".format(idx,inlier_count,best_inlier_count,n_datapoints*inlier_ratio_min,best_l1,l1,best_l1_inlier,l1_inlier))
            best_inlier_count = inlier_count
            best_eqn = eqn
            best_l1 = l1
            best_l1_inlier = l1_inlier
            best_sample_idx = sample_idx
            iterations_since_change = 0
            
        if best_inlier_count > (n_datapoints*inlier_ratio_max):
            #print("Met escape condition at idx {} with {} inliers, needed {:.0f}".format(idx,best_inlier_count,n_datapoints*inlier_ratio_max))
            break
    #print("Iterations since change: {}".format(iterations_since_change))

    if best_inlier_count == 0:
        return None,None
    else:
        return best_eqn, best_sample_idx
        
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

def adjust_edges_local_sobel(image,previous_image,previous_points,max_delta=.015):
    '''
    Given the previous image and the points selected on the previous image,
        adjust the edges for best fit on the new image

    Takes the vertical sobel in a region around the point in the previous image and sums it vertically.
    Take the sobel of the current image in the same region, and find the highest match
    Move the points to the point of highest match
    '''

    img = convertQImageToMat(image)
    prev_img = convertQImageToMat(previous_image)
    # Error checking on image shapes
    shape_img = img.shape
    shape_prev = prev_img.shape
    if (len(shape_img) == 3) and (len(shape_prev) == 3):
        height,width,_ = shape_img
    elif (len(shape_img) == 2) and (len(shape_prev) == 2):
        height,width,_ = shape_img
    else:
        raise TypeError("Invalid image dimensions {} for image and {} for previous_image. These must match!".format(shape_img,shape_prev))

    # Ensure proper dtypes
    if prev_img.dtype is np.dtype(np.uint8):
        prev_img = prev_img.astype(np.float32)/255.
    if img.dtype is np.dtype(np.uint8):
        img = img.astype(np.float32)/255.
    
    if not verify_points_order(previous_points):
        raise PointsIncorrectOrder("Points are not in correct order, must be clockwise from top left")

    filter_region_width = 7
    search_region_width = 15
    search_region_height = 55

    deltas = []
    points = []
    for idx, point in enumerate(previous_points):
        top = int(np.clip(point.y() - search_region_height//2, 0, shape_prev[0]))
        bottom = int(np.clip(point.y() + search_region_height//2, 0, shape_prev[0]))
        left = int(np.clip(point.x() - filter_region_width//2, 0, shape_prev[1]))
        right = int(np.clip(point.x() + filter_region_width//2, 0, shape_prev[1]))

        f = cv2.GaussianBlur(prev_img[top:bottom, left:right,0], (5,5), 2)
        f = np.abs(cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3))
        f = np.mean(f,axis=0)

        left = int(np.clip(point.x() - search_region_width//2, 0, shape_img[1]))
        right = int(np.clip(point.x() + search_region_width//2, 0, shape_img[1]))
        signal = cv2.GaussianBlur(img[top:bottom, left:right,0], (5,5), 2)
        signal = np.abs(cv2.Sobel(signal, cv2.CV_32F, 1, 0, ksize=3))
        signal = np.mean(signal,axis=0)
        
        rst = np.convolve(signal, f, mode="same")
        x = np.argmax(rst) + left
        points.append(QtCore.QPoint(x,point.y()))
        deltas.append(point.x() - x)
    print(deltas)
        
    return points
