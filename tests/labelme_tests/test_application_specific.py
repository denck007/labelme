import os.path as osp
import shutil
import tempfile

import labelme.app
import labelme.config
import labelme.testing

from itertools import permutations
from qtpy import QtCore
import labelme.utils
from labelme.utils import edge_adjustment

def test_verify_points_order():
    # Start with a simple square and do exhaustive testing
    points = [QtCore.QPoint(0.0,0.0),
              QtCore.QPoint(10.,0.0),
              QtCore.QPoint(10.,10.),
              QtCore.QPoint(0.0,10.)]
    response = edge_adjustment.verify_points_order(points)
    assert response == True, 'Case points in correct order failed'

    for idx,p in enumerate(permutations(points)):
        response = edge_adjustment.verify_points_order(p)
        if idx == 0: # first one are in correct order 
            assert response == True, 'Points are in correct order'
        else:
            msg = "idx: {:}".format(idx)
            for idx2, point in enumerate(p):
                msg += "\n\tpt{}: ({:6.0f},{:6.0f})".format(idx2,point.x(),point.y())
            assert response == False, 'Points not in correct order but did not verify_points_order did not return False, points:\n'+msg
    
    points = [QtCore.QPoint(1 ,1),
              QtCore.QPoint(11,0),
              QtCore.QPoint(10,10),
              QtCore.QPoint(0 ,10)]
    response = edge_adjustment.verify_points_order(points)
    assert response == True, 'Points not in correct order'
