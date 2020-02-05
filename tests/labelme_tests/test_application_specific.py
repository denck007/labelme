import os
import os.path as osp
import shutil
import tempfile

import labelme.app
import labelme.config
import labelme.testing
from labelme.label_file import LabelFile

from itertools import permutations
from qtpy import QtCore
import labelme.utils
from labelme.utils import edge_adjustment,ImageHandler

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

def test_data_to_sever():
    '''
    Check that the data that is sent from the labeling tool is correct when it is sent to the server


    This test assumes that a development server is running and is specified in the config file 
    '''
    config = labelme.config.get_config(config_file="labelme/config/config_unittests.yaml")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Attempt to login to the server and download 3 images to label
        try:
            ih = ImageHandler(config["server"],config["port"],config["username"],config["password"],tmpdir)
            ih.get_new_hits(max_number_hits=4,max_images_downloaded=3)
        except:
            EnvironmentError("Could not connect to server, is it running at the location specified in the config_unittests.yaml file?")

        # Go through and get the file locations and store basic data about the images        
        labels_to_test = [{"top_left_x":671.9,
                "top_left_y":106.9,
                "top_right_x":1186.7,
                "top_right_y":138.3,
                "bottom_left_x":703.5,
                "bottom_left_y":1101.7,
                "bottom_right_x":1211.9,
                "bottom_right_y":1122.2,
                "image_id":None,
                "hit_id":None,
                "results":{"slope_left":31.481013,
                            "slope_right":39.043651,
                            "slope":35.262332,
                            "intercept_left":-21045.192405,
                            "intercept_right":-46194.800397,
                            "intercept":-32653.94157,
                            "intercept_percent":-27.211617975,
                            "center_x":943.044318,
                            "center_x_percent":0.5894026988,
                            "width_horizontal":510.961816,
                            "width_horizontal_percent":0.319351135,
                            "width_normal":510.7564755382,
                            "width_normal_percent":0.3192227972,
                            }},
                    {"top_left_x":694.0,
                    "top_left_y":94.3,
                    "top_right_x":1200.9,
                    "top_right_y":108.4,
                    "bottom_left_x":679.9,
                    "bottom_left_y":1139.5,
                    "bottom_right_x":1186.7,
                    "bottom_right_y":1136.4,
                    "image_id":None,
                    "hit_id":None,
                    "results":{"slope_left":-74.127660,
                                "slope_right":-72.394366,
                                "slope":-73.261013,
                                "intercept_left":51538.895745,
                                "intercept_right":87046.794366,
                                "intercept":69512.510279, # note this is inverted so need to add image_height to the inventor calc
                                "intercept_percent":57.9270918992,
                                "center_x":940.643701,
                                "center_x_percent":0.5879023131,
                                "width_horizontal":506.931431,
                                "width_horizontal_percent":0.3168321444,
                                "width_normal":506.8842124439,
                                "width_normal_percent":0.3168026328,
                                }},
                    {"top_left_x":690.9,
                    "top_left_y":78.5,
                    "top_right_x":1208.8,
                    "top_right_y":72.2,
                    "bottom_left_x":734.9,
                    "bottom_left_y":1167.9,
                    "bottom_right_x":1243.4,
                    "bottom_right_y":1136.4,
                    "image_id":None,
                    "hit_id":None,
                    "results":{"slope_left":24.759091,
                                "slope_right":30.757225,
                                "slope":27.758158,
                                "intercept_left":-17027.555909,
                                "intercept_right":-37107.134104,
                                "intercept":-26296.5888881,
                                "intercept_percent":-21.9138240734,
                                "center_x":968.961583,
                                "center_x_percent":0.6056009894,
                                "width_horizontal":513.997225,
                                "width_horizontal_percent":0.3212482656,
                                "width_normal":513.6640081061,
                                "width_normal_percent":0.3210400051,
                                }}]

        images = {}
        for root, _, files in os.walk(tmpdir):
            for fname in files:
                if len(images) >= 3: # only want first 3 images
                    break
                if fname[-5:] == ".jpeg":
                    images[fname[:-5]] = {'root':root,
                                            'path': os.path.join(root,fname),
                                            'hit_id': root.split(os.sep)[-1].split("_")[-1],
                                            'image_id': fname[:-5],
                                            'label':labels_to_test[len(images)],
                                            'solution':labels_to_test[len(images)]['results']
                                            }
        assert len(images) == 3, "Did not get 3 images to label from the server!"

        # Generate labels for each image, save the path to the label in the images dict
        for image_id in images:
            image = images[image_id]
            label = image['label']

            result_fname = os.path.join(image["root"],"{}.json".format(image["image_id"]))
            images[image_id]["result_fname"] = result_fname
            result = LabelFile()
            result.save(filename=result_fname,
                        shapes = [{"label":"edge","line_color":None,"fill_color":None,"points":[[label["top_left_x"],label["top_left_y"]],
                                                                                                [label["top_right_x"],label["top_right_y"]],
                                                                                                [label["bottom_right_x"],label["bottom_right_y"]],
                                                                                                [label["bottom_left_x"],label["bottom_left_y"]]],
                                                                                                "shape_type":"polygon","flags":{}}],
                        imagePath=image["path"],
                        imageHeight=1200,
                        imageWidth=1600)

        # upload the result to the server and verfiy ImageHandler._convert_labelme_to_edges()
        for image_id in images:
            image = images[image_id]
            label = image['label']

            result_fname = images[image_id]["result_fname"]
            response = ih.submit_label_file(result_fname)
            images[image_id]['railedge_id'] = response["server_response_created_object_id"]

            # Verify the updated saved data is correct.
            # ImageHandler converts the locations in the shape key to the 4 locations needed. Verify it did it correctly
            #   This tests the ImageHandler._convert_labelme_to_edges function
            for vert in ['top','bottom']:
                for lat in ['left','right']:
                    for coord in ['x','y']:
                        floatsClose(response["{}_{}_{}".format(vert,lat,coord)],label["{}_{}_{}".format(vert,lat,coord)],msg="Occured in {}_{}_{}".format(vert,lat,coord))

        # download the results from the server and verify the result is correct
        for image_id in images:
            image = images[image_id]
            result_correct = image['label']['results']
            railedge_id = image['railedge_id']

            response = ih.client.get("{}?id={}".format(ih.url_api_edge,railedge_id))
            response_json = response.json()
            assert len(response_json) == 1, "Got {} railedge labels, should only get 1!".format(len(response_json))
            
            result = response_json[0]
            for key in result_correct:
                floatsClose(result[key],result_correct[key],msg="key: {}".format(key))

def floatsClose(a,b,eps=1e-3,msg=""):
    '''
    Assert that a and b are within eps of eachother
    '''
    delta = abs(a-b)
    assert delta < eps, "Floats {} and {} are {} apart. Should be within {} of eachother, \n{}".format(a,b,delta,eps,msg)