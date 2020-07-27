import requests
import os
import json
import base64
import datetime
from labelme.config import get_config

class ImageHandler:

    need_railedge_labeled = {}
    cache_path = {}
    client = None

    def __init__(self,server,port,username,password,cache):
        '''
        '''
        self.config = get_config()
        self.server = server
        self.port = str(port)
        
        self.username = username
        self.password = password

        self.cache = cache
        if not os.path.isdir(self.cache):
            os.makedirs(self.cache)

        self.url_login = "http://{}:{}/accounts/login/".format(self.server,self.port)
        self.url_api_image = "http://{}:{}/simple_labeler/api/images".format(self.server,self.port)
        self.url_api_edge =  "http://{}:{}/railedge_labeler/api/railedge".format(self.server,self.port)
        self.url_api_hit = "http://{}:{}/simple_labeler/api/hits".format(self.server,self.port)
        self.url_api_hit_image = "http://{}:{}/simple_labeler/api/hit_images".format(self.server,self.port)

        self._initialize_client()
    
    def _initialize_client(self):
        self.client = requests.Session()
        response = self.client.get(self.url_login)
        self.headers = {"X-CSRFToken":response.headers['Set-Cookie'].split('=')[1].split(';')[0],
                        "Referer":"",
                        "X-Requested-With":'XMLHttpRequest'}
        response = self.client.post(self.url_login,data={"username":self.username,"password":self.password},headers=self.headers)

        if response.status_code != 200:
            raise RuntimeError("Invalid response from server at {}, error code {}".format(self.url_login,response.status_code))
        elif response.url.lower() == self.url_login.lower():
            # If we are still at the login url, then login was not successful
            raise ValueError("Invalid credentials")
        else:
            print("sucessfully logged in")

    def get_new_hits(self,max_number_hits=None,max_images_downloaded=None):
        '''
        Download the next hits and corresponding images
        max_number_hits: integer specifying how many hits to download
        max_images_downloaded: integer specifying how many images to download total over all hits
            Initially implemented for testing purposes only. Value is tracked as self. _images_downloaded
        '''

        if max_number_hits is None:
            max_number_hits = int(self.config['max_hits_to_cache'])
        if max_images_downloaded is None:
            max_images_downloaded = int(self.config['max_images_to_cache'])

        # first get the next hits
        request = "hit_type__hit_type=RailEdge&completed=False"
        url = "{}?{}&format=json".format(self.url_api_hit,request)
        response = self.client.get(url)
        hits = response.json() #these are the hits that are not yet completed

        self._images_downloaded = 0
        self._max_images_downloaded = max_images_downloaded
        for hit_count,hit in enumerate(hits):
            if hit_count > max_number_hits:
                break
            if self._images_downloaded >= self._max_images_downloaded:
                break
            print("Processing hit {}".format(hit["id"]))
            image_path = os.path.join(self.cache,"hit_{:08.0f}".format(hit["id"]),"images")
            if not os.path.isdir(image_path):
                os.makedirs(image_path)
                existing_image_ids = {}
            else:
                existing_image_ids = {int(fname[:fname.find(".")]):True for fname in os.listdir(image_path)}

            # Now get the hit image for the remaining images in the hit, then request all of them
            #   Dont forget to handle pagination
            hit_images_combined = []
            request = "hit_id={}".format(hit["id"])
            page_id = 0
            while True:
                page_id += 1
                url = "{}?{}&completed=False&format=json&page={}".format(self.url_api_hit_image,request,page_id)
                response = self.client.get(url)
                hit_images = response.json()
                
                # see if we are at the last page
                if "detail" in hit_images:
                    if hit_images["detail"] == "Invalid page.":
                        break
                else:
                    hit_images_combined.extend([hit_image["image"] for hit_image in hit_images])
                    image_ids = self._download_images_from_hit_images(hit_images,image_path,existing_image_ids)
                    #image_ids = []
                    self.need_railedge_labeled[hit["id"]] = image_ids

            with open(os.path.join(self.cache,"hit_{:08.0f}".format(hit["id"]),"images_in_hit.json"),"w") as fp:
                json.dump(hit_images_combined,fp)

    def _download_images_from_hit_images(self,hit_images,image_path,existing_image_ids):
        '''
        Given the json response from hit_image api, download all of the images listed in hit_images
        The format of the json is a list of dicts

        Returns a list of the image ids
        '''
        print("\tDownloading images for hit {}".format(hit_images[0]["hit"]))
        images = []
        for hit_image in hit_images:
            #print("\timage_id: {}".format(hit_image["image"]))
            image_id = hit_image["image"]
            images.append(image_id)
            if image_id in existing_image_ids:
                continue
            request = "id={}".format(image_id)
            url = "{}?{}&format=json".format(self.url_api_image,request)
            response = self.client.get(url)
            response_json = response.json()
            if len(response_json) == 1:
                self._image_decoder_base64(response_json[0],image_path)
            elif len(response_json) == 0:
                raise RuntimeError("Did not get an image {} for hit {}".format(image_id,hit_image["hit"]))
            else:
                raise RuntimeError("Got more than one image response for image {} for hit {}".format(image_id,hit_image["hit"]))
            
            self._images_downloaded += 1
            if self._images_downloaded > self._max_images_downloaded:
                return images

        return images

    def _image_decoder_base64(self,raw_data,image_path):
        '''
        Take in dict of pk and image_base64 and save the image to image_path
        '''
        image_id = raw_data["pk"]
        image = raw_data["image_base64"]
        header_offset = image.find(",")+1 # strip off the header
        header = image[:header_offset]
        image = base64.b64decode(image[header_offset:]) # convert to bytestring
        if "image/jpg" in header:
            image_format="jpeg"
        elif "image/jpeg" in header:
            image_format="jpeg"
        elif "image/png" in header:
            image_format="png"
        else:
            raise TypeError("No valid image type returned from server, header is:\n\t{}".format(header))
        
        fname = "{}.{}".format(image_id,image_format)
        with open(os.path.join(image_path,fname),"wb") as fp:
            fp.write(image)

    def submit_label_file(self,fname):
        '''
        Submit the label file to the server
        '''
        with open(fname,'r') as fp:
            data = json.load(fp)
        data = self.submit_label(data)
        with open(fname,'w') as fp:
            json.dump(data,fp,indent=2)
        return data

    def submit_label(self,data):
        '''
        Submit the label for an image
        Data is a dict

        This function does:
            - Validated data
            - Sends the data to the server
            - Validates the server response is as expected
            - Stores the label in the cache
        '''

        # After the data is uploaded to the server, 'server_response' is appended to the data
        #   Thus there is no need to re-upload it
        if "server_response_created_object_id" in data:
            return data

        # convert the normal labelme format of the points to the x,y coordinates the server is expecting
        data = self._convert_labelme_to_edges(data)

        # Make sure the correct information is in the data
        self._validate_submission_railedge(data)

        response = self.client.post(self.url_api_edge,headers=self.headers,data=data)
        # Validate the response is as expected
        if response.status_code != 200:
            raise ValueError("Submitting a new label resulted in a status code of {} not 200.\
                                \nlabel for image {} in hit {} was likely not saved".format(response.status_code,data["image_id"],data["hit_id"]))
        if "success" not in response.json():
            raise RuntimeError("Server did not respond with 'success' when the label was submitted.\
                                \nLabel for image {} in hit {} was likely not saved".format(data["image_id"],data["hit_id"]))
        
        # now cache the result just in case
        response_data = response.json()
        if response.status_code == 200 and "success" in response_data:

            data["server_response_status_code"] = response.status_code
            data["server_response_message"] = response_data
            data["server_response_created_object_id"] = int(response_data["success"].split("'")[1])
            data["submit_time_utc"] = datetime.datetime.utcnow().isoformat()

        return data

    def _convert_labelme_to_edges(self,data):
        '''
        Convert the lableme shape format to the cooridate format the server is expecting
        '''
        for shape in data["shapes"]:
            if shape["label"] == "edge":
                pts = shape["points"]
                data["top_left_x"] = float(pts[0][0])
                data["top_left_y"] = float(pts[0][1])
                data["top_right_x"] = float(pts[1][0])
                data["top_right_y"] = float(pts[1][1])
                data["bottom_right_x"] = float(pts[2][0])
                data["bottom_right_y"] = float(pts[2][1])
                data["bottom_left_x"] = float(pts[3][0])
                data["bottom_left_y"] = float(pts[3][1])
                return data
        return data

    def _validate_submission_railedge(self,data):
        '''
        Validate a dictionary to be submitted to the server
        Does not return a value.
        Will throw error if an issue is found
        '''
        required = {"image_id": int,
                    "hit_id": int,
                    "top_left_x": float,
                    "top_left_y": float,
                    "top_right_x": float,
                    "top_right_y": float,
                    "bottom_left_x": float,
                    "bottom_left_y": float,
                    "bottom_right_x": float,
                    "bottom_right_y": float}

        try:
            data["image_id"] = int(data["image_id"])
        except:
            raise TypeError("When submitting a new label, image must be the image id as an integer")
        try:
            data["hit_id"] = int(data["hit_id"])
        except:
            raise TypeError("When submitting a new label, hit must be the hit id as an integer")
        
        for item in required:
            if type(data[item]) is not required[item]:
                raise TypeError("{} must be of type {} when submiting a label".format(item,required[item]))
