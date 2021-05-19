import numpy as np
import argparse
import torch
import cv2
import time
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

#For Raw Detections
class BoundingBox:
    label = ""
    confidence = 0.0
    coordinates = np.zeros(4, dtype=np.float32)
    id = 0
    def __init__(self):
        pass
    def set_bb(self, id, label, confidence, coordinates):
        self.id = id
        self.label = label
        self.confidence = confidence
        self.coordinates = coordinates

#For Reshaped Detections
class BoundingBoxReshaped:
    label = ""
    confidence = 0.0
    coordinates_reshaped = np.zeros(4, dtype=np.int32)
    id = 0
    def __init__(self):
        pass
    def set_bb(self, id, label, confidence, coordinates_reshaped):
        self.id = id
        self.label = label
        self.confidence = confidence
        self.coordinates_reshaped = coordinates_reshaped

#Reshape bb coordinates for images of different size
def reshape_coordinates_bb (coord_in, width_i, height_i, width_o, height_o):
    coord_out = np.zeros(len(coord_in), dtype= np.int32)
    for i in range(len(coord_in)):
        coord_out[0] = int(max(float(0), coord_in[0]*width_o/width_i))  #x1
        coord_out[1] = int(max(float(0), coord_in[1]*height_o/height_i))   #y1
        coord_out[2] = int(max(float(0), coord_in[2]*width_o/width_i))  #x2
        coord_out[3] = int(max(float(0), coord_in[3]*height_o/height_i)) #y2
    return coord_out

#Create a Bounding Box object with the predictions
def bounding_box_predictions(det, bb_number, names):
    bb_predictions = [BoundingBox() for i in range(bb_number)]
    for i in range(bb_number): #scan the prediction matrix DET/PRED (they are the same)
        coordinates=[round(det[i][0].item(),3),round(det[i][1].item(),3),round(det[i][2].item(),3),round(det[i][3].item(),3)] 
        confidence = round(det[i][4].item(),5)
        obj_class_id = int(det[i][5].item())
        obj_class = names[int(det[i][5].item())]
        bb_predictions[i].set_bb(obj_class_id, obj_class, confidence, coordinates)
    return bb_predictions

#Create a Reshaped Bounding Box object with the predictions and image
def bounding_box_predictions_reshaped(bb_predictions, bb_number, I_backtorgb, colors, rows, cols):
    bb_predictions_reshaped = [BoundingBoxReshaped() for i in range(bb_number)]
    for i in range(bb_number): 
        xyxy = reshape_coordinates_bb(bb_predictions[i].coordinates, 416, 416, rows, cols) #for a different image size
        bb_predictions_reshaped[i].set_bb(bb_predictions[i].id, bb_predictions[i].label, bb_predictions[i].confidence, xyxy)

    for i in range(bb_number):  #reshaped detections on image
            label = str(bb_predictions_reshaped[i].label) + " " + str(round(bb_predictions_reshaped[i].confidence, 2))
            plot_one_box(bb_predictions_reshaped[i].coordinates_reshaped, I_backtorgb, label=label, color=colors[bb_predictions[i].id], line_thickness=1) 

    return bb_predictions_reshaped, I_backtorgb

#Get list of active taxels per bounding box on the image 
def bb_active_taxel (bb_number, bb_predictions_reshaped, TIB, skin_faces):
    taxel_predictions = np.empty((bb_number,), dtype = object)
    taxel_predictions_info = np.empty((bb_number,), dtype = object)
    for n in range(bb_number):
        faces_predictions = []
        face_index_previous = 0
        for i in range(bb_predictions_reshaped[n].coordinates_reshaped[0], bb_predictions_reshaped[n].coordinates_reshaped[2]):
            for j in range(bb_predictions_reshaped[n].coordinates_reshaped[1], bb_predictions_reshaped[n].coordinates_reshaped[3]):
                face_index = TIB.get_pixel_face_index( i,  j)
                if face_index == (-1) or face_index >= 1218:
                    print("Wrong Face")
                    break
                if face_index == face_index_previous:
                    break
                faces_predictions.append(skin_faces[face_index][0])
                faces_predictions.append(skin_faces[face_index][1])
                faces_predictions.append(skin_faces[face_index][2])

                face_index_previous = face_index
        taxel_predictions[n] = set(faces_predictions) #set rmoves duplicates
        taxel_predictions_info[n] = bb_predictions_reshaped[n].label + ", " + str(bb_predictions_reshaped[n].confidence) + ", " + str(len(set(faces_predictions))) #this is the name, conf and # active taxels per prediction
    return taxel_predictions, taxel_predictions_info

#Get taxel responses for all bounding boxes 
def taxel_responses(bb_number, S,taxel_predictions):
    total_taxel_responses = np.empty((bb_number,), dtype = object)
    total_taxels_position = np.empty((bb_number,), dtype = object)
    for n in range(bb_number):
        taxel_response = [] #empty array for the responses of a single bounding box
        taxels_position = [] #empty array for the idus of a single bounding box
        for i in taxel_predictions[n]:
            if S.taxels[i].get_taxel_response() != 0: 
                taxel_response.append(S.taxels[i].get_taxel_response()) #set rmoves duplicatess
                taxels_position.append(S.taxels[i].get_taxel_position())              
        total_taxel_responses[n] = taxel_response
        total_taxels_position[n] = taxels_position
    return total_taxel_responses , total_taxels_position