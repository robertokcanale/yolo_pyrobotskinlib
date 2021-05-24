import numpy as np
import argparse
import torch
import cv2
import random
from operator import add 
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
        coord_out[0] = int(max(float(0), (coord_in[0]*width_o/width_i)))  #x1
        coord_out[1] = int(max(float(0), (coord_in[1]*height_o/height_i)))   #y1
        coord_out[2] = int(max(float(0), (coord_in[2]*width_o/width_i))) #x2
        coord_out[3] = int(max(float(0), (coord_in[3]*height_o/height_i))) #y2
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
        xyxy = reshape_coordinates_bb(bb_predictions[i].coordinates, 416, 416, cols, rows) #for a different image size
        bb_predictions_reshaped[i].set_bb(bb_predictions[i].id, bb_predictions[i].label, bb_predictions[i].confidence, xyxy)

    for i in range(bb_number):  #reshaped detections on image
            label = str(bb_predictions_reshaped[i].label) + " " + str(round(bb_predictions_reshaped[i].confidence, 2))
            plot_one_box(bb_predictions_reshaped[i].coordinates_reshaped, I_backtorgb, label=label, color=colors[bb_predictions[i].id], line_thickness=1) 

    return bb_predictions_reshaped, I_backtorgb

#Get list of active taxels per bounding box on the image, create an array of the taxel center
def bb_active_taxel (bb_number, S, bb_predictions_reshaped, TIB, skin_faces):
    taxel_predictions = np.empty((bb_number,), dtype = object)
    taxel_predictions_info = np.empty((bb_number,), dtype = object)
    face_centers = np.empty((bb_number,), dtype = object)
    for n in range(bb_number):
        faces_predictions = []
        info = []
        face_index_previous = 0
        for i in range(bb_predictions_reshaped[n].coordinates_reshaped[0], bb_predictions_reshaped[n].coordinates_reshaped[2]):
            for j in range(bb_predictions_reshaped[n].coordinates_reshaped[1], bb_predictions_reshaped[n].coordinates_reshaped[3]):
                face_index = TIB.get_pixel_face_index( i,  j)
                if face_index == (-1) or face_index >= 1218: #checking that taxels are withing boundss
                    break
                if face_index == face_index_previous: #not recounting the same taxel over and over
                    break
                faces_predictions.append(skin_faces[face_index][0])
                faces_predictions.append(skin_faces[face_index][1])
                faces_predictions.append(skin_faces[face_index][2])

                face_index_previous = face_index

        if faces_predictions == []: #check for a bug
            taxel_predictions[n] = []
        else:
            taxel_predictions[n] = set(faces_predictions) #set rmoves duplicates

        #Position fo the faces
        faces_positions = []
        for k in range(0, len(faces_predictions), 3):
            sum_first_two = list(map(add, S.taxels[faces_predictions[k]].get_taxel_position(),S.taxels[faces_predictions[k+1]].get_taxel_position())) 
            sum_of_three = list(map(add, sum_first_two, S.taxels[faces_predictions[k+2]].get_taxel_position())) 
            face_position = [x / 3 for x in sum_of_three]
            faces_positions.append(face_position)
        face_centers[n] = faces_positions

        #Prediction info
        info.append(bb_predictions_reshaped[n].label)
        info.append(bb_predictions_reshaped[n].confidence)
        info.append(len(set(faces_predictions)))
        taxel_predictions_info[n] = info #this is the name, conf and # active taxels per prediction
    return taxel_predictions, taxel_predictions_info, face_centers

#Get taxel responses for all bounding boxes 
def taxel_responses(bb_number, S, taxel_predictions, taxel_predictions_info):
    total_taxel_responses = np.empty((bb_number,), dtype = object)
    total_taxels_position = np.empty((bb_number,), dtype = object)
    average_responses = np.empty((bb_number,), dtype = object)
    bb_centroid = np.empty((bb_number,), dtype = object)

    #get total responses
    for n in range(bb_number):
        taxel_response = [] #empty array for the responses of a single bounding box
        taxels_position = [] #empty array for the idus of a single bounding box
        for i in taxel_predictions[n]:
            if S.taxels[i].get_taxel_response() != 0: 
                taxel_response.append(S.taxels[i].get_taxel_response()) 
                taxels_position.append(S.taxels[i].get_taxel_position()) 
        if taxel_response == [] or taxels_position == []:
            total_taxel_responses[n] = []
            total_taxels_position[n] = []
        else: 
            total_taxel_responses[n] = taxel_response
            total_taxels_position[n] = taxels_position
    
    #get average responses including taxels with 0 response
    for n in range(bb_number):
        if len(total_taxels_position[n]) != 0:
            average_response = sum(total_taxel_responses[n])/taxel_predictions_info[n][2]
            average_responses[n] = average_response
            print("Average Response of", taxel_predictions_info[n][0], "is", average_responses[n])
        else:
            average_responses[n] = 0.0
    
     
    #get average response position
    for n in range(bb_number):
        average_position = [0.0,0.0,0.0]
        if len(total_taxels_position[n]) != 0:
            for i in range(len(total_taxels_position[n])):
                average_position[0] = average_position[0] + total_taxels_position[n][i][0]
                average_position[1] = average_position[1] + total_taxels_position[n][i][1]
                average_position[2] = average_position[2] + total_taxels_position[n][i][2]
            average_position[0] = average_position[0] /len(total_taxels_position[n])
            average_position[1] = average_position[1] /len(total_taxels_position[n])
            average_position[2] = average_position[2] /len(total_taxels_position[n])

            bb_centroid[n] = average_position
            #print("Position of Centroid", taxel_predictions_info[n][0], "is", bb_centroid[n])
        else:
            bb_centroid[n] = []
    
    return total_taxel_responses, average_responses, total_taxels_position, bb_centroid


def total_responses_visualization(bb_number, V, total_taxels_position, taxel_predictions_info, color_dict):
    if bb_number !=0:
        for n in range(bb_number):
            contact_color = color_dict[taxel_predictions_info[n][0]]
            for i in range(len(total_taxels_position[n])):
                V.add_marker(1+(600*n)+i,total_taxels_position[n][i], contact_color)


def average_responses_visualization(bb_number, V, bb_centroid, taxel_predictions_info, color_dict ):
    if bb_number !=0:
        for n in range(bb_number):
            contact_color = color_dict[taxel_predictions_info[n][0]]
            V.add_marker((n*30+2*n),bb_centroid[n], contact_color)

def total_faces_visualization(bb_number, V, face_centers, taxel_predictions_info, color_dict):
    if bb_number !=0:
        for n in range(bb_number):
            contact_color = color_dict[taxel_predictions_info[n][0]]
            for i in range(len(face_centers[n])):
                V.add_marker(1+(400*n)+ 40 + i,face_centers[n][i], contact_color)


