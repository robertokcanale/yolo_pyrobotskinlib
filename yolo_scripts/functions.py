import numpy as np
import argparse
import torch
import cv2
import random
import math
from operator import add 
import time
from functions_taxel_data import *
from functions_bb import *
from functions_forces import *
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def total_responses_visualization(bb_number, V, pixel_positions, taxel_predictions_info, color_dict):
    if bb_number !=0:
        counter = 0
        for n in range(bb_number):
            contact_color = color_dict[taxel_predictions_info[n][0]]
            for i in range(len(pixel_positions[n])):
                a = random.randint(0,50)
                if a == 4:
                    V.add_marker(counter,pixel_positions[n][i], contact_color)
                counter += 1

def average_responses_visualization(bb_number, V, bb_centroid, taxel_predictions_info, color_dict ):
        counter = 0
        for n in range(bb_number):
            if bb_centroid[n] != []:
                contact_color = color_dict[taxel_predictions_info[n][0]]
                V.add_marker(counter,bb_centroid[n], contact_color)
                counter += 1

def open_files():
    palm_file = open(("../data_files/palm.txt"),"w+") 
    thumb_file = open(("../data_files/thumb.txt"),"w+") 
    index_file = open(("../data_files/index.txt"),"w+") 
    middle_file = open(("../data_files/middle.txt"),"w+") 
    ring_file = open(("../data_files/ring.txt"),"w+") 
    pinkie_file = open(("../data_files/pinkie.txt"),"w+") 
    
    return palm_file,thumb_file,index_file, middle_file, ring_file, pinkie_file

def write_responses(bb_number, taxel_predictions_info, average_responses, palm_file,thumb_file,index_file, middle_file, ring_file, pinkie_file):
    s_palm = str(round(time.time(),5)) + " " + str(0.0) + "\n"
    s_thumb = str(round(time.time(),5)) + " " + str(0.0) + "\n"
    s_index = str(round(time.time(),5)) + " " + str(0.0) + "\n"
    s_middle = str(round(time.time(),5)) + " " + str(0.0) + "\n"
    s_ring = str(round(time.time(),5)) + " " + str(0.0) + "\n"
    s_pinkie = str(round(time.time(),5)) + " " + str(0.0) + "\n"
    #initialize strings to write
    for n in range(bb_number):
        if taxel_predictions_info[n][0] == "palm":
            s_palm = str(round(time.time(),5)) + " " + str(average_responses[n]) + "\n"
        if taxel_predictions_info[n][0] == "thumb":
            s_thumb = str(round(time.time(),5)) + " " + str(average_responses[n]) + "\n"
        if taxel_predictions_info[n][0] == "index":
            s_index = str(round(time.time(),5)) + " " + str(average_responses[n]) + "\n"
        if taxel_predictions_info[n][0] == "middle":
            s_middle = str(round(time.time(),5)) + " " + str(average_responses[n]) + "\n"
        if taxel_predictions_info[n][0] == "ring":
            s_ring = str(round(time.time(),5)) + " " + str(average_responses[n]) + "\n"
        if taxel_predictions_info[n][0] == "pinkie":
            s_pinkie = str(round(time.time(),5)) + " " + str(average_responses[n]) + "\n"
            
    palm_file.write(s_palm)
    thumb_file.write(s_thumb)
    index_file.write(s_index)
    middle_file.write(s_middle)
    ring_file.write(s_ring)
    pinkie_file.write(s_pinkie)