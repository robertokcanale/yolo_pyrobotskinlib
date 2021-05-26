import numpy as np
import argparse
from tensorflow.python.ops.gen_math_ops import sqrt
import torch
import cv2
import random
from operator import add 
import time
import math

def contact(hand_contact):
    print(hand_contact)
    if hand_contact[0][0]>0.8:
        print('Hand')
    elif hand_contact[0][1]>0.8:
        print('Non-Hand')
    else:
        print('Not Recognized')

def image_prediction(I, HandsNet):
    backtorgb = cv2.cvtColor(I,cv2.COLOR_GRAY2RGB)  #converting from grayscale to rgb     
    I_resized = cv2.resize(backtorgb, (68,100), interpolation=cv2.INTER_AREA)
    input_arr_hand = np.array([I_resized])  # Convert single image to a batch.
    hand_contact = HandsNet.predict(input_arr_hand)
    I_toshow = cv2.resize(backtorgb, (500,500), interpolation=cv2.INTER_AREA)
    
    return I_toshow, hand_contact

#Total Taxel Predictions
def get_taxel_data(S, number_of_ids):
    total_taxel_response = [] #empty array for the responses 
    total_taxel_positions = []
    total_taxel_normals = []
    for i in range(number_of_ids):
        if S.taxels[i].get_taxel_response() >= 500:  #to filter out some noise
            total_taxel_response.append(S.taxels[i].get_taxel_response()) 
            total_taxel_positions.append(S.taxels[i].get_taxel_position())
            total_taxel_normals.append(S.taxels[i].get_taxel_normal())
    return total_taxel_response, total_taxel_positions, total_taxel_normals

#Taxel Distance From Center
def get_distance_from_center(total_taxel_positions, total_taxel_response):
    r = []
    for i in range(len(total_taxel_response)): #int(np.size(total_taxel_positions)/3)
        distance = math.sqrt((pow((total_taxel_positions[i][0] - 0),2) + pow((total_taxel_positions[i][1] - 0),2) + pow((total_taxel_positions[i][2] - 0),2)) )
        r.append(distance)
    return r


def get_distance_from_axis(total_taxel_positions, total_taxel_response):
    r_axis = []
    for i in range(len(total_taxel_response)): #int(np.size(total_taxel_positions)/3)
        distance = math.sqrt(pow(total_taxel_positions[i][2],2) + pow(total_taxel_positions[i][1],2))
        r_axis.append(distance)

    return r_axis


