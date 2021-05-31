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
def get_taxel_data(S, T, number_of_ids):
    total_taxel_response = [] #empty array for the responses 
    total_taxel_3d_position = []
    total_taxel_normal = []
    total_taxels_2d_position = []
    for i in range(number_of_ids):
        if S.taxels[i].get_taxel_response() >= 500:  #to filter out some noise
            total_taxel_response.append(S.taxels[i].get_taxel_response()) 
            total_taxel_3d_position.append(S.taxels[i].get_taxel_position())
            total_taxel_normal.append(S.taxels[i].get_taxel_normal())
            total_taxels_2d_position.append(T.taxels[i].get_taxel_position()) #on the tactile map
                
    return total_taxel_response, total_taxel_3d_position, total_taxel_normal, total_taxels_2d_position

#Taxel Distance From Center
def get_distance_from_center(total_taxel_position, total_taxel_response):
    r = []
    for i in range(len(total_taxel_response)): #int(np.size(total_taxel_positions)/3)
        distance = math.sqrt((pow((total_taxel_position[i][0] - 0),2) + pow((total_taxel_position[i][1] - 0),2) + pow((total_taxel_position[i][2] - 0),2)) )
        r.append(distance)
    return r

def get_distance_from_axis(total_taxel_position, total_taxel_response):
    r_axis = []
    for i in range(len(total_taxel_response)): #int(np.size(total_taxel_positions)/3)
        distance = math.sqrt(pow(total_taxel_position[i][2],2) + pow(total_taxel_position[i][1],2))
        r_axis.append(distance)

    return r_axis

#2D AND 3D CENTROID OF BB
def get_centroid(S,T, total_taxels_2d_position, number_of_ids):
    centroid2d = [0.0,0.0,0.0]
    centroid3d = [0.0,0.0,0.0]
    if len(total_taxels_2d_position) != 0:
        for i in range(len(total_taxels_2d_position)):
            centroid2d[0] = centroid2d[0] + total_taxels_2d_position[i][0]
            centroid2d[1] = centroid2d[1] + total_taxels_2d_position[i][1]
            centroid2d[2] = centroid2d[2] + total_taxels_2d_position[i][2] #z should be 0 anyway
        centroid2d[0] = centroid2d[0] / len(total_taxels_2d_position)
        centroid2d[1] = centroid2d[1] / len(total_taxels_2d_position)
        centroid2d[2] = centroid2d[2] / len(total_taxels_2d_position)
        #used for projecting a 2D centroid on the tactile map to a 3D point
        centroid3d = back_project_centroid(S, T, centroid2d, number_of_ids) 
    else:
        centroid2d = []
        centroid3d = [] 
    return centroid2d, centroid3d

#General case vector forces
def find_vector_forces(total_taxel_response, total_taxel_normal):
    total_vector_force = []
    integral_force = [0.0,0.0,0.0]
    if len(total_taxel_response) != 0:
        for i in range(len(total_taxel_response)):
            vector_force = [0.0, 0.0, 0.0]
            vector_force[0] = total_taxel_response[i] * total_taxel_normal[i][0] #x
            vector_force[1] = total_taxel_response[i] * total_taxel_normal[i][1] #y
            vector_force[2] = total_taxel_response[i] * total_taxel_normal[i][2] #z
            total_vector_force.append(vector_force)

        for i in range(len(total_taxel_response)):
            integral_force[0] = integral_force[0] + total_vector_force[i][0]
            integral_force[1] = integral_force[1] + total_vector_force[i][1]
            integral_force[2] = integral_force[2] + total_vector_force[i][2]
    return total_vector_force, integral_force

#General case vector moments
def find_vector_moments(total_vector_force, centroid3d, total_taxel_3d_position):
    total_vector_moment = []
    integral_moment = [0.0,0.0,0.0]
    moment = [0.0,0.0,0.0]
    if len(total_vector_force) != 0:
        for i in range(len(total_vector_force)):
            distance = np.subtract(total_taxel_3d_position[i], centroid3d) #between centroid and taxel position
            moment = np.cross(distance, total_vector_force[i]) #vector produce between distance and vector force on the taxel
            integral_moment = np.add(integral_moment, moment) # summing it up all the moments
            total_vector_moment.append(moment) #append the single moment in a whole vector
        #TO BE MODIFIED
        integral_moment[0] = integral_moment[0] / len(total_vector_force) #total moments divided by their number to get the average
        integral_moment[1] = integral_moment[1] / len(total_vector_force)
        integral_moment[2] = integral_moment[2] / len(total_vector_force)
    return total_vector_moment, integral_moment

#BACK PROJECT A POINT FROM 2D MAP TO 3D
def back_project_centroid(S, T, centroid2d, number_of_ids):
    #initializing
    short_dist1 = 10
    short_dist2 = 10
    short_dist3 = 10
    taxel_id1 = 0
    taxel_id2 = 0
    taxel_id3 = 0
    centroid_3d = [0.0,0.0,0.0]
    P = [0.0,0.0,0.0]
    B = [0.0,0.0,0.0]
    C = [0.0,0.0,0.0]

    #find the 3 closest taxels
    for i in range(number_of_ids):
        taxel_coords = T.taxels[i].get_taxel_position()
        x = taxel_coords[0]
        y = taxel_coords[1]
        distance = math.sqrt( math.pow(centroid2d[0] - x,2) + math.pow(centroid2d[1] -y, 2))

        if distance < short_dist1:
            short_dist3 = short_dist2
            short_dist2 = short_dist1
            short_dist1 = distance
            taxel_id3 = taxel_id2
            taxel_id2 = taxel_id1
            taxel_id1 = i
        elif distance < short_dist2:
            short_dist3 = short_dist2 
            short_dist2 = distance
            taxel_id3 = taxel_id2
            taxel_id2 = i
        elif distance < short_dist3:
            short_dist3 = distance
            taxel_id3 = i

    a = T.taxels[taxel_id1].get_taxel_position()
    b = T.taxels[taxel_id2].get_taxel_position()
    c = T.taxels[taxel_id3].get_taxel_position()

    #Compute the cofficents of the convex combination
    P[0] = centroid2d[0]-a[0]; P[1] = centroid2d[1]-a[1];
    B[0] = b[0]-a[0]; B[1] = b[1]-a[1];
    C[0] = c[0]-a[0]; C[1] = c[1]-a[1];
        
    d = B[0]*C[1] - C[0]*B[1];
    wa = ( P[0]*(B[1]-C[1]) + P[1]*(C[0]-B[0]) + B[0]*C[1] - C[0]*B[1] ) / d;
    wb = ( P[0]*C[1] - P[1]*C[0] ) / d;
    wc = ( P[1]*B[0] - P[0]*B[1] ) / d;

    v1 = S.taxels[taxel_id1].get_taxel_position()
    v2 = S.taxels[taxel_id2].get_taxel_position()
    v3 = S.taxels[taxel_id3].get_taxel_position()

    centroid_3d[0] = wa*v1[0] + wb*v2[0] + wc*v3[0];
    centroid_3d[1] = wa*v1[1] + wb*v2[1] + wc*v3[1];
    centroid_3d[2] = wa*v1[2] + wb*v2[2] + wc*v3[2]

    return centroid_3d
