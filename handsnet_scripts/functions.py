import numpy as np
from cv2 import cvtColor, resize, imshow, waitKey, INTER_AREA, COLOR_GRAY2RGB
from operator import add 
from time import time
from math import sqrt, pow

def contact(hand_contact):
    print(hand_contact)
    if hand_contact[0][0]>0.8:
        print('Hand')
    elif hand_contact[0][1]>0.8:
        print('Non-Hand')
    else:
        print('Not Recognized')

def image_prediction(I, HandsNet):
    backtorgb = cvtColor(I,COLOR_GRAY2RGB)  #converting from grayscale to rgb     
    I_resized = resize(backtorgb, (68,100), interpolation=INTER_AREA)
    input_arr_hand = np.array([I_resized])  # Convert single image to a batch.
    hand_contact = HandsNet.predict(input_arr_hand)
    I_toshow = resize(backtorgb, (500,500), interpolation=INTER_AREA)
    
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
        distance = sqrt((pow((total_taxel_position[i][0] - 0),2) + pow((total_taxel_position[i][1] - 0),2) + pow((total_taxel_position[i][2] - 0),2)) )
        r.append(distance)
    return r

def get_distance_from_axis(total_taxel_position, total_taxel_response):
    r_axis = []
    for i in range(len(total_taxel_response)): #int(np.size(total_taxel_positions)/3)
        distance = sqrt(pow(total_taxel_position[i][2],2) + pow(total_taxel_position[i][1],2))
        r_axis.append(distance)

    return r_axis

#2D AND 3D CENTROID OF BB
def get_centroid(S,T, total_taxels_2d_position, taxel_coords):
    centroid2d = [0.0,0.0,0.0]
    centroid3d = [0.0,0.0,0.0]
    if len(total_taxels_2d_position) != 0:
        for i, position in enumerate(total_taxels_2d_position):
            centroid2d[0] = centroid2d[0] + position[0]
            centroid2d[1] = centroid2d[1] + position[1]
            centroid2d[2] = centroid2d[2] + position[2] #z should be 0 anyway
        centroid2d[0] = centroid2d[0] / len(total_taxels_2d_position)
        centroid2d[1] = centroid2d[1] / len(total_taxels_2d_position)
        centroid2d[2] = centroid2d[2] / len(total_taxels_2d_position)
        #used for projecting a 2D centroid on the tactile map to a 3D point
        centroid3d = back_project_centroid(S, T, centroid2d, taxel_coords) 
    else:
        centroid2d = []
        centroid3d = [] 
    return centroid2d, centroid3d

#General case vectorW forces
def find_vector_forces(total_taxel_response, total_taxel_normal):
    total_vector_force = []
    integral_force = [0.0,0.0,0.0]
    if len(total_taxel_response) != 0:
        for i, response in enumerate(total_taxel_response): #add the 0 here for negative vals
            vector_force = [0.0, 0.0, 0.0]
            vector_force[0] = -response * total_taxel_normal[i][0] #x
            vector_force[1] = -response * total_taxel_normal[i][1] #y
            vector_force[2] = -response * total_taxel_normal[i][2] #z

            integral_force[0] = integral_force[0] + vector_force[0]
            integral_force[1] = integral_force[1] + vector_force[1]
            integral_force[2] = integral_force[2] + vector_force[2]
        total_vector_force.append(vector_force)

    return total_vector_force, integral_force

#General case vector moments
def find_vector_moments(total_vector_force, centroid3d, total_taxel_3d_position):
    total_vector_moment = []
    integral_moment = [0.0,0.0,0.0]
    moment = [0.0,0.0,0.0]
    if len(total_vector_force) != 0:
        for i, vec_force in enumerate(total_vector_force):
            distance = np.subtract(total_taxel_3d_position[i], centroid3d) #between centroid and taxel position
            moment = np.cross(distance, vec_force) #vector produce between distance and vector force on the taxel
            integral_moment = np.add(integral_moment, moment) # summing it up all the moments
            total_vector_moment.append(moment) #append the single moment in a whole vector
        #TO BE MODIFIED
        integral_moment[0] = integral_moment[0] / len(total_vector_force) #total moments divided by their number to get the average
        integral_moment[1] = integral_moment[1] / len(total_vector_force)
        integral_moment[2] = integral_moment[2] / len(total_vector_force)
    return total_vector_moment, integral_moment

#BACK PROJECT A POINT FROM 2D MAP TO 3D
def back_project_centroid(S, T, bb_centroid2d, taxel_coords):
    #initializing
    centroid_3d, P, B, C = [0.0,0.0,0.0], [0.0,0.0], [0.0,0.0], [0.0,0.0]
    #finding the indexes of the 3 closest points, with numpy is very fast
    difference = np.subtract(taxel_coords, bb_centroid2d)
    diff_pow2 = np.square(difference)
    diff_sum = np.sum(diff_pow2, axis=1)
    diff_squared = np.square(diff_sum)
    minimum_indexes = diff_squared.argsort()[:3]

    a,  b, c = T.taxels[minimum_indexes[0]].get_taxel_position(), T.taxels[minimum_indexes[1]].get_taxel_position(), T.taxels[minimum_indexes[2]].get_taxel_position()

    #Compute the cofficents of the convex combination
    P[0], P[1], B[0], B[1], C[0], C[1] = bb_centroid2d[0]-a[0], bb_centroid2d[1]-a[1], b[0]-a[0], b[1]-a[1], c[0]-a[0], c[1]-a[1]
        
    d = B[0]*C[1] - C[0]*B[1]
    wa, wb, wc = (P[0]*(B[1]-C[1]) + P[1]*(C[0]-B[0]) + B[0]*C[1] - C[0]*B[1]) / d, (P[0]*C[1] - P[1]*C[0]) / d, (P[1]*B[0] - P[0]*B[1]) / d

    v1, v2, v3 = S.taxels[minimum_indexes[0]].get_taxel_position(), S.taxels[minimum_indexes[1]].get_taxel_position(), S.taxels[minimum_indexes[2]].get_taxel_position()

    centroid_3d[0], centroid_3d[1], centroid_3d[2] = wa*v1[0] + wb*v2[0] + wc*v3[0], wa*v1[1] + wb*v2[1] + wc*v3[1], wa*v1[2] + wb*v2[2] + wc*v3[2]
    
    return centroid_3d

def open_files():
    force_file = open(("../data_files/total_force.txt"),"w+") 
    moment_file = open(("../data_files/total_moment.txt"),"w+") 
    return force_file, moment_file

def write_forces_and_moments(integral_force, integral_moment, force_file, moment_file):
    s_force =  "".join([str(round(time(),6))," ",str(integral_force[0])," ",str(integral_force[1])," ",str(integral_force[2]),"\n"])
    s_moment = "".join([str(round(time(),6))," ",str(integral_moment[0])," ",str(integral_moment[1])," ",str(integral_moment[2]),"\n"])
    print(s_force, s_moment)
    force_file.write(s_force)
    moment_file.write(s_moment)
    force_file.flush()
    moment_file.flush()
    return