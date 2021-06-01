import numpy as np
from time import time
from math import sqrt, pow
#Get list of active taxels per bounding box on the image, create an array of the taxel center
def bb_active_taxel (bb_number, T, bb_predictions_reshaped, TIB, skin_faces):
    taxel_predictions, pixel_positions,taxel_predictions_info = np.empty((bb_number,), dtype = object), np.empty((bb_number,), dtype = object), np.empty((bb_number,), dtype = object)
    for n in range(bb_number):
        faces_predictions, pixel_position, info = [], [], []
        cols = range(bb_predictions_reshaped[n].coordinates_reshaped[0], bb_predictions_reshaped[n].coordinates_reshaped[2])
        rows = range(bb_predictions_reshaped[n].coordinates_reshaped[1], bb_predictions_reshaped[n].coordinates_reshaped[3])

        for i in cols:
            for j in rows:
                face_index = TIB.get_pixel_face_index( i,  j)
                if face_index == (-1) or face_index >= 1218: #checking that taxels are withing boundss
                    break
                #Pixel_Position
                pos_on_map = TIB.get_pixel_position_on_map(i, j)
                pixel_pos = T.back_project_point(pos_on_map, face_index)
                pixel_position.append(pixel_pos)  

                #Taxel_IDs_from_faces
                faces_predictions.append(skin_faces[face_index][0])
                faces_predictions.append(skin_faces[face_index][1])
                faces_predictions.append(skin_faces[face_index][2])

            taxel_predictions[n] = set(faces_predictions) #set rmoves duplicates
            pixel_positions[n] = pixel_position
        #Prediction info
        info.append(bb_predictions_reshaped[n].label)
        info.append(bb_predictions_reshaped[n].confidence)
        info.append(len(set(faces_predictions)))
        taxel_predictions_info[n] = info #this is the name, conf and # active taxels per prediction
    return taxel_predictions, pixel_positions, taxel_predictions_info

#Get total data for the all the taxels and bounding boxes
def get_total_data(bb_number, S, T, taxel_predictions):

    total_taxel_responses = [[S.taxels[i].get_taxel_response() for i in taxel_predictions[n]] for n in range(bb_number)] 
    total_taxels_3D_position = [[S.taxels[i].get_taxel_position()for i in taxel_predictions[n]] for n in range(bb_number)] 
    total_taxel_normals = [[S.taxels[i].get_taxel_normal() for i in taxel_predictions[n]] for n in range(bb_number)] 
    total_taxels_2D_position = [[T.taxels[i].get_taxel_position() for i in taxel_predictions[n]] for n in range(bb_number)] 

    return total_taxel_responses, total_taxels_3D_position, total_taxel_normals , total_taxels_2D_position

#AVERAGE RESPONSES including taxels with 0 response
def get_average_response_per_BB(bb_number, total_taxel_responses, taxel_predictions_info):
    average_responses = [(sum(total_taxel_responses[n])/taxel_predictions_info[n][2]) for n in range(bb_number) if (len(total_taxel_responses[n]) != 0)]
    return average_responses

#2D AND 3D CENTROID OF BB
def get_bb_centroids(bb_number,S,T, total_taxels_2D_position, number_of_ids):
    bb_centroid2d, bb_centroid3d = np.empty((bb_number,), dtype = object), np.empty((bb_number,), dtype = object)
    for n in range(bb_number):
        average_position = [0.0,0.0,0.0]
        if len(total_taxels_2D_position[n]) != 0:
            for i,val in enumerate(total_taxels_2D_position[n]):
                average_position[0] = average_position[0] + val[0]
                average_position[1] = average_position[1] + val[1]
                average_position[2] = average_position[2] + val[2] #z should be 0 anyway
            average_position[0] = average_position[0] / len(total_taxels_2D_position[n])
            average_position[1] = average_position[1] / len(total_taxels_2D_position[n])
            average_position[2] = average_position[2] / len(total_taxels_2D_position[n])
            bb_centroid2d[n]=average_position
            #used for projecting a 2D centroid on the tactile map to a 3D point
            bb_centroid3d[n] = back_project_centroid(S, T, bb_centroid2d[n], number_of_ids) 
        else:
            bb_centroid2d[n] = []
            bb_centroid3d[n] = []  
    return bb_centroid2d, bb_centroid3d

#BB NORMALS, i put the minus here
def get_bb_average_normals(bb_number,total_taxel_normals):
    bb_normal = np.empty((bb_number,), dtype = object)
    #AVERAGE NORMAL
    for n in range(bb_number):
        average_normal = [0.0,0.0,0.0]
        if len(total_taxel_normals[n]) != 0:
            for i, val in enumerate(total_taxel_normals[n]):
                average_normal[0] = average_normal[0] - val[0] #on the x, it is going to be 0 of course
                average_normal[1] = average_normal[1] - val[1]
                average_normal[2] = average_normal[2] - val[2]
            average_normal[0] = average_normal[0] / len(total_taxel_normals[n])
            average_normal[1] = average_normal[1] / len(total_taxel_normals[n])
            average_normal[2] = average_normal[2] / len(total_taxel_normals[n])

            bb_normal[n] = average_normal
            #print("Position of Centroid", taxel_predictions_info[n][0], "is", bb_centroid[n])
        else:
            bb_normal[n] = []  
    return bb_normal

#BACK PROJECT A POINT FROM 2D MAP TO 3D
def back_project_centroid(S, T, bb_centroid2d, number_of_ids):
    #initializing
    short_dist1, short_dist2, short_dist3, taxel_id1, taxel_id2, taxel_id3  = 10, 10, 10, 0, 0, 0
    centroid_3d, P, B, C = [0.0,0.0,0.0], [0.0,0.0], [0.0,0.0], [0.0,0.0]

    #find the 3 closest taxels
    for i in range(number_of_ids):
        taxel_coords = T.taxels[i].get_taxel_position()
        x, y = taxel_coords[0], taxel_coords[1]
        distance = sqrt( pow(bb_centroid2d[0] - x,2) + pow(bb_centroid2d[1] -y, 2))
        if distance < short_dist1:
            short_dist3, short_dist2, short_dist1 = short_dist2, short_dist1, distance
            taxel_id3, taxel_id2, taxel_id1 = taxel_id2, taxel_id1, i
        elif distance < short_dist2:
            short_dist3, short_dist2 = short_dist2, distance
            taxel_id3, taxel_id2 = taxel_id2, i
        elif distance < short_dist3:
            short_dist3, taxel_id3 = distance, i
    a,  b, c = T.taxels[taxel_id1].get_taxel_position(), T.taxels[taxel_id2].get_taxel_position(), T.taxels[taxel_id3].get_taxel_position()

    #Compute the cofficents of the convex combination
    P[0], P[1], B[0], B[1], C[0], C[1] = bb_centroid2d[0]-a[0], bb_centroid2d[1]-a[1], b[0]-a[0], b[1]-a[1], c[0]-a[0], c[1]-a[1]
        
    d = B[0]*C[1] - C[0]*B[1]
    wa, wb, wc = (P[0]*(B[1]-C[1]) + P[1]*(C[0]-B[0]) + B[0]*C[1] - C[0]*B[1]) / d, (P[0]*C[1] - P[1]*C[0]) / d, (P[1]*B[0] - P[0]*B[1]) / d

    v1, v2, v3 = S.taxels[taxel_id1].get_taxel_position(), S.taxels[taxel_id2].get_taxel_position(), S.taxels[taxel_id3].get_taxel_position()

    centroid_3d[0], centroid_3d[1], centroid_3d[2] = wa*v1[0] + wb*v2[0] + wc*v3[0], wa*v1[1] + wb*v2[1] + wc*v3[1], wa*v1[2] + wb*v2[2] + wc*v3[2]
    
    return centroid_3d
