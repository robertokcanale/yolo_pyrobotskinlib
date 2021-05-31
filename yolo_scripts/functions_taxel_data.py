import numpy as np
from math import sqrt, pow
#Get list of active taxels per bounding box on the image, create an array of the taxel center
def bb_active_taxel (bb_number, T, bb_predictions_reshaped, TIB, skin_faces):
    taxel_predictions = np.empty((bb_number,), dtype = object)
    pixel_positions = np.empty((bb_number,), dtype = object)
    taxel_predictions_info = np.empty((bb_number,), dtype = object)
    for n in range(bb_number):
        faces_predictions = []
        pixel_position = []
        info = []
        for i in range(bb_predictions_reshaped[n].coordinates_reshaped[0], bb_predictions_reshaped[n].coordinates_reshaped[2]):
            for j in range(bb_predictions_reshaped[n].coordinates_reshaped[1], bb_predictions_reshaped[n].coordinates_reshaped[3]):
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
    total_taxel_responses = np.empty((bb_number,), dtype = object)
    total_taxels_3D_position = np.empty((bb_number,), dtype = object)
    total_taxels_2D_position = np.empty((bb_number,), dtype = object)
    total_taxel_normals = np.empty((bb_number,), dtype = object)
    #TOTAL RESPONSES
    for n in range(bb_number):
        taxel_response = [] #empty array for the responses of a single bounding box
        taxels_3d_position = [] #empty array for the 3d positions of a single bounding box
        taxels_2d_position = [] #empty array for the idus of a single bounding box
        taxel_normal = [] #empty array for the normals single bounding box
        for i in taxel_predictions[n]:
            if S.taxels[i].get_taxel_response() != 0: 
                taxel_response.append(S.taxels[i].get_taxel_response()) 
                taxels_3d_position.append(S.taxels[i].get_taxel_position()) 
                taxel_normal.append(S.taxels[i].get_taxel_normal()) 
                taxels_2d_position.append(T.taxels[i].get_taxel_position()) #on the tactile map
                
        if taxel_response == [] or taxels_3d_position == []:
            total_taxel_responses[n] = []
            total_taxels_3D_position[n] = []
            total_taxels_2D_position[n] = []
            total_taxel_normals[n] = []
        else: 
            total_taxel_responses[n] = taxel_response
            total_taxels_3D_position[n] = taxels_3d_position
            total_taxel_normals[n] = taxel_normal
            total_taxels_2D_position[n] = taxels_2d_position
    return total_taxel_responses, total_taxels_3D_position, total_taxel_normals , total_taxels_2D_position

#AVERAGE RESPONSES including taxels with 0 response
def get_average_response_per_BB(bb_number, total_taxel_responses, taxel_predictions_info):
    """ average_responses = np.empty((bb_number,), dtype = object)
    for n in range(bb_number):
        if len(total_taxel_responses[n]) != 0:
            average_response = sum(total_taxel_responses[n])/taxel_predictions_info[n][2]
            average_responses[n] = average_response
            #print("Average Response of", taxel_predictions_info[n][0], "is", average_responses[n])
        else:
            average_responses[n] = 0.0
    """
    average_responses = [(sum(total_taxel_responses[n])/taxel_predictions_info[n][2]) for n in range(bb_number) if (len(total_taxel_responses[n]) != 0)]

    return average_responses

#2D AND 3D CENTROID OF BB
def get_bb_centroids(bb_number,S,T, total_taxels_2D_position, number_of_ids):
    bb_centroid2d = np.empty((bb_number,), dtype = object)
    bb_centroid3d= np.empty((bb_number,), dtype = object)
    for n in range(bb_number):
        average_position = [0.0,0.0,0.0]
        if len(total_taxels_2D_position[n]) != 0:
            for i in range(len(total_taxels_2D_position[n])):
                average_position[0] = average_position[0] + total_taxels_2D_position[n][i][0]
                average_position[1] = average_position[1] + total_taxels_2D_position[n][i][1]
                average_position[2] = average_position[2] + total_taxels_2D_position[n][i][2] #z should be 0 anyway
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

#BB NORMALS
def get_bb_average_normals(bb_number,total_taxel_normals):
    bb_normal = np.empty((bb_number,), dtype = object)
    #AVERAGE NORMAL
    for n in range(bb_number):
        average_normal = [0.0,0.0,0.0]
        if len(total_taxel_normals[n]) != 0:
            for i in range(len(total_taxel_normals[n])):
                average_normal[0] = average_normal[0] - total_taxel_normals[n][i][0] #on the x, it is going to be 0 of course
                average_normal[1] = average_normal[1] - total_taxel_normals[n][i][1]
                average_normal[2] = average_normal[2] - total_taxel_normals[n][i][2]
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
        distance = sqrt( pow(bb_centroid2d[0] - x,2) + pow(bb_centroid2d[1] -y, 2))

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
    P[0] = bb_centroid2d[0]-a[0]; P[1] = bb_centroid2d[1]-a[1];
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