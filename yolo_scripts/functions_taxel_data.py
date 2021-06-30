import numpy as np

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

            #taxel predictions is the taxel IDs
            taxel_predictions[n] = set(faces_predictions) #set rmoves duplicates
            pixel_positions[n] = pixel_position
        #Prediction info
        info.append(bb_predictions_reshaped[n].label)
        info.append(bb_predictions_reshaped[n].confidence)
        info.append(len(set(faces_predictions)))
        taxel_predictions_info[n] = info #this is the name, conf and # active taxels per prediction
    return taxel_predictions, pixel_positions, taxel_predictions_info

def bb_active_taxel_optimized(bb_number, T, bb_predictions_reshaped, TIB, skin_faces):
    taxel_predictions,taxel_predictions_info = np.empty((bb_number,), dtype = object), np.empty((bb_number,), dtype = object)
    for n in range(bb_number):
        faces_predictions,  info = [], []
        cols = range(bb_predictions_reshaped[n].coordinates_reshaped[0], bb_predictions_reshaped[n].coordinates_reshaped[2])
        rows = range(bb_predictions_reshaped[n].coordinates_reshaped[1], bb_predictions_reshaped[n].coordinates_reshaped[3])

        for i in cols:
            for j in rows:
                face_index = TIB.get_pixel_face_index( i,  j)
                if face_index == (-1) or face_index >= 1218: #checking that taxels are withing boundss
                    break

                #Taxel_IDs_from_faces
                faces_predictions.append(skin_faces[face_index][0])
                faces_predictions.append(skin_faces[face_index][1])
                faces_predictions.append(skin_faces[face_index][2])

            #taxel predictions is the taxel IDs
            taxel_predictions[n] = set(faces_predictions) #set rmoves duplicates
        #Prediction info
        info.append(bb_predictions_reshaped[n].label)
        info.append(bb_predictions_reshaped[n].confidence)
        info.append(len(set(faces_predictions)))
        taxel_predictions_info[n] = info #this is the name, conf and # active taxels per prediction
    return taxel_predictions, taxel_predictions_info

#Get total data for the all the taxels and bounding boxes
#NORMALS ARE ALREADY NORMALIZED
def get_total_data(bb_number, S, T, taxel_predictions):
    if bb_number != 0:
        total_taxel_responses, total_taxels_3D_position, total_taxel_normals, total_taxels_2D_position = np.empty((bb_number,), dtype = object), np.empty((bb_number), dtype = object), np.empty((bb_number,), dtype = object), np.empty((bb_number,), dtype = object)
        #for each BB, get the data
        for n in range(bb_number):
            resp, norm, threedpos, twodpos = [], [], [], []
            for i in taxel_predictions[n]:
                if S.taxels[i].get_taxel_response() > 1000: #filtering out noise
                    resp.append(S.taxels[i].get_taxel_response())
                    norm.append(S.taxels[i].get_taxel_normal())
                    threedpos.append(S.taxels[i].get_taxel_position())
                    twodpos.append(T.taxels[i].get_taxel_position())                    
            total_taxel_responses[n] = resp
            total_taxel_normals[n] = norm
            total_taxels_3D_position[n] = threedpos
            total_taxels_2D_position[n] = twodpos  
        return total_taxel_responses, total_taxels_3D_position, total_taxel_normals , total_taxels_2D_position
    else:
        return [0], [[0,0,0]], [[0,0,0]], [[0,0,0]]

#AVERAGE RESPONSES including taxels with 0 response
def get_average_response_per_BB(bb_number, total_taxel_responses, taxel_predictions_info):
    average_responses = [(sum(total_taxel_responses[n])/taxel_predictions_info[n][2]) for n in range(bb_number) if (len(total_taxel_responses[n]) != 0)]
    return average_responses

#2D AND 3D CENTROID OF BB
def get_bb_centroids(bb_number,S,T, total_taxels_2D_position, taxel_coords):
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
            bb_centroid3d[n] = back_project_centroid(S, T, bb_centroid2d[n], taxel_coords) 
            if bb_centroid3d[n] == [0,0,0]:
                print("Centroid = 0")
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
def back_project_centroid(S, T, bb_centroid2d, taxel_coords):
    #initializing
    centroid_3d, P, B, C = [0.0,0.0,0.0], [0.0,0.0], [0.0,0.0], [0.0,0.0]
    #finding the indexes of the 3 closest points, with numpy is very fast
    difference = np.subtract(taxel_coords, bb_centroid2d)
    diff_pow2 = np.square(difference)
    diff_sum = np.sum(diff_pow2, axis=1)
    diff_squared = np.square(diff_sum)
    minimum_indexes = diff_squared.argsort()[:3]

    if minimum_indexes[0]==-1 or minimum_indexes[1]==-1 or minimum_indexes[2]==-1:
        v=[0,0,0]
        return v

    a,  b, c = T.taxels[minimum_indexes[0]].get_taxel_position(), T.taxels[minimum_indexes[1]].get_taxel_position(), T.taxels[minimum_indexes[2]].get_taxel_position()
    
    #Compute the cofficents of the convex combination
    P[0], P[1], B[0], B[1], C[0], C[1] = bb_centroid2d[0]-a[0], bb_centroid2d[1]-a[1], b[0]-a[0], b[1]-a[1], c[0]-a[0], c[1]-a[1]
        
    d = B[0]*C[1] - C[0]*B[1]
    wa, wb, wc = (P[0]*(B[1]-C[1]) + P[1]*(C[0]-B[0]) + B[0]*C[1] - C[0]*B[1]) / d, (P[0]*C[1] - P[1]*C[0]) / d, (P[1]*B[0] - P[0]*B[1]) / d

    v1, v2, v3 = S.taxels[minimum_indexes[0]].get_taxel_position(), S.taxels[minimum_indexes[1]].get_taxel_position(), S.taxels[minimum_indexes[2]].get_taxel_position()

    centroid_3d[0], centroid_3d[1], centroid_3d[2] = wa*v1[0] + wb*v2[0] + wc*v3[0], wa*v1[1] + wb*v2[1] + wc*v3[1], wa*v1[2] + wb*v2[2] + wc*v3[2]
    
    return centroid_3d
