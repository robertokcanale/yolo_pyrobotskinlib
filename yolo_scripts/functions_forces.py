import numpy as np

#General case vector forces
def find_vector_forces(total_taxel_responses, bb_normal):
    total_vector_forces = []
    if len(total_taxel_responses) != 0:
        for i in range(len(total_taxel_responses)):
            vector_force = [0.0, 0.0, 0.0]  #i think i have to pu a - as the forces are opposite to the taxel response
            vector_force[0] = - total_taxel_responses[i] * bb_normal[i][0] #x 
            vector_force[1] = - total_taxel_responses[i] * bb_normal[i][1] #y
            vector_force[2] = - total_taxel_responses[i] * bb_normal[i][2] #z
            total_vector_forces.append(vector_force)
    
    return total_vector_forces

#Case for each BB
def find_total_bb_forces(bb_number, total_taxel_responses, total_taxel_normals):     
    total_bb_forces = np.empty((bb_number,), dtype = object)
    for n in range(bb_number):
        total_bb_forces[n] = find_vector_forces(total_taxel_responses[n], total_taxel_normals[n])
    return total_bb_forces

def get_bb_integral_force(bb_number, total_bb_forces):  
    bb_integral_force = []
    for n in range(bb_number):
        integral_force = [0.0,0.0,0.0]
        if len(total_bb_forces[n]) != 0:
            for i in range(len(total_bb_forces[n])):
                integral_force[0] = integral_force[0] + total_bb_forces[n][i][0]
                integral_force[1] = integral_force[1] + total_bb_forces[n][i][1]
                integral_force[2] = integral_force[2] + total_bb_forces[n][i][2]
            #HERE STUFF MIGHT BE DIFFERENT AND WE MIGHT CONSIDER SOME AREA
            #STILL TO MODIFY/EVALUATE, for now it is simply divided by the total number of forces
            integral_force[0] = integral_force[0] / len(total_bb_forces[n])
            integral_force[1] = integral_force[1] / len(total_bb_forces[n])
            integral_force[2] = integral_force[2] / len(total_bb_forces[n])
        bb_integral_force.append(integral_force)

    return bb_integral_force   

def get_bb_moment(bb_number, total_bb_forces, bb_centroid3d, total_taxels_3D_position):
    bb_integral_moment = [] #sum of all moments in a bb
    total_bb_moment = np.empty((bb_number,), dtype = object) #all moments in a bb
    for n in range(bb_number):
        total_bb_moment_list = []
        moment =  [0.0,0.0,0.0]
        moment_sum =  [0.0,0.0,0.0]
        if len(total_bb_forces[n]) != 0:
            for i in range(len(total_bb_forces[n])):
                distance = np.subtract(total_taxels_3D_position[n][i], bb_centroid3d[n])
                moment = np.cross(distance, total_bb_forces[n][i])
                moment_sum = np.add(moment_sum, moment)
                total_bb_moment_list.append(moment)
        bb_integral_moment.append(moment_sum)
        total_bb_moment[n] = total_bb_moment_list
    return bb_integral_moment, total_bb_moment

#Taxel Distance From ###
def get_distance_from_center(bb_number, total_taxel_positions, total_taxel_responses):
    bb_taxels_r = np.empty((bb_number,), dtype = object)
    for n in range(bb_number):
        r = []
        for i in range(len(total_taxel_responses[n])): #int(np.size(total_taxel_positions)/3)
            distance = sqrt((pow((total_taxel_positions[n][i][0] - 0),2) + pow((total_taxel_positions[n][i][1] - 0),2) + pow((total_taxel_positions[n][i][2] - 0),2)) )
            r.append(distance)
        bb_taxels_r[n] = r
    return bb_taxels_r

def get_distance_from_axis(bb_number, total_taxel_positions, total_taxel_responses):
    bb_taxels_r_axis = np.empty((bb_number,), dtype = object)
    for n in range(bb_number):
        r_axis = []
        for i in range(len(total_taxel_responses[n])): #int(np.size(total_taxel_positions)/3)
            distance = math.sqrt(pow(total_taxel_positions[n][i][2],2) + pow(total_taxel_positions[n][i][1],2))
            r_axis.append(distance)
        bb_taxels_r_axis[n] = r_axis
    return bb_taxels_r_axis


    
        
