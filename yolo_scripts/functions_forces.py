import numpy as np
from math import sqrt, pow
#General case vector forces
def find_vector_forces(total_taxel_responses, bb_normal):  
    
    total_vector_forces = [(-np.multiply(taxel_response,bb_normal[i])) for i,taxel_response in enumerate(total_taxel_responses) if len(total_taxel_responses) != 0]
    
    return total_vector_forces

#Case for each BB, total taxel normals is not negated, so to find internal forces i need to put a - in find vector forces
def find_total_bb_forces(bb_number, total_taxel_responses, total_taxel_normals):     
    total_bb_forces = [(find_vector_forces(total_taxel_responses[n], total_taxel_normals[n])) for n in range(bb_number)]
    return total_bb_forces

def get_bb_integral_force(bb_number, total_bb_forces):  
    bb_integral_force = []
    for n in range(bb_number):
        integral_force = [0.0,0.0,0.0]
        if len(total_bb_forces[n]) != 0:
            for i, bb_force in enumerate(total_bb_forces[n]):
                integral_force[0] = integral_force[0] + bb_force[0]
                integral_force[1] = integral_force[1] + bb_force[1]
                integral_force[2] = integral_force[2] + bb_force[2]
            #HERE STUFF MIGHT BE DIFFERENT AND WE MIGHT CONSIDER SOME AREA
            #STILL TO MODIFY/EVALUATE, for now it is simply divided by the total number of forces
            integral_force[0] = integral_force[0] / len(total_bb_forces[n])
            integral_force[1] = integral_force[1] / len(total_bb_forces[n])
            integral_force[2] = integral_force[2] / len(total_bb_forces[n])
        bb_integral_force.append(integral_force)

    return bb_integral_force   

def get_bb_moment(bb_number, total_bb_forces, bb_centroid3d, total_taxels_3D_position):
    bb_integral_moment, total_bb_moment = [], np.empty((bb_number,), dtype = object) #sum of all moments in a bb, #all moments in a bb
    for n in range(bb_number):
        total_bb_moment_list, moment, moment_sum =  [], [0.0,0.0,0.0], [0.0,0.0,0.0],
        if len(total_bb_forces[n]) != 0:
            for i,force_i in enumerate(total_bb_forces[n]):
                distance = np.subtract(total_taxels_3D_position[n][i], bb_centroid3d[n])
                moment = np.cross(distance, force_i)
                moment_sum = np.add(moment_sum, moment)
                total_bb_moment_list.append(moment)
        bb_integral_moment.append(moment_sum)
        total_bb_moment[n] = total_bb_moment_list
    return bb_integral_moment, total_bb_moment

#Taxel Distance From ###
def get_distance_from_center(bb_number, total_taxel_positions):
    bb_taxels_r = np.empty((bb_number,), dtype = object)
    for n in range(bb_number):
        r = []
        for i, taxel_pos in enumerate(total_taxel_positions[n]): #int(np.size(total_taxel_positions)/3)
            distance = sqrt((pow((taxel_pos[0] - 0),2) + pow((taxel_pos[1] - 0),2) + pow((taxel_pos[2] - 0),2)) )
            r.append(distance)
        bb_taxels_r[n] = r
    return bb_taxels_r

def get_distance_from_axis(bb_number, total_taxel_positions):
    bb_taxels_r_axis = np.empty((bb_number,), dtype = object)
    for n in range(bb_number):
        r_axis = []
        for i, taxel_pos in enumerate(total_taxel_positions[n]): #int(np.size(total_taxel_positions)/3)
            distance = sqrt(pow(taxel_pos[2],2) + pow(taxel_pos[1],2))
            r_axis.append(distance)
        bb_taxels_r_axis[n] = r_axis
    return bb_taxels_r_axis


    
        
