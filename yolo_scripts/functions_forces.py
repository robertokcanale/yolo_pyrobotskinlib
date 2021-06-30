import numpy as np
from math import sqrt, pow
#General case vector forces, just considering the response equal to the force, A=1
def find_vector_forces(total_taxel_responses, bb_normal_taxel):  
    total_vector_forces = [(-np.multiply(taxel_response,bb_normal_taxel[i])) for i,taxel_response in enumerate(total_taxel_responses) if len(total_taxel_responses) != 0]
    return total_vector_forces

#Case for each BB, total taxel normals is not negated, so to find internal forces i need to put a - in find vector forces
def find_total_bb_forces(bb_number, total_taxel_responses, total_taxel_normals):     
    total_bb_forces = [(find_vector_forces(total_taxel_responses[n], total_taxel_normals[n])) for n in range(bb_number)]
    return total_bb_forces

#Base model, divide by the number of activated taxels
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

#Consider the Area coefficient. P=F*A. area is the total area a force acts on. 
#we consider aall
def get_bb_integral_pressure(bb_number, total_bb_forces, area):  
    bb_integral_pressure = []
    for n in range(bb_number):
        integral_force = [0.0,0.0,0.0]
        if len(total_bb_forces[n]) != 0:
            for i, bb_force in enumerate(total_bb_forces[n]):
                integral_force[0] = integral_force[0] + bb_force[0]
                integral_force[1] = integral_force[1] + bb_force[1]
                integral_force[2] = integral_force[2] + bb_force[2]
            #HERE STUFF MIGHT BE DIFFERENT AND WE MIGHT CONSIDER SOME AREA
            #STILL TO MODIFY/EVALUATE, for now it is simply divided by the total number of forces
            integral_force[0] = integral_force[0] / (len(total_bb_forces[n]) * area)
            integral_force[1] = integral_force[1] / (len(total_bb_forces[n]) * area)
            integral_force[2] = integral_force[2] / (len(total_bb_forces[n]) * area)
        bb_integral_pressure.append(integral_force)
    return bb_integral_pressure

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
            #HERE STUFF MIGHT BE DIFFERENT AND WE MIGHT CONSIDER SOME AREA
            #STILL TO MODIFY/EVALUATE, for now it is simply divided by the total number of forces
            moment_sum[0] = moment_sum[0] / len(total_bb_forces[n])
            moment_sum[1] = moment_sum[1] / len(total_bb_forces[n])
            moment_sum[2] = moment_sum[2] / len(total_bb_forces[n])

        bb_integral_moment.append(moment_sum)
        total_bb_moment[n] = total_bb_moment_list
    return bb_integral_moment, total_bb_moment

def get_bb_moment_pressures(bb_number, bb_integral_pressure, bb_centroid3d, total_taxels_3D_position, area):
    bb_integral_moment, total_bb_moment = [], np.empty((bb_number,), dtype = object) #sum of all moments in a bb, #all moments in a bb
    for n in range(bb_number):
        total_bb_moment_list, moment, moment_sum =  [], [0.0,0.0,0.0], [0.0,0.0,0.0],
        if len(bb_integral_pressure[n]) != 0:
            for i,force_i in enumerate(bb_integral_pressure[n]):
                distance = np.subtract(total_taxels_3D_position[n][i], bb_centroid3d[n])
                moment = np.cross(distance, (force_i/area))
                moment_sum = np.add(moment_sum, moment)
                total_bb_moment_list.append(moment)
        bb_integral_moment.append(moment_sum)
        total_bb_moment[n] = total_bb_moment_list
    return bb_integral_moment, total_bb_moment

def get_bb_moment_from_center(bb_number, total_bb_forces, total_taxels_3D_position):
    bb_integral_moment, total_bb_moment = [], np.empty((bb_number,), dtype = object) #sum of all moments in a bb, #all moments in a bb
    for n in range(bb_number):
        total_bb_moment_list, moment, moment_sum =  [], [0.0,0.0,0.0], [0.0,0.0,0.0],
        if len(total_bb_forces[n]) != 0:
            for i,force_i in enumerate(total_bb_forces[n]):
                distance = np.subtract(total_taxels_3D_position[n][i], [0,0,0])
                moment = np.cross(distance, force_i)
                moment_sum = np.add(moment_sum, moment)
                total_bb_moment_list.append(moment)

            #HERE STUFF MIGHT BE DIFFERENT AND WE MIGHT CONSIDER SOME AREA
            #STILL TO MODIFY/EVALUATE, for now it is simply divided by the total number of forces
            moment_sum[0] = moment_sum[0] / len(total_bb_forces[n])
            moment_sum[1] = moment_sum[1] / len(total_bb_forces[n])
            moment_sum[2] = moment_sum[2] / len(total_bb_forces[n])

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


    
        
