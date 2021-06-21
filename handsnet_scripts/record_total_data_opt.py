import pyrobotskinlib as rsl 
import numpy as np
from time import time
#import tensorflow as tf
from cv2 import cvtColor, resize, imshow, waitKey, INTER_AREA, COLOR_GRAY2RGB
from  time import sleep 
from functions_optimized import *
            
#MAIN
if __name__ == '__main__':
    #LOAD TACTILE IMAGE & SKIN PROPERTIES
    S = rsl.RobotSkin("../calibration_files/collaborate_handle_1_ale.json")
    u = rsl.SkinUpdaterFromShMem(S)
    T = rsl.TactileMap(S,0) # from here i do 0,1,2,3 i can create differenti images from different patches
    TIB = rsl.TactileImageBuilder(T)
    TIB.build_tactile_image()
    u.start_robot_skin_updater()
    rows = TIB.get_rows()
    cols = TIB.get_cols()
    
    skin_faces = S.get_faces()
    number_of_faces = len(skin_faces)
    taxel_ids = S.get_taxel_ids()
    number_of_ids = len(taxel_ids)
    taxel_coords = S.get_taxels_position()


    force_file, moment_file = open_files()

    while 1:
        #t0= time()
        #ACQUIRE DATA
        u.make_this_thread_wait_for_new_data()

        #Get Total Taxels Responses and Positions
        total_taxel_response, total_taxel_3d_position, total_taxel_normal, total_taxel_2d_position, active_taxels_length= get_taxel_data(S,T, number_of_ids)
        #Get Centroid
        centroid2d, centroid3d = get_centroid(S,T, total_taxel_2d_position, taxel_coords, active_taxels_length)
        #get Forces  and integral
        total_vector_force, integral_force = find_vector_forces(total_taxel_response, total_taxel_normal,active_taxels_length)
        #Get Moments and integral
        total_vector_moment, integral_moment = find_vector_moments(total_vector_force, centroid3d, total_taxel_3d_position, active_taxels_length)
        #write on file
        write_forces_and_moments(integral_force, integral_moment, force_file, moment_file)
        #samples acquired every= 0.01s
        #print("Elapsed: ",time()- t0)

