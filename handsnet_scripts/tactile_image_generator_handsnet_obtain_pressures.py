import pyrobotskinlib as rsl 
import numpy as np
from time import time
#import tensorflow as tf
from cv2 import cvtColor, resize, imshow, waitKey, INTER_AREA, COLOR_GRAY2RGB
from  time import sleep 
from functions import *
            

#MAIN
if __name__ == '__main__':
    #LIMITING THE GPU
    """ gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 5GB  of memory on the first GPU
        try:
            config = tf.config.experimental.set_memory_growth(gpus[0], True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        except RuntimeError as e:
            print(e) 
    """
    #LOAD TACTILE IMAGE & SKIN PROPERTIES
    S = rsl.RobotSkin("../calibration_files/collaborate_handle_1_ale.json")
    u = rsl.SkinUpdaterFromShMem(S)
    #u = rsl.SkinUpdaterFromFile(S,"../recorded_data/hand_record.txt")
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
    taxel_coords = [T.taxels[i].get_taxel_position() for i in range(number_of_ids)]
    #LOAD HANDSNET
    #HandsNet = tf.keras.models.load_model('../data/HandsNet_Finetuned.h5')
    #HandsNet.trainable=False

    force_file, moment_file = open_files()

    while 1:
        #ACQUIRE DATA
        t0=time()
        u.make_this_thread_wait_for_new_data()
        #IMAGE PROCESSING AND PREDICTION
        I = np.array(TIB.get_tactile_image(),np.uint8) #get the image 
        I = I.reshape([rows,cols]) #reshape it into a 2d array
        #I_toshow, hand_contact = image_prediction(I, HandsNet)
        #contact(hand_contact)
        im_to_show = resize(I, (500, 500), interpolation = INTER_AREA)
        imshow('Tactile Image',im_to_show)
        waitKey(1)

        #Get Total Taxels Responses and Positions
        total_taxel_response, total_taxel_3d_position, total_taxel_normal, total_taxel_2d_position= get_taxel_data(S,T, number_of_ids)
        centroid2d, centroid3d = get_centroid(S,T, total_taxel_2d_position, taxel_coords)

        #Active taxels distance from [0,0,0]
        #r = get_distance_from_center(total_taxel_positions,total_taxel_response) 
        #Active taxels distance from [x,0,0], the cylinder axis
        #r_axis = get_distance_from_axis(total_taxel_positions, total_taxel_response)
        total_vector_force, integral_force = find_vector_forces(total_taxel_response, total_taxel_normal)
        total_vector_moment, integral_moment = find_vector_moments(total_vector_force, centroid3d, total_taxel_3d_position)
        #total_vector_moment, integral_moment =  find_vector_moments_from_center(total_vector_force, total_taxel_3d_position)
        #print("Total Force", total_vector_force)
        #print("Total Moment", total_vector_moment)

        write_forces_and_moments(integral_force, integral_moment, force_file, moment_file)

        elapsed_time = time()-t0
        #ACQUISITION TIME 0.5s
        sleep(0.5-elapsed_time)
