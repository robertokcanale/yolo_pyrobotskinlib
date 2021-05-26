import pyrobotskinlib as rsl 
import numpy as np
import os
import argparse
import torch
#import tensorflow as tf
import cv2
from PIL import Image
import time
from functions import *
            
image_size = (100, 68)

#MAIN
if __name__ == '__main__':
    #LIMITING THE GPU
    """gpus = tf.config.experimental.list_physical_devices('GPU')
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
    T = rsl.TactileMap(S,0)# from here i do 0,1,2,3 i can create differenti images from different patches
    TIB = rsl.TactileImageBuilder(T)
    TIB.build_tactile_image()
    u.start_robot_skin_updater()
    rows = TIB.get_rows()
    cols = TIB.get_cols()
    
    skin_faces = S.get_faces()
    number_of_faces = len(skin_faces)
    taxel_ids = S.get_taxel_ids()
    number_of_ids = len(taxel_ids)
    #LOAD HANDSNET
    #HandsNet = tf.keras.models.load_model('../data/HandsNet_Finetuned.h5')
    #HandsNet.trainable=False

    while True:
        #ACQUIRE DATA
        u.make_this_thread_wait_for_new_data()
        #IMAGE PROCESSING AND PREDICTION
        I = np.array(TIB.get_tactile_image(),np.uint8) #get the image 
        I = I.reshape([rows,cols]) #reshape it into a 2d array
        #I_toshow, hand_contact = image_prediction(I, HandsNet)
        #contact(hand_contact)
        cv2.imshow('Tactile Image',I)
        cv2.waitKey(1)

        #Get Total Taxels Responses and Positions
        total_taxel_response, total_taxel_positions, total_taxel_normals = get_taxel_data(S, number_of_ids)

        print(total_taxel_normals)

        #Active taxels distance from [0,0,0]
        r = get_distance_from_center(total_taxel_positions,total_taxel_response) 
        #Active taxels distance from [x,0,0], the cylinder axis
        r_axis = get_distance_from_axis(total_taxel_positions, total_taxel_response)
