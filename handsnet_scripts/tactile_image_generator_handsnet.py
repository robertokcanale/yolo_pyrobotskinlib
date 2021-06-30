import pyrobotskinlib as rsl 
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #WITH GPU IT DOESNT WORK YET IDK WHY
import argparse
import torch
import tensorflow as tf
import cv2
from PIL import Image
from functions_optimized import *
import time

#MAIN
if __name__ == '__main__':
    """  #LIMITING THE GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 5GB  of memory on the first GPU
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 5)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        except RuntimeError as e:
            print(e)
            """
    
    image_size = (100, 68)


    #LOAD TACTILE IMAGE
    S = rsl.RobotSkin("../calibration_files/collaborate_handle_1_ale.json")
    u = rsl.SkinUpdaterFromShMem(S)
    T = rsl.TactileMap(S,0)# from here i do 0,1,2,3 i can create differenti images from different patches
    TIB = rsl.TactileImageBuilder(T)
    TIB.build_tactile_image()
    u.start_robot_skin_updater()
    rows = TIB.get_rows()
    cols = TIB.get_cols()
    #BUILD MESH VISUALIZER
    V = rsl.RobotSkinViewer(500,500,20000)
    V2 = rsl.RobotSkinViewer(500,500,20000)
    #RENDER MESH
    V.render_object(T,"Robot")
    V2.render_object(T,"Robot_Markers")

    #HandsNet = tf.keras.models.load_model('../data/HandsNet_Finetuned.h5')
    #HandsNet.trainable=False
    while True:
        u.make_this_thread_wait_for_new_data()

        I = np.array(TIB.get_tactile_image(),np.uint8) #get the image 
        I = I.reshape([rows,cols]) #reshape it into a 2d array
        #I_toshow, hand_contact = image_prediction(I, HandsNet)
        #contact(hand_contact)

        cv2.imshow('Tactile Image',I)
        cv2.waitKey(1)
        V.add_marker(0,[0.1,0.1,0], [255,0,0,1])
        V.add_marker(1,[0.1,0,0], [0,255,0,1])
        V.add_marker(2,[-0.1,-0.1,0], [0,0,255,1])
        V.add_marker(3,[0,0,0], [0,0,0,1])
        time.sleep(2)
        V.remove_markers()
        #print(  str(u.get_timestamp_in_sec()) + " | " + str(S.taxels[0].get_taxel_response()) )