import pyrobotskinlib as rsl 
import numpy as np
from time import time
import tensorflow as tf
from cv2 import cvtColor, resize, imshow, waitKey, INTER_AREA, COLOR_GRAY2RGB
from  time import sleep 
from functions import *
            

#MAIN
if __name__ == '__main__':
    #LIMITING THE GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 5GB  of memory on the first GPU
        try:
            config = tf.config.experimental.set_memory_growth(gpus[0], True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        except RuntimeError as e:
            print(e) 
    
    #LOAD 4 SKIN HANDLES and ACQUIRE DATA FROM SHMEM
    S1 = rsl.RobotSkin("../calibration_files/collaborate_handle_1.json")
    u1 = rsl.SkinUpdaterFromShMem(S1)
    S2 = rsl.RobotSkin("../calibration_files/collaborate_handle_2.json")
    u2 = rsl.SkinUpdaterFromShMem(S2)
    S3 = rsl.RobotSkin("../calibration_files/collaborate_handle_3.json")
    u3 = rsl.SkinUpdaterFromShMem(S3)
    S4 = rsl.RobotSkin("../calibration_files/collaborate_handle_4.json")
    u4 = rsl.SkinUpdaterFromShMem(S4)

    #CREATE 2D TACTILE MAPS
    T1 = rsl.TactileMap(S1,0) 
    T2 = rsl.TactileMap(S2,0) 
    T3 = rsl.TactileMap(S3,0) 
    T4 = rsl.TactileMap(S4,0) 

    #BUILD TACTILE IMAGES
    TIB1 = rsl.TactileImageBuilder(T1)
    TIB1.build_tactile_image()
    TIB2 = rsl.TactileImageBuilder(T2)
    TIB2.build_tactile_image()
    TIB3 = rsl.TactileImageBuilder(T3)
    TIB3.build_tactile_image()
    TIB4 = rsl.TactileImageBuilder(T4)
    TIB4.build_tactile_image()

    #START SKIN UPDATER
    u1.start_robot_skin_updater()
    u2.start_robot_skin_updater()
    u3.start_robot_skin_updater()
    u4.start_robot_skin_updater()

    #Get rows and cols
    rows = TIB1.get_rows()
    cols = TIB1.get_cols()
    

    #LOAD HANDSNET
    HandsNet = tf.keras.models.load_model('../data/HandsNet_Finetuned.h5')
    HandsNet.trainable=False

    while 1:
        #ACQUIRE DATA
        u1.make_this_thread_wait_for_new_data()
        u2.make_this_thread_wait_for_new_data()
        u3.make_this_thread_wait_for_new_data()
        u4.make_this_thread_wait_for_new_data()

        #IMAGE PROCESSING AND PREDICTION
        I1 = np.array(TIB1.get_tactile_image(),np.uint8) #get the images
        I1 = I1.reshape([rows,cols]) #reshape it into a 2d array
        I2 = np.array(TIB2.get_tactile_image(),np.uint8) #get the images
        I2 = I2.reshape([rows,cols]) #reshape it into a 2d array
        I3 = np.array(TIB3.get_tactile_image(),np.uint8) #get the images
        I3 = np.reshape([rows,cols]) #reshape it into a 2d array
        I4 = np.array(TIB4.get_tactile_image(),np.uint8) #get the images
        I4 = I4.reshape([rows,cols]) #reshape it into a 2d array


        #PREDICTION and RESHAPED IMAGE
        I_toshow1, hand_contact1 = image_prediction(I1, HandsNet)
        I_toshow2, hand_contact2 = image_prediction(I2, HandsNet)
        I_toshow3, hand_contact3 = image_prediction(I3, HandsNet)
        I_toshow4, hand_contact4 = image_prediction(I4, HandsNet)

        #PRINT RESULT
        print("HANDLE1:")
        contact(hand_contact1)
        print("HANDLE2:")
        contact(hand_contact2)
        print("HANDLE3:")
        contact(hand_contact3)
        print("HANDLE4:")
        contact(hand_contact4)
       
        #SHOW IMAGE
        imshow('Tactile Image',I_toshow1)
        imshow('Tactile Image',I_toshow2)
        imshow('Tactile Image',I_toshow3)
        imshow('Tactile Image',I_toshow4)

        waitKey(1)

