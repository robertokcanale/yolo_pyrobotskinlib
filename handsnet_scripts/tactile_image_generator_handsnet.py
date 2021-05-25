import pyrobotskinlib as rsl 
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #WITH GPU IT DOESNT WORK YET IDK WHY
import argparse
import torch
import tensorflow as tf
import cv2
from PIL import Image
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

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

    #LOAD HANDSNET
    HandsNet = tf.keras.models.load_model('HandsNet_Models/HandsNet_Finetuned_2.h5')

    HandsNet.trainable=False

    while True:
        u.make_this_thread_wait_for_new_data()

        I = np.array(TIB.get_tactile_image(),np.uint8) #get the image 
        I = I.reshape([rows,cols]) #reshape it into a 2d array
        backtorgb = cv2.cvtColor(I,cv2.COLOR_GRAY2RGB)  #converting from grayscale to rgb     
        I_resized = cv2.resize(backtorgb, (68,100), interpolation=cv2.INTER_AREA)
        input_arr_hand = np.array([I_resized])  # Convert single image to a batch.
        hand_contact = HandsNet.predict(input_arr_hand)
        I_toshow = cv2.resize(backtorgb, (500,500), interpolation=cv2.INTER_AREA)
        cv2.imshow('Tactile Image',I_toshow)
        cv2.waitKey(1)
        print(hand_contact)
        if hand_contact[0][0]>0.8:
            print('Hand')
        elif hand_contact[0][1]>0.8:
            print('Non-Hand')
        else:
            print('Not Recognized')
        
        #print(  str(u.get_timestamp_in_sec()) + " | " + str(S.taxels[0].get_taxel_response()) )