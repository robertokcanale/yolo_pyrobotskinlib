import pyrobotskinlib as rsl 
import numpy as np
import argparse
import time
import torch
import random
import cv2
import time
from PIL import Image
from functions import *
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


#MAIN
if __name__ == '__main__':
    
    #LOAD TACTILE SKIN&TACTILE MAP
    S = rsl.RobotSkin("../calibration_files/collaborate_handle_1_old.json")
    u = rsl.SkinUpdaterFromShMem(S)
    T = rsl.TactileMap(S,0)
    #BUILD MESH VISUALIZER
    V = rsl.RobotSkinViewer(500,500,20000)
    V2 = rsl.RobotSkinViewer(500,500,20000)
    #RENDER MESH
    V.render_object(S,"Robot")
    V2.render_object(S,"Robot_Markers")
    #BUILD IMAGE
    TIB = rsl.TactileImageBuilder(T)
    TIB.build_tactile_image()
    u.start_robot_skin_updater()
    rows = TIB.get_rows()
    cols = TIB.get_cols()
    #GET SKIN INFO
    skin_faces = S.get_faces()
    number_of_faces = len(skin_faces)
    taxel_ids = S.get_taxel_ids()
   
    box_to_plot=[10,10, 200,200]

    while True:
        #GET NEW DATA
        u.make_this_thread_wait_for_new_data()
        #CREATE TACTILE IMAGE AND PROCESS IMAGE (for recorded data)
        I = np.array(TIB.get_tactile_image(),np.uint8) #get the image 
        I = I.reshape([rows,cols]) #reshape it into a 2d array
        I_backtorgb = cv2.cvtColor(I,cv2.COLOR_GRAY2RGB)  #converting from grayscale to rgb 
        I_resized = cv2.resize(I_backtorgb, (416,416), interpolation=cv2.INTER_AREA) #resize it for yolo

        box_to_plot_reshaped =reshape_coordinates_bb(box_to_plot, 416, 416, cols, rows)
        I_transposed = np.transpose(I_resized, (2, 0, 1)) #transposing the image for processing
        plot_one_box(box_to_plot, I_resized, label="My Test", color=[0,255,0], line_thickness=1) 
        plot_one_box(box_to_plot_reshaped, I_backtorgb, label="My Test2", color=[0,255,0], line_thickness=1) 

        #testing marker plot
        faces_predictions = []
        face_index_previous = 0
        pixel_pos = [0,0,0]
        counter = 0
        for i in range(box_to_plot_reshaped[0], box_to_plot_reshaped[2]):
            for j in range(box_to_plot_reshaped[1], box_to_plot_reshaped[3]):
                face_index = TIB.get_pixel_face_index( i,  j)
                pos_on_map = TIB.get_pixel_position_on_map(i, j)

                if face_index == (-1) or face_index >= 1218: #checking that taxels are withing boundss
                    break
                pixel_pos = T.back_project_point(pos_on_map, face_index)
                a = random.randint(0,100)
                if a == 4:
                    V.add_marker(counter,pixel_pos, [0,255,0,1])
                counter += 1


        cv2.imshow('Tactile Image',I_resized)
        cv2.waitKey(1)

        cv2.imshow('Tactile Image  Original', I_backtorgb)
        cv2.waitKey(1)

        time.sleep(10)
        V.remove_markers()
