import pyrobotskinlib as rsl 
import numpy as np
import argparse
import torch
import cv2
import time
from PIL import Image
from  functions import *
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


#MAIN
if __name__ == '__main__':
    
    #LOAD TACTILE IMAGE
    S = rsl.RobotSkin("../calibration_files/collaborate_handle_1_ale.json")
    u = rsl.SkinUpdaterFromFile(S, "../data/hands_test_3.txt")
    T = rsl.TactileMap(S,0)
    TIB = rsl.TactileImageBuilder(T)
    TIB.build_tactile_image()
    u.start_robot_skin_updater()
    rows = TIB.get_rows()
    cols = TIB.get_cols()
    skin_faces = S.get_faces()
    taxel_ids = S.get_taxel_ids()
    #print("Skin_faces= ", skin_faces, "Taxel_ids= ",taxel_ids)
    #INITIALIZE YOLOV5
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    weights = '../data/best_s_6classes_finetune.pt' 
    imgsz = 416
    conf_thres = 0.5
    iou_thres = 0.5
    device = '0' #or CPU if needed
    colors = [[0,0,255], [0,255,0], [255,0,0], [100,100,100], [0,50,150], [75,150,0] ] #6 classes
    #GPU
    set_logging()
    device = select_device(device)
    #MODEL
    model = attempt_load(weights, map_location=device)
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names


    while True:
        #CREATE TACTILE IMAGE AND PROCESS IMAGE (for recorded data)
        u.make_this_thread_wait_for_new_data()
        I = np.array(TIB.get_tactile_image(),np.uint8) #get the image 
        I = I.reshape([rows,cols]) #reshape it into a 2d array
        I_backtorgb = cv2.cvtColor(I,cv2.COLOR_GRAY2RGB)  #converting from grayscale to rgb 
        I_resized = cv2.resize(I_backtorgb, (416,416), interpolation=cv2.INTER_AREA) #resize it for yolo
        erode_kernel = np.ones((2, 2), np.uint8) #erode the image
        I_erode = cv2.erode(I_resized, erode_kernel) 
        I_gaussfilter = cv2.blur(I_erode,(3,3),0)   #apply gaussian filtering  
        I_transposed = np.transpose(I_gaussfilter, (2, 0, 1)) #transposing the image for processing

        #YOLO AND DATA PREPROCESSING
        img = torch.from_numpy(I_transposed).to(device)
        img = img.float()  
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img, augment=opt.augment)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=opt.classes,agnostic=opt.agnostic_nms)
        
        for i,det in enumerate(pred):  # detections per image
            s = ''
            s += '%gx%g ' % img.shape[2:]  #string for printing
            if len(det):
                for c in det[:, -1].unique(): #print results in s I only need it if i want to print the stuff
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)} "  # add to string
                #Printing on the Image
                for *xyxy, conf, cls in det: 
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, I_resized, label=label, color=colors[int(cls)], line_thickness=1) #I THINK THIS XYXY IS WHAT I NEED
            #GENERATE MESSAGE
            print(f'Preditction:{s}.')
        
        #GET BOUNDING BOXES FROM PIXELS
        bb_number = int(len(det)) # set the number of predictions 
        bb_predictions = bounding_box_predictions(det, bb_number, names)
        
        #RESHAPE BOUNDING BOXES
        bb_predictions_reshaped, I_backtorgb = bounding_box_predictions_reshaped(bb_predictions, bb_number, I_backtorgb, colors, rows, cols)
        activated_faces = np.zeros((bb_number,3), dtype= np.int32)
        for n in range(bb_number):
            for i in range(bb_predictions_reshaped[n].coordinates_reshaped[0], bb_predictions_reshaped[n].coordinates_reshaped[2]):
                for j in range(bb_predictions_reshaped[n].coordinates_reshaped[1], bb_predictions_reshaped[n].coordinates_reshaped[3]):
                    #map_position = TIB.get_pixel_position_on_map( i,  j)
                    face_index = TIB.get_pixel_face_index( i,  j)            
                    activated_faces[n][i] = skin_faces[face_index]
                    
                    
        
        #I_resized = cv2.resize(I_resized, (rows,cols), interpolation=cv2.INTER_AREA) #resize it for yolo, ale dice di non fare il resize
        im_to_show = cv2.resize(I_resized, (500, 500), interpolation = cv2.INTER_AREA)
        cv2.imshow('Tactile Image',im_to_show)
        cv2.waitKey(1)

        cv2.imshow('Tactile Image  Original',I_backtorgb)
        cv2.waitKey(1)


        #print(  str(u.get_timestamp_in_sec()) + " | " + str(S.taxels[0].get_taxel_response()) )