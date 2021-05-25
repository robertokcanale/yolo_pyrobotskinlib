import pyrobotskinlib as rsl 
import numpy as np
import argparse
import time
import torch
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
    S = rsl.RobotSkin("../calibration_files/collaborate_handle_1_ale.json")
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
    rows = TIB.get_rows() #y
    cols = TIB.get_cols() #x
    print(rows, cols)
    #GET SKIN INFO
    skin_faces = S.get_faces()
    number_of_faces = len(skin_faces)
    taxel_ids = S.get_taxel_ids()
    #INITIALIZE YOLOV5
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    weights = '../data/best_l_6classes_finetune.pt' 
    imgsz = 416
    conf_thres = 0.5
    iou_thres = 0.5
    device = '0' #'0' or CPU if needed
    colors = [[0,0,255], [0,255,0], [255,0,0], [100,100,100], [0,50,150], [75,150,0] ] #6 classes
               #['index',   'middle',   palm',    'pinkie',    'ring',      'thumb']
    color_dict = dict({ 'palm':[0,0,255, 1], 'thumb':[0,255,0,1], 'index':[255,0,0,1], 'middle':[0,255,255,1], 'ring':[170,80,0,1], 'pinkie':[180,180,180,1]  })

    #GPU
    set_logging()
    device = select_device(device)
    #MODEL
    model = attempt_load(weights, map_location=device)
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names


    while True:
        #GET NEW DATA
        u.make_this_thread_wait_for_new_data()
        #CREATE TACTILE IMAGE AND PROCESS IMAGE (for recorded data)
        I = np.array(TIB.get_tactile_image(),np.uint8) #get the image 
        I = I.reshape([rows,cols]) #reshape it into a 2d array
        I_backtorgb = cv2.cvtColor(I,cv2.COLOR_GRAY2RGB)  #converting from grayscale to rgb 
        I_resized = cv2.resize(I_backtorgb, (416,416), interpolation=cv2.INTER_AREA) #resize it for yolo
        I_transposed = np.transpose(I_resized, (2, 0, 1)) #transposing the image for processing

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
        
        #ACTIVATED TAXELS FOR EACH BB
        taxel_predictions, pixel_positions, taxel_predictions_info = bb_active_taxel(bb_number, T, bb_predictions_reshaped, TIB, skin_faces)
        #GET RESPONSE OF ACTIVATED TAXELS
        #total_taxel_responses, average_responses, total_taxels_position, bb_centroid = taxel_responses(bb_number, S, taxel_predictions, taxel_predictions_info)

        #print("Taxel Predictions:", taxel_predictions) #here I have all the taxel indexes of my predictions, however i need to clean them 
        #print("Taxel Predictions Info:", taxel_predictions_info) #here I have all the taxel indexes of my predictions, however i need to clean them 
        #print("Taxel Responses:", total_taxel_responses) 
        #print("Taxel Positions:", total_taxels_position)
        #print("Average Taxel Responses:", average_responses) 
        #print("Average Taxel Positions:", bb_centroid)
        
        #VISUALIZE MARKERS
        total_responses_visualization(bb_number, V, pixel_positions, taxel_predictions_info, color_dict )
        #total_faces_visualization(bb_number, V, face_centers, taxel_predictions_info, color_dict)
        #average_responses_visualization(bb_number, V, bb_centroid, taxel_predictions_info, color_dict )

        cv2.imshow('Tactile Image  Original',I_backtorgb)
        cv2.waitKey(1)

        time.sleep(1)
        V.remove_markers()
