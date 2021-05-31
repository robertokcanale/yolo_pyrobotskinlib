import pyrobotskinlib as rsl 
import numpy as np
import argparse
import torch
import cv2
from time import sleep
from functions import *
from functions_bb import *
from functions_forces import *
from functions_taxel_data import *
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
    number_of_ids = len(taxel_ids)

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
    #palm_file,thumb_file,index_file, middle_file, ring_file, pinkie_file = open_files("average_responses")
    palm_file_f,thumb_file_f,index_file_f, middle_file_f, ring_file_f, pinkie_file_f = open_files("integral_forces")
    palm_file_m,thumb_file_m,index_file_m, middle_file_m, ring_file_m, pinkie_file_m = open_files("integral_moments")

    while 1:
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
        total_taxel_responses, total_taxels_3D_position, total_taxel_normals, total_taxels_2D_position = get_total_data(bb_number, S, T, taxel_predictions)
        average_responses =get_average_response_per_BB(bb_number, total_taxel_responses, taxel_predictions_info)
        bb_normal = get_bb_average_normals(bb_number,total_taxel_normals )
        bb_centroid2d, bb_centroid3d = get_bb_centroids(bb_number,S,T, total_taxels_2D_position, number_of_ids)
        #bb_taxels_r = get_distance_from_center(bb_number, total_taxels_3D_position, total_taxel_responses)
        #bb_taxels_r_axis = get_distance_from_axis(bb_number, total_taxels_3D_position, total_taxel_responses)
        total_bb_forces = find_total_bb_forces(bb_number, total_taxel_responses, total_taxel_normals)
        bb_integral_force = get_bb_integral_force(bb_number, total_bb_forces)
        bb_integral_moment, total_bb_moment = get_bb_moment(bb_number, total_bb_forces, bb_centroid3d, total_taxels_3D_position)
        
        """ if bb_number !=0:
            print("Taxel Predictions:", np.shape(taxel_predictions)) #here I have all the taxel indexes of my predictions, however i need to clean them 
            print("Taxel Predictions Info:", np.shape(taxel_predictions_info[1])) #here I have all the taxel indexes of my predictions, however i need to clean them 
            print("Taxel Responses:", np.shape(total_taxel_responses[1])) 
            print("Taxel Positions:", np.shape(total_taxels_3D_position[1]))
            print("Average Taxel Responses:", np.shape(average_responses))
            print("Average Taxel Positions:", np.shape(bb_centroid3d[1]))
            print("Total_BB_forces:", np.shape(total_bb_forces[1]))
            print("BB distances from center:", np.shape(bb_taxels_r[1]))
            print("BB distances from axis:", np.shape(bb_taxels_r_axis[1]))
            print("Taxel Responses:", np.shape(total_taxel_responses[0]))
            print("Integral force per BB", bb_integral_force)
            print("Moment per BB", bb_integral_moment)
            print("Total force per BB", total_bb_forces)
            print("Total moment per BB", total_bb_moment)
            print("Integral force per BB", bb_integral_force)
            print("Moment per BB", bb_integral_moment)

        """
        #print("Taxel Positions:", total_taxels_3D_position)
        
        #write_responses(bb_number, taxel_predictions_info, average_responses, palm_file,thumb_file,index_file, middle_file, ring_file, pinkie_file)
        write_forces(bb_number, taxel_predictions_info, bb_integral_force, palm_file_f,thumb_file_f,index_file_f, middle_file_f, ring_file_f, pinkie_file_f)
        #write_moments(bb_number, taxel_predictions_info, bb_integral_moment, palm_file_m,thumb_file_m,index_file_m, middle_file_m, ring_file_m, pinkie_file_m)

        im_to_show = cv2.resize(I_resized, (500, 500), interpolation = cv2.INTER_AREA)
        cv2.imshow('Tactile Image',im_to_show)
        cv2.waitKey(1)


        cv2.imshow('Tactile Image Original',I_backtorgb)
        cv2.waitKey(1)

        sleep(0.001)
    
