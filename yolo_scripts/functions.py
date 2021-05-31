from random import randint
from operator import add 
from time import time

def total_responses_visualization(bb_number, V, pixel_positions, taxel_predictions_info, color_dict):
    if bb_number !=0:
        counter = 0
        for n in range(bb_number):
            contact_color = color_dict[taxel_predictions_info[n][0]]
            for i in range(len(pixel_positions[n])):
                a = randint(0,200)
                if a == 4:
                    V.add_marker(counter,pixel_positions[n][i], contact_color)
                counter += 1

def average_responses_visualization(bb_number, V, bb_centroid, taxel_predictions_info, color_dict ):
        counter = 0
        for n in range(bb_number):
            if bb_centroid[n] != []:
                contact_color = color_dict[taxel_predictions_info[n][0]]
                V.add_marker(counter,bb_centroid[n], contact_color)
                counter += 1

def open_files(name):
    palm_file = open(("../data_files/palm_" + name + ".txt"),"w+") 
    thumb_file = open(("../data_files/thumb_" + name + ".txt"),"w+") 
    index_file = open(("../data_files/index_" + name + ".txt"),"w+") 
    middle_file = open(("../data_files/middle_" + name + ".txt"),"w+") 
    ring_file = open(("../data_files/ring_" + name + ".txt"),"w+") 
    pinkie_file = open(("../data_files/pinkie_" + name + ".txt"),"w+") 
    
    return palm_file,thumb_file,index_file, middle_file, ring_file, pinkie_file

def write_responses(bb_number, taxel_predictions_info, average_responses, palm_file,thumb_file,index_file, middle_file, ring_file, pinkie_file):
    s_palm = "".join([str(round(time(),5))," ",str(0.0),"\n"])
    s_thumb = "".join([str(round(time(),5))," ",str(0.0),"\n"])
    s_index = "".join([str(round(time(),5))," ",str(0.0),"\n"])
    s_middle = "".join([str(round(time(),5))," ",str(0.0),"\n"])
    s_ring = "".join([str(round(time(),5))," ",str(0.0),"\n"])
    s_pinkie = "".join([str(round(time(),5))," ",str(0.0),"\n"])
    #initialize strings to write
    for n in range(bb_number):
        if taxel_predictions_info[n][0] == "palm":
            s_palm = "".join([str(round(time(),5))," ",str(average_responses[n]),"\n"])
        if taxel_predictions_info[n][0] == "thumb":
            s_thumb = "".join([str(round(time(),5))," ",str(average_responses[n]),"\n"])
        if taxel_predictions_info[n][0] == "index":
            s_index = "".join([str(round(time(),5))," ",str(average_responses[n]),"\n"])
        if taxel_predictions_info[n][0] == "middle":
            s_middle = "".join([str(round(time(),5))," ",str(average_responses[n]),"\n"])
        if taxel_predictions_info[n][0] == "ring":
            s_ring = "".join([str(round(time(),5))," ",str(average_responses[n]),"\n"])
        if taxel_predictions_info[n][0] == "pinkie":
            s_pinkie = "".join([str(round(time(),5))," ",str(average_responses[n]),"\n"])
            
    palm_file.write(s_palm)
    thumb_file.write(s_thumb)
    index_file.write(s_index)
    middle_file.write(s_middle)
    ring_file.write(s_ring)
    pinkie_file.write(s_pinkie)

def write_forces(bb_number, taxel_predictions_info, bb_integral_force, palm_file,thumb_file,index_file, middle_file, ring_file, pinkie_file):
    s_palm =  "".join([str(round(time(),5))," ",str(0.0)," ",str(0.0)," ",str(0.0),"\n"])
    s_thumb = "".join([str(round(time(),5))," ",str(0.0)," ",str(0.0)," ",str(0.0),"\n"])
    s_index = "".join([str(round(time(),5))," ",str(0.0)," ",str(0.0)," ",str(0.0),"\n"])
    s_middle = "".join([str(round(time(),5))," ",str(0.0)," ",str(0.0)," ",str(0.0),"\n"])
    s_ring = "".join([str(round(time(),5))," ",str(0.0)," ",str(0.0)," ",str(0.0),"\n"])
    s_pinkie = "".join([str(round(time(),5))," ",str(0.0)," ",str(0.0)," ",str(0.0),"\n"])
    
    #initialize strings to write
    for n in range(bb_number):
        if taxel_predictions_info[n][0] == "palm":
            s_palm = "".join([str(round(time(),5))," ",str(bb_integral_force[n][0])," ",str(bb_integral_force[n][1])," ",str(bb_integral_force[n][2]),"\n"])
        if taxel_predictions_info[n][0] == "thumb":
            s_thumb = "".join([str(round(time(),5))," ",str(bb_integral_force[n][0])," ",str(bb_integral_force[n][1])," ",str(bb_integral_force[n][2]),"\n"])
        if taxel_predictions_info[n][0] == "index":
            s_index = "".join([str(round(time(),5))," ",str(bb_integral_force[n][0])," ",str(bb_integral_force[n][1])," ",str(bb_integral_force[n][2]),"\n"])
        if taxel_predictions_info[n][0] == "middle":
            s_middle = "".join([str(round(time(),5))," ",str(bb_integral_force[n][0])," ",str(bb_integral_force[n][1])," ",str(bb_integral_force[n][2]),"\n"])
        if taxel_predictions_info[n][0] == "ring":
            s_ring = "".join([str(round(time(),5))," ",str(bb_integral_force[n][0])," ",str(bb_integral_force[n][1])," ",str(bb_integral_force[n][2]),"\n"])
        if taxel_predictions_info[n][0] == "pinkie":
            s_pinkie = "".join([str(round(time(),5))," ",str(bb_integral_force[n][0])," ",str(bb_integral_force[n][1])," ",str(bb_integral_force[n][2]),"\n"])
    
    palm_file.write(s_palm)
    thumb_file.write(s_thumb)
    index_file.write(s_index)
    middle_file.write(s_middle)
    ring_file.write(s_ring)
    pinkie_file.write(s_pinkie)    
    
    return

def write_moments(bb_number, taxel_predictions_info, bb_integral_moment, palm_file,thumb_file,index_file, middle_file, ring_file, pinkie_file):
    s_palm =  "".join([str(round(time(),5))," ",str(0.0)," ",str(0.0)," ",str(0.0),"\n"])
    s_thumb = "".join([str(round(time(),5))," ",str(0.0)," ",str(0.0)," ",str(0.0),"\n"])
    s_index = "".join([str(round(time(),5))," ",str(0.0)," ",str(0.0)," ",str(0.0),"\n"])
    s_middle = "".join([str(round(time(),5))," ",str(0.0)," ",str(0.0)," ",str(0.0),"\n"])
    s_ring = "".join([str(round(time(),5))," ",str(0.0)," ",str(0.0)," ",str(0.0),"\n"])
    s_pinkie = "".join([str(round(time(),5))," ",str(0.0)," ",str(0.0)," ",str(0.0),"\n"])
    
    #initialize strings to write
    for n in range(bb_number):
        if taxel_predictions_info[n][0] == "palm":
            s_palm = "".join([str(round(time(),5))," ",str(bb_integral_moment[n][0])," ",str(bb_integral_moment[n][1])," ",str(bb_integral_moment[n][2]),"\n"])
        if taxel_predictions_info[n][0] == "thumb":
            s_thumb = "".join([str(round(time(),5))," ",str(bb_integral_moment[n][0])," ",str(bb_integral_moment[n][1])," ",str(bb_integral_moment[n][2]),"\n"])
        if taxel_predictions_info[n][0] == "index":
            s_index = "".join([str(round(time(),5))," ",str(bb_integral_moment[n][0])," ",str(bb_integral_moment[n][1])," ",str(bb_integral_moment[n][2]),"\n"])
        if taxel_predictions_info[n][0] == "middle":
            s_middle = "".join([str(round(time(),5))," ",str(bb_integral_moment[n][0])," ",str(bb_integral_moment[n][1])," ",str(bb_integral_moment[n][2]),"\n"])
        if taxel_predictions_info[n][0] == "ring":
            s_ring = "".join([str(round(time(),5))," ",str(bb_integral_moment[n][0])," ",str(bb_integral_moment[n][1])," ",str(bb_integral_moment[n][2]),"\n"])
        if taxel_predictions_info[n][0] == "pinkie":
            s_pinkie = "".join([str(round(time(),5))," ",str(bb_integral_moment[n][0])," ",str(bb_integral_moment[n][1])," ",str(bb_integral_moment[n][2]),"\n"])
    
    palm_file.write(s_palm)
    thumb_file.write(s_thumb)
    index_file.write(s_index)
    middle_file.write(s_middle)
    ring_file.write(s_ring)
    pinkie_file.write(s_pinkie)    
    
    return

    return

