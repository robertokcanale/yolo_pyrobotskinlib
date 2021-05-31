import numpy as np
from utils.plots import plot_one_box

#For Raw Detections
class BoundingBox:
    label = ""
    confidence = 0.0
    coordinates = np.zeros(4, dtype=np.float32)
    id = 0
    def __init__(self):
        pass
    def set_bb(self, id, label, confidence, coordinates):
        self.id = id
        self.label = label
        self.confidence = confidence
        self.coordinates = coordinates

#For Reshaped Detections
class BoundingBoxReshaped:
    label = ""
    confidence = 0.0
    coordinates_reshaped = np.zeros(4, dtype=np.int32)
    id = 0
    def __init__(self):
        pass
    def set_bb(self, id, label, confidence, coordinates_reshaped):
        self.id = id
        self.label = label
        self.confidence = confidence
        self.coordinates_reshaped = coordinates_reshaped

#Reshape bb coordinates for images of different size
def reshape_coordinates_bb (coord_in, width_i, height_i, width_o, height_o):
    coord_out = np.zeros(len(coord_in), dtype= np.int32)
    #x1y1, x2y2
    coord_out[0], coord_out[1], coord_out[2], coord_out[3] = (coord_in[0]*width_o/width_i) ,(coord_in[1]*height_o/height_i), (coord_in[2]*width_o/width_i), (coord_in[3]*height_o/height_i) 
    return coord_out

#Create a Bounding Box object with the predictions
def bounding_box_predictions(det, bb_number, names):
    bb_predictions = [BoundingBox() for i in range(bb_number)]
    for i in range(bb_number): #scan the prediction matrix DET/PRED (they are the same)
        coordinates=[round(det[i][0].item(),3),round(det[i][1].item(),3),round(det[i][2].item(),3),round(det[i][3].item(),3)] 
        confidence, obj_class_id, obj_class = round(det[i][4].item(),5), int(det[i][5].item()) ,names[int(det[i][5].item())]

        bb_predictions[i].set_bb(obj_class_id, obj_class, confidence, coordinates)
    return bb_predictions

#Create a Reshaped Bounding Box object with the predictions and image
def bounding_box_predictions_reshaped(bb_predictions, bb_number, I_backtorgb, colors, rows, cols):
    bb_predictions_reshaped = [BoundingBoxReshaped() for i in range(bb_number)]
    for i in range(bb_number): 
        xyxy = reshape_coordinates_bb(bb_predictions[i].coordinates, 416, 416, cols, rows) #for a different image size
        bb_predictions_reshaped[i].set_bb(bb_predictions[i].id, bb_predictions[i].label, bb_predictions[i].confidence, xyxy)
        label = "".join([str(bb_predictions_reshaped[i].label)," ",str(round(bb_predictions_reshaped[i].confidence, 2))])
        plot_one_box(bb_predictions_reshaped[i].coordinates_reshaped, I_backtorgb, label=label, color=colors[bb_predictions[i].id], line_thickness=1) 

    return bb_predictions_reshaped, I_backtorgb

