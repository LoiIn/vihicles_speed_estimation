import numpy as np
from .utils import format_center_point

class SpeedAbleObject:
    def __init__(self, objectID, bbox, timestamp, class_name):
        self.objectID = objectID
        # self.centroid = [centroid]
        
        self.bbox = format_center_point(bbox)
        
        if class_name == "motorbike":
            self.object_height = 1.09
        elif class_name == "car":
            self.object_height = 1.47
        else:
            self.object_height = 0.7

        self.timestamp = timestamp

        