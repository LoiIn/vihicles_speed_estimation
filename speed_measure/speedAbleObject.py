import numpy as np
import core.utils as utils

_classes = [
    {'name': 'motorbike', 'height': 1.09},
    {'name': 'car', 'height': 1.45},
    {'name': 'bicycle', 'height': 0.7},
]

class SpeedAbleObject:
    def __init__(self, objectID, bbox, timestamp, class_name):
        self.objectID = objectID
        # self.centroid = [centroid]
        
        self.bbox = utils.format_center_point(bbox)
        for _class in _classes:
            if _class.name == class_name:    
                self.object_height = _class.height
                break

        self.timestamp = timestamp

        