import numpy as np
from .utils import format_center_point

class SpeedAbleObject:
    def __init__(self, objectID, centroids, color, flags):
        self.objectID = objectID
        self.centroids = centroids
        self.bbox = centroids
        self.scale = centroids[0] / 1.8
        self.color = color

        self.flags = flags
        self.points = [("A", "B"), ("B", "C"), ("C", "D")]
        self.timestamp = {"A": 0, "B": 0, "C":0, "D": 0}
        self.position = {"A": None, "B": None, "C":None, "D": None}
        self.lastPoint = False

        self.speed = None
        
        # if class_name == "motorbike":
        #     self.object_height = 1.
        # elif class_name == "car":
        #     self.object_height = 1.47
        # else:
        #     self.object_height = 0.7
    

    def updateSpeedObject(self, bbox, frame_num):
        centroids = format_center_point(bbox)
        
        if self.timestamp["A"] == 0:
            if centroids[0] > self.flags["A"]:
                self.timestamp["A"] = frame_num
                self.position["A"] = centroids[:2]
            
        elif self.timestamp["B"] == 0:
            if centroids[0] > self.flags["B"]:
                self.timestamp["B"] = frame_num
                self.position["B"] = centroids[:2]
        
        elif self.timestamp["C"] == 0:
            if centroids[0] > self.flags["C"]:
                self.timestamp["C"] = frame_num
                self.position["C"] = centroids[:2]

        elif self.timestamp["D"] == 0:
            if centroids[0] > self.flags["D"]:
                self.timestamp["D"] = frame_num
                self.position["D"] = centroids[:2]
                self.lastPoint = True
        
    def calculate_speed(self, estimatedSpeeds):
        avera_speed = np.average(estimatedSpeeds)
        
        self.speed = round(avera_speed * 3.6 / self.scale, 1)