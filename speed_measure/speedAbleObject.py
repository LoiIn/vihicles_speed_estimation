import numpy as np

class SpeedAbleObject:
    def __init__(self, objectID, centroid, color, flags):
        self.objectID = objectID
        self.centroids = [centroid[:2]]
        self.bbox = centroid
        self.scale = centroid[2] / 1.908
        self.color = color
        self.direction = None
        self.flags = flags
        self.points = [("A", "B"), ("B", "C"), ("C", "D")]
        self.timestamp = {"A": 0, "B": 0, "C":0, "D": 0}
        self.speeds = {"AB": 0.0, "BC": 0.0, "CD": 0.0}
        self.position = {"A": None, "B": None, "C":None, "D": None}
        self.lastPoint = False
        self.estimated = False
        self.speed = None
        
        # if class_name == "motorbike":
        #     self.object_height = 1.
        # elif class_name == "car":
        #     self.object_height = 1.47
        # else:
        #     self.object_height = 0.7
    

    def updateSpeedObject(self, centroid, frame_num):
        self.bbox = centroid
        self.scale = centroid[2] / 1.908
        if not self.estimated:
            self.centroids.append(centroid[:2])
        
            if self.direction is None:
                self.direction = 0
            if self.direction == 0 and (len(self.centroids) > 2):
                x = [c[0] for c in self.centroids]
                self.direction = centroid[0] - np.mean(x)
            
            if self.direction > 0:
                if self.timestamp["A"] == 0:
                    if centroid[0] > self.flags["A"]:
                        self.timestamp["A"] = frame_num
                        self.position["A"] = centroid[:2]
                    
                elif self.timestamp["B"] == 0:
                    if centroid[0] > self.flags["B"]:
                        self.timestamp["B"] = frame_num
                        self.position["B"] = centroid[:2]
                
                elif self.timestamp["C"] == 0:
                    if centroid[0] > self.flags["C"]:
                        self.timestamp["C"] = frame_num
                        self.position["C"] = centroid[:2]

                elif self.timestamp["D"] == 0:
                    if centroid[0] > self.flags["D"]:
                        self.timestamp["D"] = frame_num
                        self.position["D"] = centroid[:2]
                        self.lastPoint = True
            
            elif self.direction < 0:
                if self.timestamp["A"] == 0:
                    if centroid[0] < self.flags["D"]:
                        self.timestamp["A"] = frame_num
                        self.position["A"] = centroid[:2]
                    
                elif self.timestamp["B"] == 0:
                    if centroid[0] < self.flags["C"]:
                        self.timestamp["B"] = frame_num
                        self.position["B"] = centroid[:2]
                
                elif self.timestamp["C"] == 0:
                    if centroid[0] < self.flags["B"]:
                        self.timestamp["C"] = frame_num
                        self.position["C"] = centroid[:2]

                elif self.timestamp["D"] == 0:
                    if centroid[0] < self.flags["A"]:
                        self.timestamp["D"] = frame_num
                        self.position["D"] = centroid[:2]
                        self.lastPoint = True
        
    def calculate_speed(self, estimatedSpeeds, asRt):
        avera_speed = np.average(estimatedSpeeds)
        
        _asRt = asRt if asRt is not None else self.scale 
        self.speed = round(avera_speed * 3.6 / _asRt, 1)