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
        self.points = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "F")]
        self.timestamp = {"A": 0, "B": 0, "C":0, "D": 0, "E": 0, "F": 0}
        self.speeds = {"AB": None, "BC": None, "CD": None, "DE": None, "EF": None}
        self.position = {"A": None, "B": None, "C":None, "D": None, "E": None, "F": None}
        self.lastPoint = False
        self.estimated = False
        self.speed = None

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
                
                elif self.timestamp["E"] == 0:
                    if centroid[0] > self.flags["E"]:
                        self.timestamp["E"] = frame_num
                        self.position["E"] = centroid[:2]
                
                elif self.timestamp["F"] == 0:
                    if centroid[0] > self.flags["F"]:
                        if centroid[0] < (self.flags["F"] + self.flags["A"]):
                            self.position["F"] = centroid[:2]
                        else:
                            self.position["F"] = [self.flags["F"] + 2*self.flags["A"], centroid[1]]
                        self.timestamp["F"] = frame_num
                        self.lastPoint = True

            elif self.direction < 0:
                if self.timestamp["A"] == 0:
                    if centroid[0] < self.flags["F"]:
                        self.timestamp["A"] = frame_num
                        self.position["A"] = centroid[:2]
                    
                elif self.timestamp["B"] == 0:
                    if centroid[0] < self.flags["E"]:
                        self.timestamp["B"] = frame_num
                        self.position["B"] = centroid[:2]
                
                elif self.timestamp["C"] == 0:
                    if centroid[0] < self.flags["D"]:
                        self.timestamp["C"] = frame_num
                        self.position["C"] = centroid[:2]

                elif self.timestamp["D"] == 0:
                    if centroid[0] < self.flags["C"]:
                        self.timestamp["D"] = frame_num
                        self.position["D"] = centroid[:2]
                
                elif self.timestamp["E"] == 0:
                    if centroid[0] < self.flags["B"]:
                        self.timestamp["E"] = frame_num
                        self.position["E"] = centroid[:2]
                
                elif self.timestamp["F"] == 0:
                    if centroid[0] < self.flags["A"]:
                        if centroid[0] > 0:
                            self.position["F"] = centroid[:2]
                        else:
                            self.position["F"] = [0-self.flags["A"],centroid[1]]
                        self.timestamp["F"] = frame_num
                        self.lastPoint = True
        
    def calculate_speed(self, estimatedSpeeds):
        avera_speed = np.average(estimatedSpeeds)
        
        self.speed = round(avera_speed, 1)