import numpy as np

class SpeedAbleObject:
    def __init__(self, objectID, centroid, color, flags, _points):
        self.objectID = objectID
        self.centroids = [centroid[:2]]
        self.bbox = centroid
        self.scale = centroid[2] / 1.908
        self.color = color
        self.direction = None
        self.flags = flags
        # self.points = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "F"), ("F", "G"), ("G", "H")]
        self.points = self.initPoints(_points)
        # self.timestamp = {"A": 0, "B": 0, "C":0, "D": 0, "E": 0, "F": 0, "G": 0, "H": 0}
        self.timestamp = self.initTimestamp(_points + 1)
        # self.speeds = {"AB": None, "BC": None, "CD": None, "DE": None, "EF": None, "FG": None, "GH": None}
        self.speeds = self.initSpeeds(_points)
        self.position = self.initPosition(_points + 1)
        self.lastPoint = False
        self.estimated = False
        self.speed = None
        self.logged = False    
        self.truthPoints = _points

    def initPoints(self, _points):
        points = []
        for x in range(1, _points):
            points.append((str(x), str(x + 1)))
        return points
    
    def initTimestamp(self, _points):
        timestamps = {}
        for x in range(1, _points):
            timestamps[str(x)] = 0
        return timestamps

    def initSpeeds(self, _points):
        speeds = {}
        for x in range(1, _points):
            speeds[str(x) + str(x+1)] = None
        return speeds

    def initPosition(self, _points):
        poses = {}
        for x in range(1, _points):
            poses[str(x)] = None
        return poses

    def update(self, centroid, frame_num):
        self.bbox = centroid
        self.scale = centroid[2] / 1.908
        if not self.estimated:
            self.centroids.append(centroid[:2])
        
            if self.direction is None:
                self.direction = 0
            if self.direction == 0 and (len(self.centroids) > 2):
                vertical = [c[0] for c in self.centroids]
                self.direction = centroid[0] - np.mean(vertical)
            
            if self.direction > 0:
                for x in range(1, self.truthPoints + 1):
                    if self.timestamp[str(x)] == 0:
                        if x % self.truthPoints == 0:
                            if centroid[0] > self.flags[str(x)]:
                                if centroid[0] < (self.flags[str(x)] + self.flags[str(1)]):
                                    self.position[str(x)] = centroid[:2]
                                else:
                                    self.position[str(x)] = [self.flags[str(x)] + 2*self.flags[str(1)], centroid[1]]
                            else:
                                self.position[str(x)] = centroid[:2]
                            self.timestamp[str(x)] = frame_num
                            self.lastPoint = True
                        else:
                            if centroid[0] > self.flags[str(x)]:
                                self.timestamp[str(x)] = frame_num
                                self.position[str(x)] = centroid[:2]

                    # if self.timestamp["A"] == 0:
                    #     if centroid[0] > self.flags["A"]:
                    #         self.timestamp["A"] = frame_num
                    #         self.position["A"] = centroid[:2]
                        
                    # elif self.timestamp["B"] == 0:
                    #     if centroid[0] > self.flags["B"]:
                    #         self.timestamp["B"] = frame_num
                    #         self.position["B"] = centroid[:2]
                    
                    # elif self.timestamp["C"] == 0:
                    #     if centroid[0] > self.flags["C"]:
                    #         self.timestamp["C"] = frame_num
                    #         self.position["C"] = centroid[:2]

                    # elif self.timestamp["D"] == 0:
                    #     if centroid[0] > self.flags["D"]:
                    #         self.timestamp["D"] = frame_num
                    #         self.position["D"] = centroid[:2]
                    
                    # elif self.timestamp["E"] == 0:
                    #     if centroid[0] > self.flags["E"]:
                    #         self.timestamp["E"] = frame_num
                    #         self.position["E"] = centroid[:2]
                    
                    # elif self.timestamp["F"] == 0:
                    #     if centroid[0] > self.flags["F"]:
                    #         self.timestamp["F"] = frame_num
                    #         self.position["F"] = centroid[:2]
                        
                    # elif self.timestamp["G"] == 0:
                    #     if centroid[0] > self.flags["G"]:
                    #         self.timestamp["G"] = frame_num
                    #         self.position["G"] = centroid[:2]
                    
                    # elif self.timestamp["H"] == 0:
                    #     if centroid[0] > self.flags["H"]:
                    #         if centroid[0] < (self.flags["H"] + self.flags["A"]):
                    #             self.position["H"] = centroid[:2]
                    #         else:
                    #             self.position["H"] = [self.flags["H"] + 2*self.flags["A"], centroid[1]]
                    #     else:
                    #         self.position["H"] = centroid[:2]
                    #     self.timestamp["H"] = frame_num
                    #     self.lastPoint = True

            elif self.direction < 0:
                for x in range(1, self.truthPoints + 1):
                    if self.timestamp[str(x)] == 0:
                        if (x) % self.truthPoints == 0:
                            if centroid[0] < self.flags[str(1)]:
                                if centroid[0] > 0:
                                    self.position[str(x)] = centroid[:2]
                                else:
                                    self.position[str(x)] = [0-self.flags[str(1)],centroid[1]]
                            else:
                                self.position[x] = centroid[:2]
                            self.timestamp[x] = frame_num
                            self.lastPoint = True
                        else:
                            if centroid[0] < self.flags[str(x)]:
                                self.timestamp[str(x)] = frame_num
                                self.position[str(x)] = centroid[:2]

                # if self.timestamp["A"] == 0:
                #     if centroid[0] < self.flags["H"]:
                #         self.timestamp["A"] = frame_num
                #         self.position["A"] = centroid[:2]
                    
                # elif self.timestamp["B"] == 0:
                #     if centroid[0] < self.flags["G"]:
                #         self.timestamp["B"] = frame_num
                #         self.position["B"] = centroid[:2]
                
                # elif self.timestamp["C"] == 0:
                #     if centroid[0] < self.flags["F"]:
                #         self.timestamp["C"] = frame_num
                #         self.position["C"] = centroid[:2]

                # elif self.timestamp["D"] == 0:
                #     if centroid[0] < self.flags["E"]:
                #         self.timestamp["D"] = frame_num
                #         self.position["D"] = centroid[:2]
                
                # elif self.timestamp["E"] == 0:
                #     if centroid[0] < self.flags["D"]:
                #         self.timestamp["E"] = frame_num
                #         self.position["E"] = centroid[:2]

                # elif self.timestamp["F"] == 0:
                #     if centroid[0] < self.flags["C"]:
                #         self.timestamp["F"] = frame_num
                #         self.position["F"] = centroid[:2]
                    
                # elif self.timestamp["G"] == 0:
                #     if centroid[0] < self.flags["B"]:
                #         self.timestamp["G"] = frame_num
                #         self.position["G"] = centroid[:2]
                
                # elif self.timestamp["H"] == 0:
                #     if centroid[0] < self.flags["A"]:
                #         if centroid[0] > 0:
                #             self.position["H"] = centroid[:2]
                #         else:
                #             self.position["H"] = [0-self.flags["A"],centroid[1]]
                #     else:
                #         self.position["H"] = centroid[:2]
                #     self.timestamp["H"] = frame_num
                #     self.lastPoint = True
        
    def calculate_average_speed(self, estimatedSpeeds):
        avera_speed = np.average(estimatedSpeeds)
        
        self.speed = round(avera_speed, 1)