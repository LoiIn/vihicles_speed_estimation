import numpy as np

class SpeedHorizontialObject:
    def __init__(self, objectID, centroid, color, flags, _points):
        self.objectID = objectID
        self.centroids = [centroid[:2]]
        self.bbox = centroid
        self.scale = centroid[2] / 1.908
        self.color = color
        self.direction = None
        self.flags = flags
        self.points = self.initPoints(_points)
        self.timestamp = self.initTimestamp(_points + 1)
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
        centroid[0] = round(centroid[0],2)
        centroid[1] = round(centroid[1],2)
        self.bbox = centroid
        self.scale = centroid[2] / 1.908
        if not self.estimated:
            self.centroids.append(centroid[:2])
        
            if self.direction is None:
                self.direction = 0
            if self.direction == 0 and (len(self.centroids) > 2):
                vertical = [c[1] for c in self.centroids]
                self.direction = centroid[1] - np.mean(vertical)

            if self.direction > 0:
                _pl = None
                _pl_val = None
                for x in range(1, self.truthPoints + 1):
                    if self.timestamp[str(x)] == 0:
                        _pl = x
                        break
                
                if _pl is not None: 
                    _pl_val = str(_pl)
                    if _pl % self.truthPoints == 0:
                        if centroid[1] > self.flags[_pl_val]:
                            if centroid[1] < (self.flags[_pl_val] + self.flags[str(1)]):
                                self.position[_pl_val] = centroid[:2]
                            else:
                                self.position[_pl_val] = [self.flags[_pl_val] + self.flags[str(1)], centroid[1]]
                            
                            self.timestamp[_pl_val] = frame_num
                            self.lastPoint = True
                    else:
                        if centroid[1] > self.flags[_pl_val]:
                            self.timestamp[_pl_val] = frame_num
                            self.position[_pl_val] = centroid[:2]

            elif self.direction < 0:
                _pl = None
                _pl_val = None
                for x in range(1, self.truthPoints + 1):
                    if self.timestamp[str(x)] == 0:
                        _pl = x
                        break
            
                if _pl is not None: 
                    _pl_val = str(_pl)
                    if _pl % self.truthPoints == 0:
                        if centroid[1] < self.flags[str(1)]:
                            if centroid[1] > 0:
                                self.position[_pl_val] = centroid[:2]
                            else:
                                self.position[_pl_val] = [0-self.flags[str(1)],centroid[1]]
                            self.timestamp[_pl_val] = frame_num
                            self.lastPoint = True
                    else:
                        if centroid[1] < self.flags[str(self.truthPoints + 1 - _pl)]:
                            self.timestamp[_pl_val] = frame_num
                            self.position[_pl_val] = centroid[:2]
        
        
    def calculate_average_speed(self, estimatedSpeeds):
        avera_speed = np.average(estimatedSpeeds)
        
        self.speed = round(avera_speed, 1)