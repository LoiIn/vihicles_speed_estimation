import numpy as np
from speed_measure.utils import convertSecondToMinute

class SpeedHorizontialObject:
    def __init__(self, objectID, centroid, color, flags, points):
        self.objectID = objectID
        self.centroids = [(round(centroid[0], 2), round(centroid[1], 2))]
        self.bbox = centroid
        self.scale = centroid[2] / 1.908
        self.color = color
        self.direction = None
        self.flags = flags
        self.points = self.initPoints(points)
        self.timestamps = self.initTimestamp(points + 1)
        self.speeds = self.initSpeeds(points)
        self.positions = self.initPosition(points + 1)
        self.lastPoint = False
        self.estimated = False
        self.speed = None
        self.logged = False
        self.truthPoints = points
        self.realtimes = self.initRealtimes(points + 1)

    def initPoints(self, points):
        listP = []
        for x in range(1, points):
            listP.append((str(x), str(x + 1)))
        return listP
    
    def initTimestamp(self, points):
        timestamps = {}
        for x in range(1, points):
            timestamps[str(x)] = 0
        return timestamps

    def initSpeeds(self, points):
        speeds = {}
        for x in range(1, points):
            speeds[str(x) + str(x+1)] = None
        return speeds

    def initPosition(self, points):
        poses = {}
        for x in range(1, points):
            poses[str(x)] = None
        return poses

    def initRealtimes(self, points):
        rts = {}
        for x in range(1, points):
            rts[str(x)] = None
        return rts

    def update(self, centroid, frame_num):
        self.bbox = centroid
        centroid[0] = round(centroid[0],2)
        centroid[1] = round(centroid[1],2)
        self.scale = centroid[2] / 1.908
        if not self.estimated:
            self.centroids.append(centroid[:2])
        
            if self.direction is None:
                self.direction = 0
            if self.direction == 0 and (len(self.centroids) > 1):
                vertical = [c[0] for c in self.centroids]
                self.direction = centroid[0] - np.mean(vertical)

            if self.direction > 0:
                _pl = None
                _pl_val = None
                for x in range(1, self.truthPoints + 1):
                    if self.timestamps[str(x)] == 0:
                        _pl = x
                        break
                
                if _pl is not None:
                    # dot = round(centroid[0] + centroid[2] / 2, 2)
                    xdot = round(centroid[0],2)
                    ydot = round(centroid[1] + centroid[3] / 2)
                    _pl_val = str(_pl)
                    if _pl == 1:
                        self.timestamps['1'] = frame_num
                    elif _pl == self.truthPoints:
                        self.timestamps[_pl_val] = frame_num
                        self.lastPoint = True
                    else:
                        if xdot > self.flags[_pl_val]:
                            self.timestamps[_pl_val] = frame_num
                            self.positions[_pl_val] = [xdot, ydot]

            elif self.direction < 0:
                _pl = None
                _pl_val = None
                for x in range(1, self.truthPoints + 1):
                    if self.timestamps[str(x)] == 0:
                        _pl = x
                        break
            
                if _pl is not None: 
                    # dot = round(centroid[0] - centroid[2] / 2, 2)
                    xdot = round(centroid[0],2)
                    ydot = round(centroid[1] + centroid[3] / 2)
                    _pl_val = str(_pl)
                    if _pl == 1:
                        self.timestamps['1'] = frame_num
                    elif _pl == self.truthPoints:
                        self.timestamps[_pl_val] = frame_num
                        self.lastPoint = True
                    else:
                        if xdot < self.flags[str(self.truthPoints + 1 - _pl)]:
                            self.timestamps[_pl_val] = frame_num
                            self.positions[_pl_val] = [xdot, ydot]
    
    def customSpeed(self):
        estimatedSpeeds = []
        custom = 0
        for (i, j) in self.points:
            if int(i) != self.truthPoints:
                if self.speeds[i+j] is not None:
                    self.speeds[i+j] += custom
                    estimatedSpeeds.append(round(self.speeds[i+j],2))
        
        return estimatedSpeeds

    def calculateAverageSpeed(self):
        _estimate = self.customSpeed()
        
        avera_speed = np.mean(_estimate)
        
        self.speed = round(avera_speed, 1)
        self.estimated = True
    
    def calculateRealTime(self, fps):
        for i in range(1, self.truthPoints + 1):
            if self.timestamps[str(i)] != 0:
                _time = round(self.timestamps[str(i)] / fps, 2)
                self.realtimes[str(i)] = convertSecondToMinute(_time)

