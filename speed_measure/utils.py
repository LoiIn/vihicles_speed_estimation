import math

def formatCenterPoint(bbox):
    _width = bbox[2] - bbox[0]
    _height = bbox[3] - bbox[1]
    xcenter = bbox[0] + _width/2
    ycenter = bbox[1] + _height/2
    return [xcenter, ycenter, _width, _height]

def convertSecondToMinute(time):
    i, d = math.modf(time)
    m = int(d // 60)
    s = int(d -m*60)
    return str(m) + ':' + str(s) + ':' + str(int(i*100)) 