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
    milisecond = ''
    if i < 0.1:
        milisecond = '0'
    
    milisecond += str(int(i*100))
    return str(m) + ':' + str(s) + ':' + milisecond

def calSecondPositonInPixel(rwf, rws, vH):
    return (vH * rwf) / (rws - rwf)

def getVideoName(originalName):
    return originalName.split(".")[0]