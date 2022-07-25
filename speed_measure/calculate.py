import math

from numpy import arctan

def calRectangle(w, h):
    return (w * h)

def calParallelogram(w1, w2, h):
    return ((w1 + w2) * h /2)

def perDistance(prevPos, curPos):
    d = math.sqrt(math.pow(curPos[0] - prevPos[0], 2) + math.pow(curPos[1] - prevPos[1], 2))
    return d

def getLine(i):
    lineW = (2*3.5 *(8-i))/8 + 5
    return lineW

def getAsRt(y):
    if y > 1038.75 and y < 1080:
        return (getLine(7) + getLine(8)) / 2
    elif y > 997.5 and y < 1038.75:
        return (getLine(6) + getLine(7)) / 2
    elif y > 956.25 and y < 997.5:
        return (getLine(5) + getLine(6)) / 2
    elif y > 915 and y < 956.25:
        return (getLine(4) + getLine(5)) / 2
    elif y > 873.75 and y < 915:
        return (getLine(3) + getLine(4)) / 2
    elif y > 832.5 and y < 873.75:
        return (getLine(2) + getLine(3)) / 2
    elif y > 791.25 and y < 832.5:
        return (getLine(1) + getLine(2)) / 2
    elif y > 750 and y < 791.25:
        return (12 + getLine(1)) / 2
    elif y < 750:
        return 12
    else:
        return 5

def calculateSpeed(_obj, fps, asRt, vW, vH):
    if not _obj.estimated:
        # asRt = round(calRectangle(vW, vH - 750) / calParallelogram(5, 12.05, 7.25), 4)
        # asRt = round(vW / 9.65, 3)
        for (i, j) in _obj.points:
            prevPos = _obj.positions[i]
            curPos = _obj.positions[j]
            if i == '1' or int(j) == _obj.truthPoints:
                continue
            if (curPos is None or prevPos is None):
              break   
            
            d = perDistance(curPos, prevPos)
            frames = abs(_obj.timestamps[i] - _obj.timestamps[j])
            t = frames/fps
            if d == 0 or t == 0:  
                continue
            
            speed = d / t 
            # _asRt = asRt if asRt is not None else _obj.scale 
            _asRt = round(wW / getAsRt(curPos[1]),2)
            
            _obj.speeds[i+j] = round(speed * 3.6 / _asRt,2)

    if _obj.timestamps[str(_obj.truthPoints)] != 0:
        _obj.calculateAverageSpeed()
        _obj.calculateRealTime(fps)
