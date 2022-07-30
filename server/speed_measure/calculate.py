import math

def perDistance(prevPos, curPos):
    d = math.sqrt(math.pow(curPos[0] - prevPos[0], 2) + math.pow(curPos[1] - prevPos[1], 2))
    return d

def calculateSpeed(_obj, fps, vW, way):
    if not _obj.estimated:
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
            _asRt = round(vW / _obj.getAsRt(curPos[1], way),2)
            
            _obj.speeds[i+j] = round(speed * 3.6 / _asRt,2)

    if _obj.timestamps[str(_obj.truthPoints)] != 0:
        _obj.calculateAverageSpeed()
        _obj.calculateRealTime(fps)
