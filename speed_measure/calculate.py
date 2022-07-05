import math

def per_distance(prevPos, curPos):
    d = math.sqrt(math.pow(curPos[0] - prevPos[0], 2) + math.pow(curPos[1] - prevPos[1], 2))
    return d

def calculate_speed(_obj, fps, asRt):
    print(_obj.timestamps)
    lenCheck = 0
    if not _obj.estimated:
        for (i, j) in _obj.points:
            lenCheck += 1
            if i == '1' or int(j) == _obj.truthPoints:
                continue
            if (_obj.positions[i] is None or _obj.positions[j] is None):
              break   
            d = per_distance(_obj.positions[i], _obj.positions[j])
            frames = abs(_obj.timestamps[i] - _obj.timestamps[j])
            t = frames/fps
            if d == 0 or t == 0:  
                continue
            
            speed = d / t 
            _asRt = asRt if asRt is not None else _obj.scale 
            _obj.speeds[i+j] = round(speed * 3.6 / _asRt,2)

    if _obj.timestamps[str(_obj.truthPoints)] != 0:
        _obj.calAverageSpeed()
        _obj.calRealTime(fps)
