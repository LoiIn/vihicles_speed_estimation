import math

def per_distance(prevPos, curPos):
    d = math.sqrt(math.pow(curPos[0] - prevPos[0], 2) + math.pow(curPos[1] - prevPos[1], 2))
    return d

def calculate_speed(_obj, fps, asRt, start_time):
    estimatedSpeeds = []
    if not _obj.estimated:
        for (i, j) in _obj.points:
            if (_obj.position[i] is None or _obj.position[j] is None):
                break
            d = per_distance(_obj.position[i], _obj.position[j])
            frames = abs(_obj.timestamp[i] - _obj.timestamp[j])
            t = frames/fps
            # _moment = start_time + _obj.timestamp[i] / fpsP
            if d == 0 or t == 0:  
                continue
            
            speed = d / t 
            _asRt = asRt if asRt is not None else _obj.scale 
            _obj.speeds[i+j] = round(speed * 3.6 / _asRt, 2)
            estimatedSpeeds.append(speed * 3.6 / _asRt)

    if len(estimatedSpeeds) == _obj.truthPoints - 1:
        _obj.calculate_average_speed(estimatedSpeeds)
        _obj.estimated = True