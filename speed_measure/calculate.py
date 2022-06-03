import math

def calculate_speed(prevPos, curPos, fps):
    # print(curPos)
    _asRt = curPos[3] /1.2
    
    # distance per pixcel
    distance_in_pixels = math.sqrt(math.pow(curPos[0] - prevPos[0], 2) + math.pow(curPos[1] - prevPos[1], 2))

    # distance per met
    distance_in_met = distance_in_pixels / _asRt

    return round(distance_in_met * 3.6 * fps, 2)

def calculate_speed_2(prevPos, curPos, _time, obj_width, obj_height):
    # print(curPos)
    _asRt1 = curPos[2] / obj_width
    _asRt2 = curPos[3] / obj_height

    speedX = (prevPos[0] - curPos[0]) / _asRt1
    speedY = (prevPos[1] - curPos[1]) / _asRt2
    
    _distance = math.sqrt(math.pow(speedX, 2) + math.pow(speedY, 2))
  
    # distance per met
    distance_in_met = _distance / _time

    return round(distance_in_met * 3.6, 2)

def calculate_speed_3(prevPos, curPos, _time, obj_height):
    # print(curPos)
    _asRt = curPos[3] / obj_height
    
    # distance per pixcel
    distance_in_pixels = math.sqrt(math.pow(curPos[0] - prevPos[0], 2) + math.pow(curPos[1] - prevPos[1], 2))

    # distance per met
    distance_in_met = (distance_in_pixels / _asRt) / _time

    return round(distance_in_met * 3.6, 2)

def calculate_per_distance(prevPos, curPos):
    d = math.sqrt(math.pow(curPos[0] - prevPos[0], 2) + math.pow(curPos[1] - prevPos[1], 2))
    return d

def calculate_speed_4(_obj, fps, asRt):
    estimatedSpeeds = []
    if _obj.lastPoint and not _obj.estimated:
        for (i, j) in _obj.points:
            d = calculate_per_distance(_obj.position[i], _obj.position[j])
            frames = abs(_obj.timestamp[i] - _obj.timestamp[j])
            t = frames/fps
            if d == 0 or t == 0:
                continue

            speed = d / t
            _obj.speeds[i+j] = speed
            estimatedSpeeds.append(speed)

    if len(estimatedSpeeds):
        _obj.calculate_speed(estimatedSpeeds, asRt)
        _obj.estimated = True