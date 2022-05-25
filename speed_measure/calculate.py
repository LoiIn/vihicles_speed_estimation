import numpy as np
import math

def calculate_speed(prevPos, curPos, fps):
    # print(curPos)
    _asRt = curPos[3] /1.2
    
    # distance per pixcel
    distance_in_pixels = math.sqrt(math.pow(curPos[0] - prevPos[0], 2) + math.pow(curPos[1] - prevPos[1], 2))

    # distance per met
    distance_in_met = distance_in_pixels / _asRt

    return round(distance_in_met * 3.6 * fps, 2)

def calculate_speed_2(prevPos, curPos, fps, ppmX, ppmY):
    speedX = (prevPos[0] - curPos[0]) / ppmX
    speedY = (prevPos[1] - curPos[1]) / ppmY
    _distance = math.sqrt(math.pow(speedX, 2) + math.pow(speedY, 2))

    return round(_distance * 3.6 * fps, 0)

def calculate_speed_3(prevPos, curPos, _time, obj_height):
    # print(curPos)
    _asRt = curPos[3] / obj_height
    
    # distance per pixcel
    distance_in_pixels = math.sqrt(math.pow(curPos[0] - prevPos[0], 2) + math.pow(curPos[1] - prevPos[1], 2))

    # distance per met
    distance_in_met = (distance_in_pixels / _asRt) / _time

    return round(distance_in_met * 3.6, 2)