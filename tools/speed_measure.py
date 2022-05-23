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

def calculate_speed_2(prevPos, curPos, fps):
    part_1 = math.pow(math.sin((prevPos[0] - curPos[0])/2), 2)
    part_2 = math.cos(prevPos[0]) * math.cos(curPos[0])
    part_3 = part_2 * math.pow(math.sin((prevPos[1] - curPos[1])/2), 2)
    distance_ = 2 * np.arcsin(math.sqrt(part_1 + part_3))

    return round(distance_ * 3.6 * fps, 2)

def calculate_speed_3(prevPos, curPos, fps, ppmX, ppmY):
    speedX = (prevPos[0] - curPos[0]) / ppmX
    speedY = (prevPos[1] - curPos[1]) / ppmY
    _distance = math.sqrt(math.pow(speedX, 2) + math.pow(speedY, 2))

    return round(_distance * 3.6 * fps, 0)

def calculate_speed_4(prevPos, curPos, time, height, far):
    _angle = np.arctan(height / far)
    u1 = prevPos[0]
    u2 = curPos[0]
    v1 = prevPos[1]
    v2 = curPos[1]
    _asRt = curPos[2] / 1.5

    _A = math.pow(v2-v1, 2) + math.pow((u1-u2)*math.sin(_angle), 2)
    _B = 2 * (u1-u2) * (u1*v2-u2*v1)*math.sin(_angle)*math.cos(_angle)
    _C = math.pow((u1*v2-u2*v1)*math.cos(_angle), 2)
    _D = math.pow(math.sin(_angle), 4)
    _E = 2*(v1+v2)*math.pow(math.sin(_angle), 3)*math.cos(_angle)
    _F = (math.pow(v1+v2, 2) + 2*v1*v2)*math.pow(math.sin(_angle)*math.cos(_angle), 2)
    _G = 2*(v1+v2)*v1*v2*math.sin(_angle)*math.pow(math.cos(_angle), 3)
    _H = math.pow(v1*v2, 2) * math.pow(math.cos(_angle), 4)

    _above = _A*math.pow(_asRt, 2) + _B*_asRt + _C
    _under = _D*math.pow(_asRt, 4) + _E*math.pow(_asRt, 3) + _F*math.pow(_asRt, 2) + _G*_asRt + _H

    _distance = height * math.sqrt(_above / _under)

    return round((_distance / time) * 3.6, 2)