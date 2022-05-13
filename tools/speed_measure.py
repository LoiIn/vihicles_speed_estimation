from turtle import distance
import numpy as np
import math

def calculate_speed(prevPos, curPos, fps, pixel_per_metter):
    # distance per pixcel
    distance_in_pixels = math.sqrt(math.pow(curPos[0] - prevPos[0], 2) + math.pow(curPos[1] - prevPos[1], 2))

    # distance per met
    distance_in_met = distance_in_pixels / pixel_per_metter

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

    return round(_distance * 3.6 * fps, 2)
