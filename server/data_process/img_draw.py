import tensorflow as tf
# from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from absl import app

def main(_argv):
    video_path = "../../videos/vu1.mp4"
    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)


    # get video's attributes
    _fps = vid.get(cv2.CAP_PROP_FPS)
    _width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    _height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # calculate size of text when put to output frame
    text_size = 1
    _estimated_distance = _width  - 0 * 2
    _number_distances = 9 - 1
    _flags = {}
    for x in range(1, 9 + 1):
        _flags[str(x)] = 0 + round((x-1)*_estimated_distance / _number_distances)

    frame_idx = 0

    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            print("hihi")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_idx += 1

        for f in _flags:
            cv2.putText(frame,str(_flags[f]),(int(_flags[f]), _height - 20),0, text_size, (255,0,0),text_size)
            cv2.line(frame, (int(_flags[f]), 0), (int(_flags[f]), _height), (0,0,255), 2)

        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if frame_idx == 1:
            cv2.imwrite("../outputs/imgs/test.jpg", result)
        else: break

        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
