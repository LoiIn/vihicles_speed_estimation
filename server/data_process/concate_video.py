import cv2
import numpy as np
from absl import app, flags
from absl.flags import FLAGS

flags.DEFINE_string('video1', None, 'original video')
flags.DEFINE_string('video2', None, 'truth video 1')
flags.DEFINE_string('video3', None, 'truth video 2')
flags.DEFINE_string('output', None, 'output video')
# flags.DEFINE_integer('output_w', 480, 'output width')
# flags.DEFINE_integer('output_h', 400, 'output height')
# flags.DEFINE_integer('output_type', 0, '0: Vertically, 1: Horizontally')

def main(_argv):
    try:
        vid1 = cv2.VideoCapture(int(FLAGS.video1))
    except:
        vid1 = cv2.VideoCapture(FLAGS.video1)

    try:
        vid2 = cv2.VideoCapture(int(FLAGS.video2))
    except:
        vid2 = cv2.VideoCapture(FLAGS.video2)

    try:
        vid3 = cv2.VideoCapture(int(FLAGS.video3))
    except:
        vid3 = cv2.VideoCapture(FLAGS.video3)

    

    out = None
    # _width =  FLAGS.output_w if FLAGS.output_type == 0 else FLAGS.output_w * 2
    # _width = 960
    # _height = FLAGS.output_h * 2 if FLAGS.output_type == 0 else FLAGS.output_h
    # _height = 400
    fps = 24
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(FLAGS.output, codec, fps, (1140, 400))

    # _w, _h = FLAGS.output_w, FLAGS.output_h
    while True:
        success1, frame1 = vid1.read()
        success2, frame2 = vid2.read()
        success3, frame3 = vid3.read()

        if success1:
            try: 
                frame1 = cv2.resize(frame1, (760, 400))
                frame2 = cv2.resize(frame2, (380, 200))
                frame3 = cv2.resize(frame3, (380, 200))
            except:
                break
        else:
            print('Video has ended!')
            break
        
        cv2.putText(frame2,"green",(20, 20),0, 1, (255,0,0),2)
        cv2.putText(frame3,"white",(20, 20),0, 1, (255,0,0),2)
        frame_23 = np.concatenate((frame2, frame3), axis=0)
        _frame = np.concatenate((frame1, frame_23), axis=1)
        _frame = cv2.cvtColor(_frame, cv2.COLOR_BGR2RGB)

        rs = np.asarray(_frame)
        rs = cv2.cvtColor(_frame, cv2.COLOR_RGB2BGR)
        out.write(rs)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    vid1.release()
    vid2.release()
    vid3.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass