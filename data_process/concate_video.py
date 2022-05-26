import cv2
import numpy as np
from absl import app, flags
from absl.flags import FLAGS

flags.DEFINE_string('video1', None, 'video original')
flags.DEFINE_string('video2', None, 'video truth')
flags.DEFINE_string('output', None, 'output video')
flags.DEFINE_integer('output_w', 480, 'output width')
flags.DEFINE_integer('output_h', 300, 'output height')
flags.DEFINE_integer('output_type', 0, '0: Vertically, 1: Horizontally')

def main(_argv):
    try:
        vid1 = cv2.VideoCapture(int(FLAGS.video1))
    except:
        vid1 = cv2.VideoCapture(FLAGS.video1)

    try:
        vid2 = cv2.VideoCapture(int(FLAGS.video2))
    except:
        vid2 = cv2.VideoCapture(FLAGS.video2)

    out = None
    _width =  FLAGS.output_w if FLAGS.output_type == 0 else FLAGS.output_w * 2
    _height = FLAGS.output_h * 2 if FLAGS.output_type == 0 else FLAGS.output_h
    fps = 24
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(FLAGS.output, codec, fps, (_width, _height))

    _w, _h = FLAGS.output_w, FLAGS.output_h
    while True:
        success1, frame1 = vid1.read()
        success2, frame2 = vid2.read()

        if success1:
            frame1 = cv2.resize(frame1, (_w, _h))
            frame2 = cv2.resize(frame2, (_w, _h))
        else:
            print('Video has ended!')
            break

        _frame = np.concatenate((frame1, frame2), axis=FLAGS.output_type)
        _frame = cv2.cvtColor(_frame, cv2.COLOR_BGR2RGB)

        rs = np.asarray(_frame)
        out.write(rs)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    vid1.release()
    vid2.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass