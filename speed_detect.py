from inspect import ArgSpec
from multiprocessing.dummy import current_process
import os

from speed_measure.utils import format_center_point
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags
from absl.flags import FLAGS
import core.utils as utils
import speed_measure.calculate as measure
# from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
# from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416', 'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')
flags.DEFINE_float('distance', None, 'reality of distance')
flags.DEFINE_integer('A_point', 10, 'define x position of cross line')
flags.DEFINE_float('start_time', 0, 'time start at real world')

from speed_measure.speedAbleObject import SpeedAbleObject
import pandas as pd

def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config()
    input_size = FLAGS.size
    video_path = FLAGS.video

    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None
    _fps = int(vid.get(cv2.CAP_PROP_FPS))
    _width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    _estimated_distance = _width - FLAGS.A_point * 2
    _flags = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0}
    _flags['A'] = FLAGS.A_point
    _flags['B'] = FLAGS.A_point + _estimated_distance / 5
    _flags['C'] = FLAGS.A_point + 2* _estimated_distance / 5
    _flags['D'] = FLAGS.A_point + 3* _estimated_distance / 5
    _flags['E'] = FLAGS.A_point + 4* _estimated_distance / 5
    _flags['F'] = _width - FLAGS.A_point

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = _width
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = _fps
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    objSpeed = {}
    csv_data = {
        "ID" : [],
        "ClassName" : [],
        "Speed" : [],
        "Speeds": [],
        "Position": [],
        "Timestamp": []
    }
    frame_idx = 0

    # read in all class names from config
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)

    # by default allow all classes in .names file
    # allowed_classes = list(class_names.values())
    
    # custom allowed classes (uncomment line below to customize tracker for only people)
    allowed_classes = ['motorbike', 'car']

    # while video is running
    while True:
        # start_time = time.time()
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_idx += 1
    
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)

        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)

        # print('Frame #: ', frame_idx)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)        
        detections = [detections[i] for i in indices]      

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        for track in tracker.tracks:
            _i = track.track_id
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            color = colors[track.track_id % len(colors)]
            color = [j * 255 for j in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.putText(frame,"v" + str(_i) + "-",(int(bbox[0]), int(bbox[1]) - 10),0, 0.9, (255,255,255),2)
            
            centroid = format_center_point(bbox)
            if _i not in objSpeed:            
                objSpeed[_i] = SpeedAbleObject(_i, centroid, color, _flags)
            else:
                objSpeed[_i].update(bbox, frame_idx)
            
            if FLAGS.distance is not None:
                measure.calculate_speed( objSpeed[_i], _fps, _width/FLAGS.distance, FLAGS.start_time)
            else:
                measure.calculate_speed( objSpeed[_i], _fps, None, FLAGS.start_time)
            
            if objSpeed[_i].speeds["EF"] is not None:
                cv2.putText(frame,str(objSpeed[_i].speeds["EF"]),(int(bbox[0]) + 50, int(bbox[1]) - 10),0, 0.9, (255,255,255),2)
            elif objSpeed[_i].speeds["DE"] is not None:
                cv2.putText(frame,str(objSpeed[_i].speeds["DE"]),(int(bbox[0]) + 50, int(bbox[1]) - 10),0, 0.9, (255,255,255),2)
            elif objSpeed[_i].speeds["CD"] is not None:
                cv2.putText(frame,str(objSpeed[_i].speeds["CD"]),(int(bbox[0]) + 50, int(bbox[1]) - 10),0, 0.9, (255,255,255),2)
            elif objSpeed[_i].speeds["BC"] is not None:
                cv2.putText(frame,str(objSpeed[_i].speeds["BC"]),(int(bbox[0]) + 50, int(bbox[1]) - 10),0, 0.9, (255,255,255),2)
            elif objSpeed[_i].speeds["AB"] is not None:
                cv2.putText(frame,str(objSpeed[_i].speeds["AB"]),(int(bbox[0]) + 50, int(bbox[1]) - 10),0, 0.9, (255,255,255),2)
            
            if objSpeed[_i].speed is not None: 
                csv_data['ID'].append(_i)
                csv_data['ClassName'].append(track.get_class())
                csv_data["Speed"].append(objSpeed[_i].speed)
                csv_data["Speeds"].append(objSpeed[_i].speeds)
                csv_data["Position"].append(objSpeed[_i].position)
                csv_data["Timestamp"].append(objSpeed[_i].timestamp)
            
        for f in flags:
            cv2.line(frame, (flags[f], 0), (flags[f], int(video_height)), (0,0,255), 2)

        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    df = pd.DataFrame(csv_data, columns = ["ID", "ClassName", "Speed", "Speeds", "Position", "Timestamp"])
    df.to_csv(r'test.csv', index = False, header = True)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
