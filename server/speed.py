from email.mime import image
from inspect import ArgSpec
from multiprocessing.dummy import current_process
import os

from speed_measure.utils import formatCenterPoint, calSecondPositonInPixel, getVideoName
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
from datetime import date
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print("you are using GPU ...")
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
flags.DEFINE_string('weights', './detections/yolov4-416', 'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('input', None, 'path to input video')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_float('rwf', None, 'width first of screen in real world')
flags.DEFINE_float('rws', None, 'width second of screen in real world')
flags.DEFINE_integer('A_point', 0, 'define first point positon')
flags.DEFINE_integer('points', 9, 'Number of truth point')
flags.DEFINE_integer('limit', 20, 'limit of speed')
flags.DEFINE_integer('video_type', 0, 'O: horizontical, 1: vertical')

from speed_measure.speedVerticalObject import SpeedVerticalObject as vertObjSpeed
from speed_measure.speedHorizontialObject import SpeedHorizontialObject as horzObjSpeed
import pandas as pd

def main(_argv):
    # Definition of the parameters for object tracking
    max_cosine_distance = 0.6
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
    video_path = os.path.join(cfg.SPEED.INPUT, FLAGS.input)
    limit_speed = int(FLAGS.limit)

    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    # get video's name and define csv and output location
    today = date.today()
    timeFile = today.strftime("%b-%d-%Y")
    _name = getVideoName(FLAGS.input)
    path_csv = os.path.join(cfg.SPEED.CSV, timeFile + '_' +  _name + '.csv')
    path_img = os.path.join(cfg.SPEED.IMG, timeFile + '_' + _name)
    os.mkdir(path_img)
    
    # path_output = os.path.join(cfg.SPEED.OUTPUT, timeFile + '-' + _name + '.mp4')

    # get video's attributes
    _fps = vid.get(cv2.CAP_PROP_FPS)
    _width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    _height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # line position by pixel that can contain way 
    rwf = FLAGS.rwf
    rws = FLAGS.rws
    _way = 750

    # get video ready to save locally if flag is set
    # out = None
    # codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # out = cv2.VideoWriter(path_output, codec, _fps, (_width, _height))

    # calculate size of text when put to output frame
    text_size = 1
    # if _width > 1600:
    #     text_size *= 2
    # elif _width > 1200 and _width <= 1600:
    #     text_size *= 1.5
   
   # Definition of speed estimation
    _estimated_distance = (_width if FLAGS.video_type == 0 else _height) - FLAGS.A_point * 2
    _number_distances = FLAGS.points - 1
    _flags = {}
    for x in range(1, FLAGS.points + 1):
        _flags[str(x)] = FLAGS.A_point + round((x-1)*_estimated_distance / _number_distances)

    objSpeed = {}
    csv_data = {
        "ID" : [],
        "ClassName" : [],
        "Speed" : [],
        "Speeds": [],
        "Positions": [],
        "Timestamps": [],
        "Times": []
    }

    # others definition
    frame_idx = 0
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)
    # allowed_classes = list(class_names.values())
    allowed_classes = ['motorbike']

    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
            imgSaved = False
            beforeSaved = None
            _i = track.track_id
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            color = colors[track.track_id % len(colors)]
            color = [j * 255 for j in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.putText(frame,"ID" + str(_i),(int(bbox[0]), int(bbox[1]) - 10),0, 1, (255,0,0),3)
            
            centroid = formatCenterPoint(bbox) 
            if centroid[0] > int(_width / 2) and not imgSaved:
                imgSaved = True
                imgCopy = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                beforeSaved = imgCopy.copy()

            cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 10, (255, 0, 0), 5)
            if _i not in objSpeed:            
                if FLAGS.video_type == 0:
                    objSpeed[_i] = horzObjSpeed(_i, centroid, _flags, FLAGS.points, rwf, rws, _width, _height)
                elif FLAGS.video_type == 1:
                    objSpeed[_i] = vertObjSpeed(_i, centroid, _flags, FLAGS.points)                    
            else:
                objSpeed[_i].update(centroid, frame_idx)
            
            measure.calculateSpeed(objSpeed[_i], _fps, _width, _way)

            for x in range(1, FLAGS.points):
                str_x = str(FLAGS.points - x) + str(FLAGS.points + 1 - x )
                if objSpeed[_i].speeds[str_x] is not None and objSpeed[_i].speeds[str_x] > 3.0:
                    cv2.putText(frame,str(objSpeed[_i].speeds[str_x]),(int(bbox[0]) + 150, int(bbox[1]) - 10),0, text_size, (255,0,0),2*text_size)
                    break
            
            if objSpeed[_i].speed is not None and not objSpeed[_i].logged: 
                csv_data['ID'].append(_i)
                csv_data['ClassName'].append(track.get_class())
                csv_data["Speed"].append(objSpeed[_i].speed)
                csv_data["Speeds"].append(objSpeed[_i].speeds)
                csv_data["Positions"].append(objSpeed[_i].positions)
                csv_data["Timestamps"].append(objSpeed[_i].timestamps)
                csv_data["Times"].append(objSpeed[_i].realtimes['3'] + '-' + objSpeed[_i].realtimes['8'])
                objSpeed[_i].logged = True

            if objSpeed[_i].logged and objSpeed[_i].speed > limit_speed and beforeSaved is not None:
                imgName = os.path.join(path_img, str(_i) + ".jpg")
                cv2.imwrite(imgName, beforeSaved)

        # cv2.line(frame, (0, 750), (_width, 750), (0,0,255), 2)
        # cv2.line(frame, (0, _height), (_width, _height), (0,0,255), 2)

        # for f in _flags:
        #     if FLAGS.video_type == 0:
        #         # if int(f) != 1 and int(f) != FLAGS.points:
        #         cv2.putText(frame,str(_flags[f]),(int(_flags[f]), _height - 20),0, text_size, (255,0,0),text_size)
        #         cv2.line(frame, (int(_flags[f]), 0), (int(_flags[f]), _height), (0,0,255), 2)
        #     else:
        #         # if int(f) != 1 and int(f) != FLAGS.points:
        #         cv2.line(frame, (0, int(_flags[f])), (_width, int(_flags[f])), (0,0,255), 2)

        # result = np.asarray(frame)
        # result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # if output flag is set, save video file
        # out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    df = pd.DataFrame(csv_data, columns = ["ID", "ClassName", "Speed", "Speeds", "Positions", "Timestamps", "Times"])
    df.to_csv(path_csv, index = False, header = True)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
