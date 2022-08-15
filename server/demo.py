import os
from pathlib import Path
import sys
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0]
SERVER_ROOT = FILE.parents[0]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(ROOT / 'server') not in sys.path:
    sys.path.append(str(ROOT / 'server'))

from speed_measure.utils import formatCenterPoint, renderFileName, getVideoName
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from datetime import date
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print("you are using GPU ...")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("not GPU")
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

from speed_measure.speedHorizontialObject import SpeedHorizontialObject as horzObjSpeed
import pandas as pd

def run(
    # weights = WEIGHTS / 'yolov4-416',
    size = 416,
    input = None,
    iou = 0.45,
    score = 0.5,
    rwf = None, 
    rws = None, 
    points = 15, 
    limit = 20, 
    save_video = False, 
    client = None,
    types = None
):
    # Definition of the parameters for object tracking
    max_cosine_distance = 0.6
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'server/model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config()
    input_size = size
    video_path = os.path.join(cfg.SPEED.INPUT, input)
    limit_speed = limit

    saved_model_loaded = tf.saved_model.load('server/detections/yolov4-416', tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    # get video's name and define csv and output location
    today = date.today()
    timeFile = today.strftime("%b-%d-%Y")
    _name = getVideoName(input)
    path_client = os.path.join(cfg.SPEED.CLIENT, client)
    os.mkdir(path_client)
    path_csv = os.path.join(path_client, 'speed.csv')

    # get video's attributes
    _fps = vid.get(cv2.CAP_PROP_FPS)
    _width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    _height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # line position by pixel that can contain way 
    rwf = rwf
    rws = rws
    _way = 750

    # get video ready to save locally if flag is set
    if save_video:
        path_output = os.path.join(cfg.SPEED.OUTPUT, timeFile + '-' + _name + '.mp4')
        out = None
        codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out = cv2.VideoWriter(path_output, codec, _fps, (_width, _height))

    # calculate size of text when put to output frame
    text_size = 1.5
    if _width > 1600:
        text_size *= 1.6
    elif _width > 1200 and _width <= 1600:
        text_size *= 1.2
   
   # Definition of speed estimation
   # Use array to archives of the point used to truncated 
    _estimated_distance = _width
    _number_distances = points - 1
    _flags = {}
    for x in range(1, points + 1):
        _flags[str(x)] = round((x-1)*_estimated_distance / _number_distances)

    # Definition variables for save iamge of the vihicles, which have speed > limit_speed
    objSpeed = {}
    imgSaved = {}
    beforeSaved = {}
    imgCaptured = {}

    # prepare csv's contents
    csv_data = {
        "ID" : [],
        "ClassName" : [],
        "Speed" : [],
        "Times": []
    }

    # others definition
    allowed_classes = None
    class_names = utils.read_class_names(ROOT / cfg.YOLO.CLASSES)
    frame_idx = 0
    if types[0] == 'all':
        allowed_classes = list(class_names.values())
    # allowed_classes = ['motorbike']
    else:
        allowed_classes = types

    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_idx += 1
        # print(frame_idx)
    
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
            iou_threshold=iou,
            score_threshold=score
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
            _i = track.track_id
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            # get bbox and put ID to frame
            bbox = track.to_tlbr()
            color = colors[track.track_id % len(colors)]
            color = [j * 255 for j in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.putText(frame,"ID" + str(_i),(int(bbox[0]), int(bbox[1]) - 10),0, 1, (255,0,0),3)
            
            centroid = formatCenterPoint(bbox) 

            # initialization speed and update speeds of vihicles
            if _i not in objSpeed:            
                objSpeed[_i] = horzObjSpeed(_i, centroid, _flags, points, rwf, rws, _width, _height)                
            else:
                objSpeed[_i].update(centroid, frame_idx)
            
            measure.calculateSpeed(objSpeed[_i], _fps, _width, _way)

            cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 10, (255, 0, 0), 5)

            # Prepare the frame to save the image contain vihicles have speed > limit_speed
            if _i not in imgSaved:
                imgSaved[_i] = False
                beforeSaved[_i] = None
                imgCaptured[_i] = False
            else:
                if not imgSaved[_i]:
                    imgSaved[_i] = False

                if not imgSaved[_i] and objSpeed[_i].direction is not None:
                    if (objSpeed[_i].direction > 0 and centroid[0] > _width / 2) or (objSpeed[_i].direction < 0 and centroid[0] < _width / 2):
                        imgSaved[_i] = True
                        imgCopy = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        beforeSaved[_i] = imgCopy.copy()

            # save info of vihicles to csv
            if objSpeed[_i].speed is not None and not objSpeed[_i].logged and objSpeed[_i].speed > limit_speed: 
                csv_data['ID'].append(_i)
                csv_data['ClassName'].append(track.get_class())
                csv_data["Speed"].append(objSpeed[_i].speed)
                csv_data["Times"].append(objSpeed[_i].realtimes['2'] + '-' + objSpeed[_i].realtimes[str(points - 1)])
                objSpeed[_i].logged = True

            # if speed > limit_speed, export frame contained object
            if objSpeed[_i].logged and objSpeed[_i].speed > limit_speed and beforeSaved[_i] is not None:
                if not imgCaptured[_i]:
                    imgName = os.path.join(path_client, str(_i) + ".jpg")
                    cv2.imwrite(imgName, beforeSaved[_i])
                    imgCaptured[_i] = True

        # save output video
        if save_video:
            result = np.asarray(frame)
            result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    # save csv file
    df = pd.DataFrame(csv_data, columns = ["ID", "ClassName", "Speed", "Times"])
    df.to_csv(path_csv, index = False, header = True)

