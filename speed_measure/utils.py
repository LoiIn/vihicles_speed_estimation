def format_center_point(bbox):
    _width = bbox[2] - bbox[0]
    _height = bbox[3] - bbox[1]
    xcenter = bbox[0] + _width/2
    ycenter = bbox[1] + _height/2
    bbox[0], bbox[1], bbox[2], bbox[3] = xcenter, ycenter, _width, _height
    
    return bbox