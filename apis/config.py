#! /usr/bin/env python
# coding=utf-8
from easydict import EasyDict as edict


__C                           = edict()
# Consumers can get config by: from config import cfg

cfg                           = __C

# YOLO options
__C.API                     = edict()

__C.API.DB              = "../videos/clients"
__C.API.PORT            = 3000
__C.API.ORIGIN          =  [  
                            "http://localhost",
                            "https://localhost",
                            "http://localhost:3000",
                            "http://localhost:8080"
                        ]
                    

