from ntpath import join
from apis.config import cfg
import uvicorn
from fastapi import FastAPI, Form, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
import os
from apis.utils import renderFileName, getTypes
from apis.firebase import storage
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(ROOT / 'apis') not in sys.path:
    sys.path.append(str(ROOT / 'apis'))

import server.demo as demo

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.API.ORIGIN,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# test api
@app.get("/")
async def root():
    return {"Hello": "World"}

# save video and return rs
@app.post('/excute', response_class=PlainTextResponse)
async def excute(
    video: bytes = File(),
    rwf: float = Form(),
    rws: float = Form(),
    limit: float = Form(),
    type: str = Form()
):
    # return getTypes(type)
    clientDir = renderFileName() 
    clientName = clientDir + '.mp4'
    filePath = os.path.join(ROOT / cfg.API.VIDEO, clientName)
    f = open(filePath, 'wb')
    f.write(video)

    # return clientDir
    demo.run(input=clientName, rwf=rwf, rws=rws, limit=limit, client=clientDir, types = getTypes(type))
    
    saveFile(clientDir)
    rs = getLink(clientDir)
    return rs


@app.get('/test')
async def get_all():
    return getLink("ekthsudxh")

def saveFile(dir):
    path = os.path.join(ROOT / cfg.API.DB , dir)
    files = os.listdir(path)
    for f in files:
        path_cloud = cfg.API.CLOUD + '/' + dir + '_' + f
        path_local = os.path.join(path, f)
        storage.child(path_cloud).put(path_local)

def getLink(dir):
    path = os.path.join(ROOT / cfg.API.DB , dir)
    files = os.listdir(path)
    listFiles = ''
    for f in files:
        path_cloud = 'speeds/' + dir + '_' + f
        listFiles += storage.child(path_cloud).get_url(None) + ','
    
    return listFiles
    
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=cfg.API.PORT)