from apis.config import cfg
import uvicorn
from fastapi import FastAPI, Request, UploadFile, Response, Form, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
# import server.test as test
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

@app.get('/test')
async def test():
    saveFile()
    return getLink("demo/test.csv")

# test api
@app.get("/")
async def root():
    return {"Hello": "World"}

# save video and return rs
@app.post('/excute', response_class=FileResponse)
async def root(
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
    
    responseFilePath = os.path.join(ROOT / cfg.API.DB, clientDir + '/' + 'speed.csv')
    return responseFilePath

def saveFile():
    path_on_cloud = "speed"
    path_local = os.path.join(ROOT / cfg.API.DB, 'test.csv')
    storage.child(path_on_cloud).put(path_local)

def getLink(pathCloud):
    return storage.child(pathCloud).get_url(None)
    
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=cfg.API.PORT)