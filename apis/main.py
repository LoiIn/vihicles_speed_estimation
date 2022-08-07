from config import cfg
import uvicorn
from fastapi import FastAPI, Request, UploadFile, Response, Form, File
from fastapi.middleware.cors import CORSMiddleware
import os
from ..server.speed import main as service
from utils import renderFileName

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
def root():
    return {"Hello": "World"}

# save video from client
@app.post('/upload')
async def root(video: bytes = File()):
    fileName = renderFileName() + '.mp4'
    filePath = os.path.join(cfg.API.DB, fileName)
    f = open(filePath, 'wb')
    f.write(video)
    return fileName

# run api
@app.post("/excute")
async def excute(
    video: str = Form(),
    type: str = Form(),
    rwf: int = Form(),
    rws: int = Form(),
    speed: int = Form()
): 
    test = service(video, rwf, rws, speed, 0)
    print(test)
    return "done"
    
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=cfg.API.PORT)