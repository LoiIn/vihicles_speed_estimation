from pandas import array
from pyparsing import Optional
import uvicorn
from fastapi import FastAPI, Request, UploadFile, Response, Form, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
import os

file_path = "videos/u/u1.mp4"

app = FastAPI()

origins = [
    "http://localhost",
    "https://localhost",
    "http://localhost:3000",
    "http://localhost:8080"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PORT = 3000


app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

class Data(BaseModel):
    # video: UploadFile = File(),
    rwf: str = Form(),
    rws: str = Form(),
    speed: str = Form()

@app.get("/")
def root():
    return {"Hello": "World"}

@app.get("/items/{id}", response_class=HTMLResponse)
async def read_item(request: Request, id: str):
    return templates.TemplateResponse("index.html", {"request": request, "id": id})

@app.post("/laravel")
async def test_api(
    video: UploadFile = File(),
    rwf: str = Form(),
    rws: str = Form(),
    speed: str = Form()
): 
    return speed


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=PORT)