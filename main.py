from fastapi import FastAPI, File, UploadFile, Form, Cookie, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import requests
from PIL import Image
import io
import base64
import os


# FastAPI
app = FastAPI()


allow_origin = "allow_origin"

base_url = "url_to_stablediffusion"
UPLOAD_DIR = 'temp'
model = "modelname"


# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=[allow_domain],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["Content-Type","Cookie"],
)


class PayloadType(BaseModel):
    prompt: str

class ControlnetRequest:
    def __init__(self, prompt, file):
        self.url = f'{base_url}/sdapi/v1/img2img'
        self.prompt = prompt
        self.file = file
        self.file_path = None
        self.png_data = None
        self.body = None

    def build_body(self):
        self.img_to_path_data()
        self.body = {
            "init_images": self.png_data,
            "prompt": self.prompt,
            "negative_prompt": "(nude), (anime), (ugly), (unbalanced), (low quality), error",
            "batch_size": 1,
            "steps": 20,
            "cfg_scale": 7,
            "width": 512,
            "height": 512,
            "alwayson_scripts": {
                "controlnet": {
                    "args": [
                        {
                            "enabled": True,
                            "module": "reference_only",
                            "model": "None",
                            "weight": 1.0,
                            "resize_mode": 1,
                            "lowvram": False,
                            "processor_res": 64,
                            "threshold_a": 0.5,
                            "threshold_b": 64,
                            "guidance_start": 0.0,
                            "guidance_end": 1.0,
                            "control_mode": 0,
                            "pixel_perfect": False
                        }
                    ]
                }
            }
        }

    def send_request(self):
        response = requests.post(url=self.url, json=self.body)
        return response.json()
    
    
    def img_to_path_data(self):
        self.file_path = os.path.join(UPLOAD_DIR, self.file.filename)
        with open(self.file_path, "wb") as buffer:
            shutil.copyfileobj(self.file.file,buffer)
        image = Image.open(self.file_path)
        
        with io.BytesIO() as img_bytes:
            image.save(img_bytes, format='PNG')
            img_b64 = base64.b64encode(img_bytes.getvalue()).decode()
        self.png_data = [img_b64]
        return 



@app.post("/generate")
async def img2img(file: UploadFile = File(...), text: str = Form(...), user_id: str = Cookie(None)):
    print(user_id)
    
    control_net = ControlnetRequest(text, file)
    control_net.build_body()
    output = control_net.send_request()
    result = output['images'][0]
    image = Image.open(io.BytesIO(base64.b64decode(result)))
    file_name = (
        "../frontend/public/images/image_"
        + "temp"
        + ".png"
    )#本当は良くないらしい
    image.save(file_name)