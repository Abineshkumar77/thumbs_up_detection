from fastapi import FastAPI, File, UploadFile
import torch
import cv2
import numpy as np
import time

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
async def home():
    return {"message": "Upload your images"}


model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/abi/Projects/thumbs_up/model/best.pt')


def detect_thumbs_up(image):
    start_time = time.time()  
    
   
    results = model(image)

   
    predictions = results.pred[0]  

 
    thumbs_up_count = 0
    for *box, conf, cls in predictions:
        if model.names[int(cls)] == 'thumbs_up':
            thumbs_up_count += 1

    inference_time = time.time() - start_time  

    
    return {"message": "Yes" if thumbs_up_count > 0 else "No", "inference_time": inference_time}

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    
    image_data = await file.read()
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    
    result = detect_thumbs_up(img)

    return result


