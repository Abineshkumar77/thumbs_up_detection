from fastapi import FastAPI, File, UploadFile
import mediapipe as mp
import cv2
import numpy as np
from typing import List
from fastapi.responses import PlainTextResponse
import time  # Import the time module

app = FastAPI()

mp_hands = mp.solutions.hands

@app.get("/")
async def home():
    return {"message": "Upload your images"}

def detect_thumbs_up(image) -> bool:
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(type(image_rgb))
        
        start_time = time.time()  # Start the timer
        
        results = hands.process(image_rgb)
        
        inference_time = time.time() - start_time  # Calculate inference time
        print(f"Inference time: {inference_time:.4f} seconds")  # Print the inference time

        if not results.multi_hand_landmarks:
            return False

        for hand_landmarks in results.multi_hand_landmarks:
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            # Check if the thumb is raised and the other fingers are folded
            if (thumb_tip.y < thumb_ip.y) and \
               (index_finger_tip.y > thumb_tip.y) and \
               (middle_finger_tip.y > thumb_tip.y) and \
               (ring_finger_tip.y > thumb_tip.y) and \
               (pinky_tip.y > thumb_tip.y):
                return True
    
    return False

@app.post("/analyze-images/", response_class=PlainTextResponse)
async def analyze_images(files: List[UploadFile] = File(...)):
    for file in files:
        contents = await file.read()
        nparr = np.fromstring(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if detect_thumbs_up(image):
            return "yes"

    return "no"


