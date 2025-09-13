import uvicorn
import random 
import torch
import uuid
import numpy as np
import cv2
import io
import base64
from torch import nn
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw, ImageFont

app = FastAPI(title="EMNIST CNN Letter Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://torch.localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NUM_CLASSES = 26

LABELS = [chr(i) for i in range(ord("A"), ord("Z")+1)]

TARGETS = {} 

model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),

    nn.Flatten(),
    nn.Linear(128 * 3 * 3, 128),
    nn.ReLU(),
    nn.Linear(128, NUM_CLASSES)
)

model.load_state_dict(torch.load("cnn_emnist_upper_weights.pth", map_location="cpu"))

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

def preprocess_image(file_bytes):
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (28, 28))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0) 
    img = np.expand_dims(img, axis=0)  
    return torch.from_numpy(img)

@app.get("/random_label_image")
async def random_label_image():
    token = str(uuid.uuid4())
    label = random.choice(LABELS)
    TARGETS[token] = label

    img = Image.new("RGBA", (112, 112), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("ARIAL.TTF", size=64)
    except:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), label, font=font)
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0

    draw.text(
        ((112 - w) / 2 - x0, (112 - h) / 2 - y0),
        label,
        fill=(255, 255, 255, 255),
        font=font
    )

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")

    return {"token": token, "image": img_b64}

@app.post("/predict")
async def predict(file: UploadFile = File(...), token: str = Form(...)):
    try:
       
        target = TARGETS.get(token)
        if target is None:
            return JSONResponse({"error": "Invalid or missing token"}, status_code=400)

        img_tensor = preprocess_image(await file.read())
        if img_tensor is None:
            return JSONResponse({"error": "Invalid image"}, status_code=400)

        img_tensor = img_tensor.to(device)
        with torch.no_grad():
            logits = model(img_tensor)
            pred_idx = torch.argmax(logits, dim=1).item()
            pred_label = LABELS[pred_idx]

        is_match = pred_label == target

        return {
            "match": is_match
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
