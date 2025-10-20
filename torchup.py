import os
import io
import base64
import random
import torch
import torch.nn as nn
import numpy as np
import cv2
import uvicorn
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()
secret_key = os.getenv("EMNIST_SECRET_KEY")
if not secret_key:
    raise RuntimeError("EMNIST_SECRET_KEY environment variable not set")

# ----------------------------
# FastAPI setup
# ----------------------------
app = FastAPI(title="EMNIST CNN Letter Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://torch.localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    SessionMiddleware,
    secret_key=secret_key,
    session_cookie="emnist_session",
)

# ----------------------------
# Model + labels
# ----------------------------
NUM_CLASSES = 26
LABELS = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load trained model
model = CNN(NUM_CLASSES).to(device)
model.load_state_dict(torch.load("cnn_emnist_upper_weights.pth", map_location=device))
model.eval()

# ----------------------------
# Helper functions
# ----------------------------
def preprocess_image(file_bytes):
    """Convert uploaded image bytes to normalized tensor (1x1x28x28)."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (28, 28))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=(0, 1))  # shape: (1, 1, 28, 28)
    return torch.from_numpy(img)

def generate_label_image(label: str) -> str:
    """Generate base64-encoded PNG image of the target label."""
    img = Image.new("RGBA", (112, 112), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("ARIAL.TTF", size=64)
    except:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    draw.text(((112 - w) / 2 - x0, (112 - h) / 2 - y0), label, fill=(255, 255, 255, 255), font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def predict_label(file_bytes) -> str | None:
    """Run model inference on an uploaded image and return predicted label."""
    img_tensor = preprocess_image(file_bytes)
    if img_tensor is None:
        return None
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        logits = model(img_tensor)
        pred_idx = torch.argmax(logits, dim=1).item()
    return LABELS[pred_idx]

# ----------------------------
# API endpoints
# ----------------------------
@app.get("/random_label_image")
async def random_label_image(request: Request):
    """Generate a random target letter image for user to match."""
    label = random.choice(LABELS)
    request.session["target"] = label
    img_b64 = generate_label_image(label)
    return {"image": img_b64, "label": label}

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    """Predict uploaded handwritten letter and check if it matches the session target."""
    target = request.session.get("target")
    if not target:
        return JSONResponse({"error": "Invalid or expired session"}, status_code=400)
    file_bytes = await file.read()
    pred_label = predict_label(file_bytes)
    if pred_label is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)
    is_match = pred_label == target
    return {"match": is_match, "predicted": pred_label, "target": target}

# ----------------------------
# Run server
# ----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
