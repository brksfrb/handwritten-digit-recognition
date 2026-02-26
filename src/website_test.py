from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from model import DigitNet

app = FastAPI()

# Allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- Load trained model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DigitNet().to(device)
model.load_state_dict(torch.load("../digit_model.pth", map_location=device))
model.eval()

# --- Preprocessing ---
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.post("/predict")
async def predict_digit(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("L")

    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1)
        digit = torch.argmax(probs, dim=1).item()
        confidence = probs[0, digit].item()

    return {"digit": digit, "confidence": float(confidence)}
