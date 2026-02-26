import tkinter as tk
from pathlib import Path

from PIL import Image, ImageDraw
import time
import torch
from torchvision import transforms
import torch.nn.functional as F
from model import DigitNet

# ================================
# 1. Load trained model
# ================================
model = DigitNet()
BASE_DIR = Path(__file__).resolve().parent.parent
weights_path = BASE_DIR / "weights" / "digit_model.pth"

model.load_state_dict(torch.load(weights_path, map_location="cpu"))
model.eval()

# ================================
# 2. Tkinter setup
# ================================
canvas_size = 28      # matches training input
display_size = 280
idle_time = 1.0
last_draw_time = time.time()

root = tk.Tk()
root.title("Draw Digit (0-9)")

canvas_widget = tk.Canvas(root, width=display_size, height=display_size, bg='black')
canvas_widget.pack()

label_widget = tk.Label(root, text="Draw a digit...", font=("Arial", 20))
label_widget.pack()

scale = display_size // canvas_size

# Create blank image for drawing
img = Image.new("L", (canvas_size, canvas_size), color=0)
draw = ImageDraw.Draw(img)

drawing = False
last_x, last_y = None, None

# limit drawing area
margin = int(display_size * 0.1)
min_x, max_x = margin, display_size - margin
min_y, max_y = margin, display_size - margin

def clip_bounds(x, y):
    return max(min_x, min(x, max_x)), max(min_y, min(y, max_y))

def start_draw(event):
    global drawing, last_x, last_y, last_draw_time
    drawing = True
    last_x, last_y = clip_bounds(event.x, event.y)
    last_draw_time = time.time()

def draw_motion(event):
    global last_x, last_y, last_draw_time
    if drawing:
        x, y = clip_bounds(event.x, event.y)
        # draw on canvas
        canvas_widget.create_line(last_x, last_y, x, y, fill="white", width=3)
        # draw on scaled 28x28 image
        x1, y1 = last_x // scale, last_y // scale
        x2, y2 = x // scale, y // scale
        draw.line([x1, y1, x2, y2], fill=255, width=1)
        last_x, last_y = x, y
        last_draw_time = time.time()

def end_draw(event):
    global drawing, last_draw_time
    drawing = False
    last_draw_time = time.time()

canvas_widget.bind("<ButtonPress-1>", start_draw)
canvas_widget.bind("<B1-Motion>", draw_motion)
canvas_widget.bind("<ButtonRelease-1>", end_draw)

# ================================
# 3. Preprocessing
# ================================
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ================================
# 4. Auto predict when idle
# ================================
def check_idle():
    global img, draw, last_draw_time
    if not drawing and (time.time() - last_draw_time) > idle_time and img.getbbox():
        x = transform(img).unsqueeze(0)  # 1x1x28x28
        with torch.no_grad():
            out = model(x)
            probs = F.softmax(out, dim=1)
            pred_digit = torch.argmax(probs).item()
            confidence = probs[0, pred_digit].item()

        label_widget.config(
            text=f"Prediction: {pred_digit}  (conf {confidence:.2f})"
        )

        # Clear canvas
        canvas_widget.delete("all")
        canvas_widget.create_rectangle(
            margin, margin,
            display_size - margin, display_size - margin,
            outline="#00FF00", width=1
        )
        img.paste(0, (0, 0, canvas_size, canvas_size))

    root.after(100, check_idle)

root.after(100, check_idle)

# ================================
# 5. Center window
# ================================
win_w, win_h = 400, 450
sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
x = (sw - win_w) // 2
y = (sh - win_h) // 2
root.geometry(f"{win_w}x{win_h}+{x}+{y}")

root.mainloop()