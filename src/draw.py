import tkinter as tk
from PIL import Image, ImageDraw
import time
import os
import re

# --- Parameters ---
canvas_size = 28
display_size = 280  # for easier drawing

# Ask label first
label = input("Enter digit label (0–9): ")
if not label.isdigit() or not (0 <= int(label) <= 9):
    raise ValueError("Label must be a digit between 0 and 9.")
# --- Get project root dynamically ---
project_root = os.path.dirname(os.path.abspath(__file__))  # if draw.py is in src/
project_root = os.path.dirname(project_root)  # one level above src/

# --- Save directory ---
save_dir = os.path.join(project_root, "data", label)
os.makedirs(save_dir, exist_ok=True)

# --- Counter: continue from latest saved file ---
existing_files = os.listdir(save_dir)
numbers = []
for f in existing_files:
    match = re.match(r"img_(\d+)\.png", f)
    if match:
        numbers.append(int(match.group(1)))
counter = max(numbers)+1 if numbers else 0

print(f"Starting from img_{counter + 1}.png")

last_draw_time = time.time()
idle_time = 1  # seconds to wait before saving

# --- Create image and draw object ---
img = Image.new("L", (canvas_size, canvas_size), color=0)  # 28x28 grayscale
draw = ImageDraw.Draw(img)

# --- Tkinter setup ---
root = tk.Tk()
root.title(f"Draw {label}")

canvas_widget = tk.Canvas(root, width=display_size, height=display_size, bg='black')
canvas_widget.pack()

# --- Scale drawing ---
scale = display_size // canvas_size

# --- Drawing variables ---
drawing = False
last_x, last_y = None, None

def start_draw(event):
    global drawing, last_x, last_y, last_draw_time
    drawing = True
    last_x, last_y = event.x, event.y
    last_draw_time = time.time()

def draw_motion(event):
    global last_x, last_y, last_draw_time
    if drawing:
        canvas_widget.create_line(last_x, last_y, event.x, event.y, fill="white", width=3)
        # Draw scaled line on 28x28 image
        x1, y1 = last_x // scale, last_y // scale
        x2, y2 = event.x // scale, event.y // scale
        draw.line([x1, y1, x2, y2], fill=255, width=1)
        last_x, last_y = event.x, event.y
        last_draw_time = time.time()

def end_draw(event):
    global drawing, last_draw_time
    drawing = False
    last_draw_time = time.time()

canvas_widget.bind("<ButtonPress-1>", start_draw)
canvas_widget.bind("<B1-Motion>", draw_motion)
canvas_widget.bind("<ButtonRelease-1>", end_draw)

# --- Main loop with auto-save ---
def check_idle():
    global img, draw, counter, last_draw_time
    if not drawing and (time.time() - last_draw_time) > idle_time and img.getbbox():
        filename = os.path.join(save_dir, f"img_{counter + 1}.png")
        img.save(filename)
        print(f"Saved {filename}")
        counter += 1
        # Clear canvas
        canvas_widget.delete("all")
        img = Image.new("L", (canvas_size, canvas_size), color=0)
        draw = ImageDraw.Draw(img)
    root.after(100, check_idle)

root.after(100, check_idle)
root.mainloop()
