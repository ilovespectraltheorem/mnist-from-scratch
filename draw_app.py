"""Draw digits with the mouse and watch the model read them.

    python main.py draw

Drawings go through preprocess_drawing() from src/robustness.py. The small
picture on the right is exactly what the network sees.
"""
import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw, ImageTk

from src.experiments import load_model
from src.robustness import preprocess_drawing

SIZE = 280        # drawing area in screen pixels
BRUSH = 10        # brush radius; after scaling to 28x28 this is roughly MNIST's stroke width
BLUE, PALE = "#2a78d6", "#b7d3f6"


def main(model_path):
    model, cfg = load_model(model_path)
    root = tk.Tk()
    root.title(f"Draw a digit  ({model_path.name})")
    root.configure(bg="white")

    board = tk.Canvas(root, width=SIZE, height=SIZE, bg="black", highlightthickness=0, cursor="pencil")
    board.grid(row=0, column=0, rowspan=3, padx=14, pady=14)
    image = Image.new("L", (SIZE, SIZE), 0)
    pen = ImageDraw.Draw(image)

    tk.Label(root, text="What the network sees (28x28)", bg="white").grid(row=0, column=1, sticky="w")
    preview = tk.Label(root, bg="white")
    preview.grid(row=1, column=1, sticky="nw")
    bars = tk.Canvas(root, width=230, height=10 * 22 + 4, bg="white", highlightthickness=0)
    bars.grid(row=2, column=1, sticky="nw", padx=(0, 14))
    last = {"xy": None}

    def stroke(event):
        x, y = event.x, event.y
        if last["xy"] is not None:
            board.create_line(*last["xy"], x, y, fill="white", width=2 * BRUSH, capstyle=tk.ROUND)
            pen.line([last["xy"], (x, y)], fill=255, width=2 * BRUSH)
        board.create_oval(x - BRUSH, y - BRUSH, x + BRUSH, y + BRUSH, fill="white", outline="white")
        pen.ellipse([x - BRUSH, y - BRUSH, x + BRUSH, y + BRUSH], fill=255)
        last["xy"] = (x, y)

    def release(_event):
        last["xy"] = None
        predict()

    def predict():
        x = preprocess_drawing(np.asarray(image))
        probs = model.predict_proba(x[None])[0]
        big = Image.fromarray((x.reshape(28, 28) * 255).astype(np.uint8)).resize((112, 112), Image.NEAREST)
        photo = ImageTk.PhotoImage(big)
        preview.configure(image=photo)
        preview.image = photo                       # keep a reference or Tk drops it
        bars.delete("all")
        best = int(np.argmax(probs))
        for d in range(10):
            y0 = d * 22 + 4
            weight = "bold" if d == best else "normal"
            bars.create_text(6, y0 + 8, text=str(d), anchor="w", font=("Helvetica", 12, weight))
            bars.create_rectangle(26, y0 + 2, 26 + 150 * probs[d], y0 + 15,
                                  fill=BLUE if d == best else PALE, outline="")
            bars.create_text(184, y0 + 8, text=f"{100 * probs[d]:.0f}%", anchor="w", font=("Helvetica", 10))

    def clear():
        board.delete("all")
        pen.rectangle([0, 0, SIZE, SIZE], fill=0)
        bars.delete("all")
        preview.configure(image="")

    board.bind("<Button-1>", stroke)
    board.bind("<B1-Motion>", stroke)
    board.bind("<ButtonRelease-1>", release)
    tk.Button(root, text="Clear", command=clear).grid(row=3, column=0, pady=(0, 14))
    root.mainloop()
