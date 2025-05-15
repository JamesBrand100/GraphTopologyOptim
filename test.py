import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# (re‑use your logits, temps, softmax, etc. from before)
num_classes = 10
k = 4
logits = np.zeros(num_classes)
logits[[1,3,5,7]] = np.array([2.0, 1.5, 3.0, 2.5])
temps = np.linspace(0.1, 5.0, 100)

fig, ax = plt.subplots(figsize=(8,4))
bars = ax.bar(range(num_classes), np.zeros(num_classes), color='skyblue')
ax.set_ylim(0, k)
ax.set_xlabel("Class Index")
ax.set_ylabel(f"Expected k‑hot Count (k={k})")

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def update(frame):
    tau = temps[frame]
    probs = softmax(logits / tau)
    for bar, h in zip(bars, probs * k):
        bar.set_height(h)
    ax.set_title(f"Temperature τ = {tau:.2f}")
    return bars

anim = FuncAnimation(fig, update, frames=len(temps), blit=True, interval=100)

# Convert to JS and display
HTML(anim.to_jshtml())
