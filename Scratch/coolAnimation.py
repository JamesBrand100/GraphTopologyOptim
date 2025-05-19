import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def sharpen_top_n_np(X, normLim=3, temperature=5.0):
    X = X.copy()
    sorted_X = np.sort(X, axis=1)
    ref_val = (sorted_X[:, -normLim] + sorted_X[:, -normLim - 1]) / 2  # avg of n and n-1
    ref_val = ref_val[:, np.newaxis]

    # Avoid division by zero
    eps = 1e-8
    x_pow = (X / (ref_val + eps)) ** temperature
    complement_pow = ((1 - X) / (1 - ref_val + eps)) ** temperature

    sharpened = x_pow / (x_pow + complement_pow + eps)
    return np.clip(sharpened, 0, 1)

X = np.arange(10, dtype=np.float32).reshape(1, -1)
X = (X - X.min()) / (X.max() - X.min())

fig, ax = plt.subplots()
line, = ax.plot([], [], 'o-', lw=2)
text = ax.text(0.5, 0.9, '', transform=ax.transAxes, ha='center')

def init():
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-0.1, 1.1)
    return line, text

def animate(i):
    temperature = 0.5 + (i / 49) * 20
    Y = sharpen_top_n_np(X, normLim=3, temperature=temperature).squeeze()
    line.set_data(np.arange(10), Y)
    text.set_text(f'Temperature: {temperature:.2f}')
    return line, text

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=50, blit=True)

plt.show()
