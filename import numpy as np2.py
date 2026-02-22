import numpy as np
import sounddevice as sd
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# -----------------------
# PARAMETERS
# -----------------------
fs = 19000     # Sampling frequency
T = 0.3            # Duration of tone (seconds)

# DTMF Frequency Table
dtmf_freq = {
    "1": (697, 1209), "2": (697, 1336), "3": (697, 1477), "A": (697, 1633),
    "4": (770, 1209), "5": (770, 1336), "6": (770, 1477), "B": (770, 1633),
    "7": (852, 1209), "8": (852, 1336), "9": (852, 1477), "C": (852, 1633),
    "*": (941, 1209), "0": (941, 1336), "#": (941, 1477), "D": (941, 1633),
}

# -----------------------
# SIGNAL GENERATION
# -----------------------
def play_dtmf(key):
    low, high = dtmf_freq[key]
    
    t = np.linspace(0, T, int(fs*T), endpoint=False)
    signal = np.sin(2*np.pi*low*t) + np.sin(2*np.pi*high*t)
    
    # Normalize to prevent clipping
    signal = signal / np.max(np.abs(signal))
    
    # Play sound
    sd.play(signal, fs)
    
    # Update graph
    ax.clear()
    ax.plot(t[:400], signal[:400])  # Show small portion for clarity
    ax.set_title(f"DTMF Signal for Key {key} ({low}Hz + {high}Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    canvas.draw()

# -----------------------
# GUI SETUP
# -----------------------
root = tk.Tk()
root.title("DTMF Signal Generator")

# Matplotlib Figure
fig, ax = plt.subplots(figsize=(6,3))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().grid(row=0, column=0, columnspan=4)

# Keypad Layout
keys = [
    ["1","2","3","A"],
    ["4","5","6","B"],
    ["7","8","9","C"],
    ["*","0","#","D"]
]

for r in range(4):
    for c in range(4):
        key = keys[r][c]
        btn = tk.Button(root, text=key, width=6, height=3,
                        command=lambda k=key: play_dtmf(k))
        btn.grid(row=r+1, column=c)

root.mainloop()