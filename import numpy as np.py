import numpy as np
import matplotlib.pyplot as plt

# Frequencies
f0 = 111
f1 = f0
f2 = f0/2
f3 = 10*f0

fs = 11100

# Common time axis (3 periods of lowest frequency)
T_common = 3/f2
t = np.arange(0, T_common, 1/fs)

# Signals
x1 = np.sin(2*np.pi*f1*t)
x2 = np.sin(2*np.pi*f2*t)
x3 = np.sin(2*np.pi*f3*t)
x_sum = x1 + x2 + x3

plt.figure(figsize=(10,9))

signals = [x1, x2, x3, x_sum]
titles = ["f1 = 111 Hz",
          "f2 = 55.5 Hz",
          "f3 = 1110 Hz",
          "Sum of Signals"]

for i in range(4):
    plt.subplot(4,1,i+1)
    plt.plot(t, signals[i])
    
    # Only axis lines
    plt.axhline(0)
    plt.axvline(0)
    
    plt.title(titles[i])
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()