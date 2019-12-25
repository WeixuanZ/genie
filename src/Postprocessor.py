import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


# signal processing techniques to remove outlier and smooth the plot

def smooth(vec):
    vec = signal.medfilt(vec, kernel_size=51)
    win = signal.hann(51)
    vec = signal.fftconvolve(vec, win, mode='same') / sum(win)

    return vec


if __name__ == '__main__':
    vec = np.loadtxt('../data/data.csv', delimiter=',')
    plt.plot(np.arange(1, len(vec) + 1, 1), vec)
    plt.plot(np.arange(1, len(vec) + 1, 1), smooth(vec))
    plt.ylim([7.5e6, 7.6e6])
    plt.show()
