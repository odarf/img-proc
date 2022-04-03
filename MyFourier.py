import numpy as np
from PIL import Image


class MyFourier:

    def __init__(self, complex_part, dt: int):
        self.complex_part = complex_part
        self.dt = dt

    def amplitude(self):
        y = np.abs(self.complex_part * self.dt)
        x = np.fft.fftfreq(len(self.complex_part), self.dt)
        zero_index = x.tolist().index(np.max(x))
        return np.array([x[:zero_index], y[:zero_index]])
