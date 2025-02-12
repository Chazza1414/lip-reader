import cv2 as cv
import imageio as iio
import numpy as np

class Video(object):
    def __init__(self):
        self.frames = np.array([])
    def from_path(self, path):
        self.frames = np.array(iio.imread(path, plugin='pyav'))
        return self
    def flip_video(self):
        if (self.frames != []):
            self.frames = np.flip(self.frames, axis=2)
