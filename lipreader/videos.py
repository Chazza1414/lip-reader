import cv2 as cv
import imageio as iio
import numpy as np

class Video(object):
    def __init__(self):
        self.frames = []
    def from_path(self, path):
        self.frames = np.array(iio.imread(path, plugin='pyav'))
    def flip_video(self):
        if (self.frames != []):
            self.frames = np.flip(self.frames)

class VideoAugmenter(object):
    @staticmethod
    def horizontal_flip(video):
        _video = Video(video.vtype, video.face_predictor_path)
        _video.face = np.flip(video.face, 2)
        _video.mouth = np.flip(video.mouth, 2)
        _video.set_data(_video.mouth)
        return _video