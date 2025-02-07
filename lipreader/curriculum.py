from videos import Video
import numpy as np
from common.constants import IMAGE_FLIP_PROBABILITY

class Curriculum:
    def apply(self, video, train):
        # TODO: implement subsentence
        # if self.sentence_length > 0:
        #     video, align = VideoAugmenter.pick_subsentence(video, align, self.sentence_length)
        # Only apply horizontal flip on training
        if train:
            if np.random.ranf() < IMAGE_FLIP_PROBABILITY:
                video.flip_video()
        return video