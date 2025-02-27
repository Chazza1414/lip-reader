from keras.optimizers import Adam
from keras import backend as K
from keras import models
from keras.losses import MeanSquaredError
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lipreader.model import LipReader
from lipreader.spell import Spell
from lipreader.decoders import Decoder
from lipreader.helpers import labels_to_text
from lipreader.videos import Video, VideoHelper
import tensorflow as tf
import numpy as np
from lipreader.common.constants import MODEL_SAVE_LOCATION, NUM_PHONEMES, VIDEO_FRAME_NUM, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, PHONEME_LIST
from pathlib import Path

class Predictor():
    def __init__(self, test_set_path, model_file_name):
        self.test_set_path = test_set_path
        self.model_file_name = model_file_name
        self.img_c = IMAGE_CHANNELS
        self.img_w = IMAGE_WIDTH
        self.img_h = IMAGE_HEIGHT
        self.frames_n = VIDEO_FRAME_NUM
    
    def predict(self):
        testing_model = models.load_model(self.model_file_name)
        videohelper = VideoHelper()
        X_data_path = videohelper.enumerate_videos(self.test_set_path)

        #X_data_path = self.enumerate_videos(self.test_set_path)
        X_data = []
        
        for path in X_data_path:
            print(path)
            video = Video().from_path(path)

            if (video is not None):
                X_data.append(video.frames)

        X_data = np.array(X_data).astype(np.float32) / 255

        predictions = testing_model.predict(X_data)[0]
        #most_likely_phonemes = tf.nn.softmax(predictions)
        #most_likely_phonemes = most_likely_phonemes.numpy()
        #print(predictions)
        most_likely_phonemes = np.argmax(predictions, axis=1)
        phoneme_list = np.array(PHONEME_LIST)
        print(most_likely_phonemes)
        print(phoneme_list[most_likely_phonemes])

        # print(predictions)
        # print(predictions.shape)
        # print(np.sum(predictions, axis=1))

# model_file_name = Path(MODEL_SAVE_LOCATION) / '2025-02-24-10-11-55'
#model_file_name = Path(MODEL_SAVE_LOCATION) / '2025-02-24-12-46-03'
#model_file_name = Path(MODEL_SAVE_LOCATION) / '2025-02-24-13-41-32' #4 epoch
model_file_name = Path(MODEL_SAVE_LOCATION) / '2025-02-26-11-54-35'
test_set_path = "H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\datasets\\evaluate"

predictor = Predictor(test_set_path, model_file_name)
predictor.predict()