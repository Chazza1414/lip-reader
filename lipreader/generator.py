from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from keras import backend as K
from lipreader.common.constants import IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS, DATASET_PATH, VIDEO_FRAME_NUM, NUM_PHONEMES
from lipreader.videos import Video, VideoHelper
from lipreader.align import Align
from lipreader.helpers import get_list_safe
import multiprocessing
from lipreader.helpers import threadsafe_generator
from pathlib import Path
import threading

class Generator():
    # any paths passed should use '/'
    def __init__(self, minibatch_size, dataset_path=DATASET_PATH, steps_per_epoch=None):
        self.dataset_path = dataset_path
        self.minibatch_size = minibatch_size
        self.steps_per_epoch = steps_per_epoch
        self.img_c          = IMAGE_CHANNELS
        self.img_w          = IMAGE_WIDTH
        self.img_h          = IMAGE_HEIGHT
        self.output_size = NUM_PHONEMES
        self.cur_train_index = multiprocessing.Value('i', 0)
        self.cur_val_index   = multiprocessing.Value('i', 0)
        self.cur_eval_index   = multiprocessing.Value('i', 0)
        self.train_list_lock = multiprocessing.Lock()
        self.steps_per_epoch     = steps_per_epoch
        self.shared_train_epoch  = multiprocessing.Value('i', -1)
        self.process_train_epoch = -1
        self.random_seed = 13
        self.failed_videos = 0
    
    def build(self):

        self.train_path = Path(self.dataset_path) / 'train'
        self.validate_path = Path(self.dataset_path) / 'validate'
        self.evaluate_path = Path(self.dataset_path) / 'evaluate'
        self.align_path = Path(self.dataset_path) / 'phoneme-alignment'

        self.video_helper = VideoHelper()
        self.train_list = self.video_helper.enumerate_videos(self.train_path)
        self.validate_list   = self.video_helper.enumerate_videos(self.validate_path)
        self.evaluate_list = self.video_helper.enumerate_videos(self.evaluate_path)
        self.align_hash = self.enumerate_phoneme_alignment(self.train_list + self.validate_list + self.evaluate_list)

        self.steps_per_epoch  = self.default_training_steps if self.steps_per_epoch is None else self.steps_per_epoch

        np.random.shuffle(self.train_list)

        return self
    
    @property
    def default_training_steps(self):
        return len(self.train_list) / self.minibatch_size

    @property
    def default_validation_steps(self):
        return len(self.validate_list) / self.minibatch_size
    
    def enumerate_phoneme_alignment(self, video_list):
        phoneme_alignment_dict = {}

        for video_path in video_list:
            
            video_id = os.path.splitext(video_path)[0].split('\\')[-1]
            #print(self.align_path, video_id, video_path)
            phoneme_alignment_path = os.path.join(self.align_path, video_id)+".txt"
            phoneme_alignment_dict[video_id] = Align(phoneme_alignment_path)

        return phoneme_alignment_dict
    
    def get_alignment(self, id):
        return self.align_hash[id]
    
    def get_batch(self, index, size, set_type):
        if set_type == 'train':
            video_list = self.train_list
        elif set_type == 'validate':
            video_list = self.validate_list
        else:
            video_list = self.evaluate_list

        #print(video_list)

        X_data_path = get_list_safe(video_list, index, size)
        X_data = []
        Y_data = []
        
        for path in X_data_path:
            #print(path)
            video = Video().from_path(path)

            align = self.get_alignment(path.split('\\')[-1].split(".")[0])

            X_data.append(video.frames)
            Y_data.append(align.alignment_matrix)

        Y_data = np.array(Y_data)
        X_data = np.array(X_data).astype(np.float32) / 255 # Normalize image data to [0,1], TODO: mean normalization over training data

        #print(X_data.shape)

        return (X_data, Y_data)
    
    def next_train(self):
        r = np.random.RandomState(self.random_seed)
        while 1:
            # print "SI: {}, SE: {}".format(self.cur_train_index.value, self.shared_train_epoch.value)
            with self.cur_train_index.get_lock(), self.shared_train_epoch.get_lock():
                cur_train_index = self.cur_train_index.value
                self.cur_train_index.value += self.minibatch_size
                # Shared epoch increment on start or index >= training in epoch
                if cur_train_index >= self.steps_per_epoch * self.minibatch_size:
                    cur_train_index = 0
                    self.shared_train_epoch.value += 1
                    self.cur_train_index.value = self.minibatch_size
                if self.shared_train_epoch.value < 0:
                    self.shared_train_epoch.value += 1
                # Shared index overflow
                if self.cur_train_index.value >= len(self.train_list):
                    self.cur_train_index.value = self.cur_train_index.value % self.minibatch_size
                # Calculate differences between process and shared epoch
                epoch_differences = self.shared_train_epoch.value - self.process_train_epoch
            if epoch_differences > 0:
                self.process_train_epoch += epoch_differences
                for _ in range(epoch_differences):
                    with self.train_list_lock:
                        r.shuffle(self.train_list) # Catch up

            ret = self.get_batch(cur_train_index, self.minibatch_size, set_type='train')
            yield ret

    def next_validate(self):
        while 1:
            with self.cur_val_index.get_lock():
                cur_val_index = self.cur_val_index.value
                self.cur_val_index.value += self.minibatch_size
                if self.cur_val_index.value >= len(self.validate_list):
                    self.cur_val_index.value = self.cur_val_index.value % self.minibatch_size

            ret = self.get_batch(cur_val_index, self.minibatch_size, set_type='validate')
            yield ret

    def next_evaluate(self):
        while 1:
            with self.cur_eval_index.get_lock():
                cur_eval_index = self.cur_eval_index.value
                self.cur_eval_index.value += self.minibatch_size
                if self.cur_eval_index.value >= len(self.evaluate_list):
                    self.cur_eval_index.value = self.cur_eval_index.value % self.minibatch_size

            ret = self.get_batch(cur_eval_index, self.minibatch_size, set_type='evaluate')
            yield ret

class LockedIterator(object):
    def __init__(self, it):
        self.lock = threading.Lock()
        self.it = iter(it)

    def __iter__(self): 
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()