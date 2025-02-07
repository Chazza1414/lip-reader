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
from common.constants import IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS, DATASET_PATH
from videos import Video

class Generator():
    def __init__(self, steps_per_epoch, dataset_path=DATASET_PATH):
        self.dataset_path = dataset_path
        self.steps_per_epoch = steps_per_epoch
    
    def build(self):
        self.train_path     = os.path.join(self.dataset_path, 'train')
        self.validate_path       = os.path.join(self.dataset_path, 'validate')
        self.align_path     = os.path.join(self.dataset_path, 'phoneme-alignment')
        # Set steps to dataset size if not set
        #self.steps_per_epoch  = self.default_training_steps if self.steps_per_epoch is None else self.steps_per_epoch
        #self.validation_steps = self.default_validation_steps if self.validation_steps is None else self.validation_steps

        self.train_list = self.enumerate_videos(self.train_path)
        self.validate_list   = self.enumerate_videos(self.validate_path)
        self.align_hash = self.enumerate_phoneme_alignment(self.train_list + self.validate_list)

        np.random.shuffle(self.train_list)

        return self
    
    # adds the video file path to an array with some fancy error handling
    def enumerate_videos(self, path):
        video_list = []
        for video_path in glob.glob(path):
            try:
                if os.path.isfile(video_path):
                    video = Video().from_path(video_path)
                else:
                    raise FileNotFoundError
            except AttributeError as err:
                raise err
            except FileNotFoundError as err:
                raise err
            except:
                print("Error loading video: "+video_path)
                continue
            if K.image_data_format() == 'channels_first' and video.data.shape != (self.img_c,self.frames_n,self.img_w,self.img_h):
                print("Video "+video_path+" has incorrect shape "+str(video.data.shape)+", must be "+str((self.img_c,self.frames_n,self.img_w,self.img_h))+"")
                continue
            if K.image_data_format() != 'channels_first' and video.data.shape != (self.frames_n,self.img_w,self.img_h,self.img_c):
                print("Video "+video_path+" has incorrect shape "+str(video.data.shape)+", must be "+str((self.frames_n,self.img_w,self.img_h,self.img_c))+"")
                continue
            video_list.append(video_path)
        return video_list
    
    def enumerate_phoneme_alignment(self, video_list):
        phoneme_alignment_dict = {}
        for video_path in video_list:
            video_id = os.path.splitext(video_path)[0].split('/')[-1]
            align_path = os.path.join(self.align_path, video_id)+".align"
            phoneme_alignment_dict[video_id] = Align(self.absolute_max_string_len, text_to_labels).from_file(align_path)
        return phoneme_alignment_dict