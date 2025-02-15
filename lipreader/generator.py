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
from lipreader.videos import Video
from lipreader.align import Align
from lipreader.helpers import get_list_safe
import multiprocessing
from lipreader.helpers import threadsafe_generator
from pathlib import Path

class Generator():
    # any paths passed should use '/'
    def __init__(self, minibatch_size, dataset_path=DATASET_PATH, steps_per_epoch=None, **kwargs):
        self.dataset_path = dataset_path
        self.minibatch_size = minibatch_size
        self.steps_per_epoch = steps_per_epoch
        self.img_c          = IMAGE_CHANNELS
        self.img_w          = IMAGE_WIDTH
        self.img_h          = IMAGE_HEIGHT
        self.output_size = NUM_PHONEMES
        self.cur_train_index = multiprocessing.Value('i', 0)
        self.cur_val_index   = multiprocessing.Value('i', 0)
        self.train_list_lock = multiprocessing.Lock()
        self.steps_per_epoch     = kwargs.get('steps_per_epoch', None)
        self.validation_steps    = kwargs.get('validation_steps', None)
        self.process_epoch       = -1
        self.shared_train_epoch  = multiprocessing.Value('i', -1)
        self.process_train_epoch = -1
        self.process_train_index = -1
        self.process_val_index   = -1
        self.random_seed = 13
    
    def build(self):
        # training video names should be of the form speakerid_filename.wav
        # self.train_path     = os.path.join(self.dataset_path, 'train')
        # self.validate_path       = os.path.join(self.dataset_path, 'validate')
        # self.align_path     = os.path.join(self.dataset_path, 'phoneme-alignment')

        self.train_path     = Path(self.dataset_path) / 'train'
        self.validate_path       = Path(self.dataset_path) / 'validate'
        self.align_path     = Path(self.dataset_path) / 'phoneme-alignment'

        # Set steps to dataset size if not set
        print("train path")
        print(self.train_path)
        self.train_list = self.enumerate_videos(self.train_path)
        self.validate_list   = self.enumerate_videos(self.validate_path)
        self.align_hash = self.enumerate_phoneme_alignment(self.train_list + self.validate_list)

        self.steps_per_epoch  = self.default_training_steps if self.steps_per_epoch is None else self.steps_per_epoch
        self.validation_steps = self.default_validation_steps if self.validation_steps is None else self.validation_steps

        np.random.shuffle(self.train_list)

        return self
    
    @property
    def default_training_steps(self):
        return len(self.train_list) / self.minibatch_size

    @property
    def default_validation_steps(self):
        return len(self.validate_list) / self.minibatch_size

    # adds the video file path to an array with some fancy error handling
    def enumerate_videos(self, path):
        print("enumerating vidoes")
        print(path)
        video_list = []
        for video_path in Path(path).glob('*'):
            try:
                if os.path.isfile(video_path):
                    video = Video().from_path(video_path)
                    if K.image_data_format() == 'channels_first' and video.frames.shape != (self.img_c,VIDEO_FRAME_NUM,self.img_w,self.img_h):
                        print("Video "+str(video_path)+" has incorrect shape "+str(video.frames.shape)+", must be "+str((self.img_c,VIDEO_FRAME_NUM,self.img_w,self.img_h))+"")
                        continue
                    if K.image_data_format() != 'channels_first' and video.frames.shape != (VIDEO_FRAME_NUM,self.img_w,self.img_h,self.img_c):
                        print("Video "+str(video_path)+" has incorrect shape "+str(video.frames.shape)+", must be "+str((VIDEO_FRAME_NUM,self.img_w,self.img_h,self.img_c))+"")
                        continue
                else:
                    raise FileNotFoundError
            except AttributeError as err:
                raise err
            except FileNotFoundError as err:
                raise err
            except:
                print("Error loading video: "+ str(video_path))
                continue
            video_list.append(video_path)
        return video_list
    
    def enumerate_phoneme_alignment(self, video_list):
        phoneme_alignment_dict = {}

        for video_path in video_list:
            video_id = os.path.splitext(video_path)[0].split('/')[-1]
            phoneme_alignment_path = os.path.join(self.align_path, video_id)+".txt"
            phoneme_alignment_dict[video_id] = Align(phoneme_alignment_path)

        return phoneme_alignment_dict
    
    def get_alignment(self, id):
        return self.align_hash[id]
    
    def get_batch(self, index, size, train):
        if train:
            video_list = self.train_list
        else:
            video_list = self.validate_list

        print("getting batch")
        print(video_list, index, size)

        X_data_path = get_list_safe(video_list, index, size)
        X_data = []
        Y_data = []
        label_length = []
        input_length = []
        source_str = []
        
        for path in X_data_path:
            #print(path)
            video = Video().from_path(path)
            align = self.get_alignment(path.split('/')[-1])
            
            # if self.curriculum is not None:
            #     video, align, video_unpadded_length = self.curriculum.apply(video, align)

            X_data.append(video.frames)
            Y_data.append(align.alignment_matrix)

            # label_length.append(align.label_length) # CHANGED [A] -> A, CHECK!
            # # input_length.append([video_unpadded_length - 2]) # 2 first frame discarded
            # input_length.append(video.length) # Just use the video padded length to avoid CTC No path found error (v_len < a_len)
            # source_str.append(align.sentence) # CHANGED [A] -> A, CHECK!

        # source_str = np.array(source_str)
        # label_length = np.array(label_length)
        # input_length = np.array(input_length)
        Y_data = np.array(Y_data)
        X_data = np.array(X_data).astype(np.float32) / 255 # Normalize image data to [0,1], TODO: mean normalization over training data

        # inputs = {'the_input': X_data,
        #           'the_labels': Y_data,
        #           'input_length': input_length,
        #           'label_length': label_length,
        #           'source_str': source_str  # used for visualization only
        #           }
        # inputs = {'the_input': X_data, 'the_labels': Y_data}
        # outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function

        # return (inputs, outputs)
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

            # if self.curriculum is not None and self.curriculum.epoch != self.process_train_epoch:
            #     self.update_curriculum(self.process_train_epoch, train=True)

            ret = self.get_batch(cur_train_index, self.minibatch_size, train=True)
            print("yielding")
            print(ret)
            yield ret

    def next_val(self):
        while 1:
            with self.cur_val_index.get_lock():
                cur_val_index = self.cur_val_index.value
                self.cur_val_index.value += self.minibatch_size
                if self.cur_val_index.value >= len(self.validate_list):
                    self.cur_val_index.value = self.cur_val_index.value % self.minibatch_size

            # if self.curriculum is not None and self.curriculum.epoch != self.process_epoch:
            #     self.update_curriculum(self.process_epoch, train=False)

            ret = self.get_batch(cur_val_index, self.minibatch_size, train=False)
            yield ret