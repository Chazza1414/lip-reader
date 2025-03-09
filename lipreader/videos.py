import cv2 as cv
import imageio.v3 as iio
import numpy as np
import dlib
import keras.backend as K
from lipreader.common.constants import IMAGE_HEIGHT, IMAGE_WIDTH, VIDEO_FRAME_NUM, IMAGE_CHANNELS
from pathlib import Path
import os

class Video(object):
    def __init__(self):
        self.frames = np.array([])
        #self.face_detector = dlib.get_frontal_face_detector()
        #self.landmark_detector = dlib.shape_predictor("../training/predictors/shape_predictor_68_face_landmarks.dat")
        #self.path = path
        self.lip_frames = np.array([])
        #self.identify_face()

    def from_path(self, path):
        try:
            self.frames = np.array(iio.imread(path, plugin='pyav'))
            num_frames = self.frames.shape[0]
            width = self.frames.shape[1]
            height = self.frames.shape[2]

            if (num_frames < 30):
                print("not enough frames " + str(num_frames))
                raise Exception
            # if the frame rate is very low double the length
            if (num_frames*2 < VIDEO_FRAME_NUM):
                self.frames = np.repeat(self.frames, 2, axis=0)
            # if we are missing a few frames extend the silence
            if (num_frames < VIDEO_FRAME_NUM):
                num_frames_needed = VIDEO_FRAME_NUM - num_frames
                silence_frame = self.frames[-1:]
                repeated_silence = np.repeat(silence_frame, num_frames_needed, axis=0)

                self.frames = np.concatenate([self.frames, repeated_silence], axis=0)
            elif (num_frames > VIDEO_FRAME_NUM):
                self.frames = self.frames[:VIDEO_FRAME_NUM]

            if (width == IMAGE_HEIGHT and height == IMAGE_WIDTH):
                self.frames = np.swapaxes(self.frames, 1, 2)
        except Exception as error:
            #print("Error loading video")
            return None

        return self

    def flip_video(self):
        if (self.frames.shape == (75, 100, 50, 3)):
            self.frames = np.flip(self.frames, axis=1)

class VideoHelper():
    def __init__(self):
        self.img_c = IMAGE_CHANNELS
        self.frame_num = VIDEO_FRAME_NUM
        self.img_w = IMAGE_WIDTH
        self.img_h = IMAGE_HEIGHT

    def enumerate_videos(self, path):
        video_list = []
        for video_path in Path(path).glob('*'):
            try:
                if os.path.isfile(video_path):
                    video = Video().from_path(video_path)
                    if (video is not None):
                        if K.image_data_format() == 'channels_first' and video.frames.shape != (self.img_c,self.frame_num,self.img_w,self.img_h):
                            print("Video "+str(video_path)+" has incorrect shape "+str(video.frames.shape)+", must be "+str((self.img_c,self.frame_num,self.img_w,self.img_h))+"")
                            raise AttributeError
                        if K.image_data_format() != 'channels_first' and video.frames.shape != (self.frame_num,self.img_w,self.img_h,self.img_c):
                            print("Video "+str(video_path)+" has incorrect shape "+str(video.frames.shape)+", must be "+str((self.frame_num,self.img_w,self.img_h,self.img_c))+"")
                            raise AttributeError
                    else:
                        raise Exception
                elif os.path.isdir(video_path):
                    continue
                else:
                    raise FileNotFoundError
            except AttributeError as err:
                raise err
            except FileNotFoundError as err:
                raise err
            except Exception as e:
                print("Error loading video: "+ str(video_path))
                continue
            video_list.append(str(video_path))
        return video_list
