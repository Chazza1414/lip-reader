import cv2 as cv
import imageio.v3 as iio
import numpy as np
import dlib
import keras.backend as K
from lipreader.common.constants import IMAGE_HEIGHT, IMAGE_WIDTH

class Video(object):
    def __init__(self):
        self.frames = np.array([])
        #self.face_detector = dlib.get_frontal_face_detector()
        #self.landmark_detector = dlib.shape_predictor("../training/predictors/shape_predictor_68_face_landmarks.dat")
        #self.path = path
        self.lip_frames = np.array([])
        #self.identify_face()

    def from_path(self, path):
        self.frames = np.array(iio.imread(path, plugin='pyav'))
        width = self.frames.shape[1]
        height = self.frames.shape[2]

        if (width == IMAGE_HEIGHT and height == IMAGE_WIDTH):
            self.frames = np.swapaxes(self.frames, 1, 2)

        return self

    def flip_video(self):
        if (self.frames != []):
            self.frames = np.flip(self.frames, axis=2)

    # def identify_face(self):
    #     capture = cv.VideoCapture(self.path)

    #     if not capture.isOpened():
    #         print("Error: Could not open video.")
    #         raise ValueError
        
    #     while capture.isOpened():

    #         ret, frame = capture.read()
    #         frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    #         #frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #         #frame = dlib.load_rgb_image(frame)
    #         #frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

    #         if not ret:
    #             print("Can't receive frame")
    #             raise ValueError

    #         dets = self.face_detector(frame)

    #         for i, d in enumerate(dets):
    #             face = self.landmark_detector(frame, d)

    #             section = face.part(51).y - face.part(33).y
    #             # draw_rectangle = dlib.rectangle(left=face.part(33).x - (section * 3), 
    #             #                                 top=face.part(33).y, 
    #             #                                 right=face.part(33).x + (section * 3), 
    #             #                                 bottom=face.part(33).y + (section * 3)) 
    #             # win.add_overlay(draw_rectangle, dlib.rgb_pixel(0,0,255))

    #             left = face.part(33).x - (section * 3)
    #             right = face.part(33).x + (section * 3)
    #             top = face.part(33).y
    #             bottom = face.part(33).y + (section * 3)

    #             if (K.image_data_format() == 'channels_last'):
    #                 lips = frame[left:right, top:bottom]
    #                 rescaled_lips = cv.resize(lips, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv.INTER_LINEAR)
    #                 rescaled_lips = rescaled_lips.swapaxes(0,1)
    #                 #print(rescaled_lips.shape)
    #                 self.lip_frames = np.append(self.lip_frames, [np.array(rescaled_lips)], axis=0)
    #                 #print(np.array(frame).shape)

#vid = Video("H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\video\\s1.mpg_6000.part1\\s1\\video\\mpg_6000\\bbaf2n.mpg")