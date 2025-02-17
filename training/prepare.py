import os
import glob
import subprocess
import sys
import tarfile
import dlib
import cv2 as cv
from pathlib import Path
import keras.backend as K
import numpy as np
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lipreader.common.constants import VIDEO_PATH, DATASET_PATH, VALIDATION_SAMPLES, IMAGE_HEIGHT, IMAGE_WIDTH, VIDEO_FRAME_NUM

'''
Usage: 
$ python prepare.py

where the number of samples is the number of speakers to be reserved for validation

iterate through every video file tree
create a symlink from the video to the dataset folder
renaming the file to be 's[speaker_number]_[video_name]'
'''

face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor("./training/predictors/shape_predictor_68_face_landmarks.dat")

def identify_face(path):
    capture = cv.VideoCapture(path)
    lip_frames = np.empty((VIDEO_FRAME_NUM, IMAGE_WIDTH, IMAGE_HEIGHT))

    if not capture.isOpened():
        print("Error: Could not open video.")
        raise ValueError
    
    frame_count = 0
    while capture.isOpened():

        ret, frame = capture.read()

        if (frame is None): break
        
        frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        #frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #frame = dlib.load_rgb_image(frame)
        #frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        if not ret:
            print("Can't receive frame")
            raise ValueError

        #(_, d) = next(enumerate(face_detector(frame)))
        dets = face_detector(frame)

        for i, d in enumerate(dets):
            face = landmark_detector(frame, d)

            section = face.part(51).y - face.part(33).y
            # draw_rectangle = dlib.rectangle(left=face.part(33).x - (section * 3), 
            #                                 top=face.part(33).y, 
            #                                 right=face.part(33).x + (section * 3), 
            #                                 bottom=face.part(33).y + (section * 3)) 
            # win.add_overlay(draw_rectangle, dlib.rgb_pixel(0,0,255))

            left = face.part(33).x - (section * 3)
            right = face.part(33).x + (section * 3)
            top = face.part(33).y
            bottom = face.part(33).y + (section * 3)

            if (K.image_data_format() == 'channels_last'):
                lips = frame[left:right, top:bottom]
                rescaled_lips = cv.resize(lips, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv.INTER_LINEAR)
                rescaled_lips = rescaled_lips.swapaxes(0,1)
                #print(rescaled_lips.shape)
                lip_frames[frame_count] = np.array(rescaled_lips)
                #print(np.array(frame).shape)

            frame_count += 1
    return lip_frames

def prepare_videos():
    n = 0
    for compressed_path in [d for d in Path(VIDEO_PATH).glob('*')]:
    #for compressed_path in glob.glob(os.path.join(VIDEO_PATH, '*')):
        print(os.path.splitext(compressed_path)[0].split('\\')[-1])
        for speaker_path in Path(compressed_path).glob('*'):
            speaker_id = os.path.splitext(speaker_path)[0].split('\\')[-1]

            for video_path1 in Path(speaker_path).glob('*'):
                for video_path2 in Path(video_path1).glob('*'):
                    for video_path3 in Path(video_path2).glob('*'):

                        video_name = os.path.splitext(video_path3)[0].split('\\')[-1]
                        # does this include the extension - no

                        # print(video_path3, os.path.join(DATASET_PATH, 'validate', speaker_id + "_" + video_name))
                        # print(speaker_id)
                        #print()
                        # link_name = Path(DATASET_PATH) / 'validate' / (speaker_id + "_" + video_name + '.mpg')
                        # print(link_name)

                        validate_link_name = Path(DATASET_PATH) / 'validate' / (speaker_id + "_" + video_name + '.mpg')
                        train_link_name = Path(DATASET_PATH) / 'train' / (speaker_id + "_" + video_name + '.mpg')

                        if os.path.exists(validate_link_name):
                            os.remove(validate_link_name)  # Remove existing link or file
                        if os.path.exists(train_link_name):
                            os.remove(train_link_name)  # Remove existing link or file
                            
                        try:
                            if n < VALIDATION_SAMPLES:
                                os.symlink(video_path3, validate_link_name)
                                # subprocess.check_output(
                                #     "mklink {} {}".format(link_name, video_path3), shell=True)
                            else:
                                os.symlink(video_path3, train_link_name)
                                # subprocess.check_output(
                                #     "mklink {} {}".format(link_name, video_path3), shell=True)
                        
                        except Exception as error:
                            raise(error)
                        n += 1


frames = identify_face("H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\video\\s6.mpg_6000.part2\\s6\\video\\mpg_6000\\pbab5n.mpg")
# print(frames.shape)
win = dlib.image_window()
for frame in frames:
    win.clear_overlay()
    win.set_image(frame)
    time.sleep(0.1)

cv.destroyAllWindows()