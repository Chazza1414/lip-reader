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
import skvideo.io
#from scipy.misc import imresize
from scipy import ndimage
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lipreader.common.constants import VIDEO_PATH, DATASET_PATH, VALIDATION_FRACTION, IMAGE_HEIGHT, IMAGE_WIDTH, VIDEO_FRAME_NUM, MAX_NUM_VIDEOS

'''
Usage: 
$ python prepare.py

where the number of samples is the number of speakers to be reserved for validation

iterate through every video file tree
create a symlink from the video to the dataset folder
renaming the file to be 's[speaker_number]_[video_name]'
'''

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

def prepare_videos(max_num_videos):
    validation_number = int(VALIDATION_FRACTION * int(max_num_videos))
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
                            if n < validation_number:
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

def prepare_test_videos(input_video_location, output_dataset_location, max_num_videos):
    validation_number = int(VALIDATION_FRACTION * int(max_num_videos))
    n = 0

    for video_path in Path(input_video_location).glob('*'):

        speaker_id = os.path.splitext(video_path)[0].split('\\')[-1].split("_")[0]

        video_name = os.path.splitext(video_path)[0].split('\\')[-1].split("_")[1]
        
        validate_link_name = Path(output_dataset_location) / 'validate' / (speaker_id + "_" + video_name + '.mpg')
        train_link_name = Path(output_dataset_location) / 'train' / (speaker_id + "_" + video_name + '.mpg')

        if os.path.exists(validate_link_name):
            os.remove(validate_link_name)  # Remove existing link or file
        if os.path.exists(train_link_name):
            os.remove(train_link_name)  # Remove existing link or file
            
        try:
            print("writing test file")
            if n < validation_number:
                os.symlink(video_path, validate_link_name)
                # subprocess.check_output(
                #     "mklink {} {}".format(link_name, video_path3), shell=True)
            else:
                os.symlink(video_path, train_link_name)
                # subprocess.check_output(
                #     "mklink {} {}".format(link_name, video_path3), shell=True)
        
        except Exception as error:
            raise(error)
        n += 1

def get_frames_mouth(detector, predictor, frames):
    MOUTH_WIDTH = 100
    MOUTH_HEIGHT = 50
    HORIZONTAL_PAD = 0.19
    normalize_ratio = None
    mouth_frames = []
    #print("starting detection")
    for frame in frames:
        dets = detector(frame, 1)
        shape = None
        for k, d in enumerate(dets):
            shape = predictor(frame, d)
            i = -1
        if shape is None: # Detector doesn't detect face, just return as is
            return frames
        mouth_points = []
        for part in shape.parts():
            i += 1
            if i < 48: # Only take mouth region
                continue
            mouth_points.append((part.x,part.y))
        np_mouth_points = np.array(mouth_points)

        mouth_centroid = np.mean(np_mouth_points[:, -2:], axis=0)

        # if normalize_ratio is None:
        #     mouth_left = np.min(np_mouth_points[:, :-1]) * (1.0 - HORIZONTAL_PAD)
        #     mouth_right = np.max(np_mouth_points[:, :-1]) * (1.0 + HORIZONTAL_PAD)

        #     normalize_ratio = MOUTH_WIDTH / float(mouth_right - mouth_left)

        #new_img_shape = (int(frame.shape[0] * normalize_ratio), int(frame.shape[1] * normalize_ratio))
        #resized_img = cv.resize(frame, new_img_shape, interpolation=cv.INTER_LINEAR)
        #resized_img = frame
        #print(resized_img.shape)
        #resized_img = imresize(frame, new_img_shape)

        #mouth_centroid_norm = mouth_centroid * normalize_ratio

        # mouth_l = int(mouth_centroid_norm[0] - MOUTH_WIDTH / 2)
        # mouth_r = int(mouth_centroid_norm[0] + MOUTH_WIDTH / 2)
        # mouth_t = int(mouth_centroid_norm[1] - MOUTH_HEIGHT / 2)
        # mouth_b = int(mouth_centroid_norm[1] + MOUTH_HEIGHT / 2)

        mouth_l = int(mouth_centroid[0] - MOUTH_WIDTH / 2)
        mouth_r = int(mouth_centroid[0] + MOUTH_WIDTH / 2)
        mouth_t = int(mouth_centroid[1] - MOUTH_HEIGHT / 2)
        mouth_b = int(mouth_centroid[1] + MOUTH_HEIGHT / 2)

        # if frame is not None:
        #     x = int(mouth_centroid[0])
        #     y = int(mouth_centroid[1])
        #     # x = int(mouth_centroid_norm[0])
        #     # y = int(mouth_centroid_norm[1])
        #     cv.circle(frame, (x, y), 5, color=(0, 255, 0), thickness=-1)
        #     cv.rectangle(frame, (mouth_l, mouth_t), (mouth_r, mouth_b), color=(0,0,255), thickness=2)
        #     cv.imshow("Video", frame)
        #     if cv.waitKey(30) & 0xFF == ord('q'):  # Press 'q' to exit
        #         break
        #print(mouth_centroid)

        mouth_crop_image = frame[mouth_t:mouth_b, mouth_l:mouth_r]
        #print(mouth_t, mouth_b, mouth_l, mouth_r, frame.shape, mouth_crop_image.shape)
        #mouth_crop_image = resized_img[mouth_l:mouth_r, mouth_t:mouth_b]

        #mouth_frames.append(mouth_crop_image)
        mouth_frames.append(mouth_crop_image)
    return mouth_frames

def get_video_frames(path):
    videogen = skvideo.io.vreader(path)
    frames = np.array([cv.cvtColor(frame, cv.COLOR_RGB2GRAY) for frame in videogen])
    return frames

def show_frames(frames, delay=30):
    """Display a list of frames (NumPy arrays) sequentially using OpenCV.

    Args:
        frames (list of np.ndarray): List of frames (3D NumPy arrays).
        delay (int): Delay in milliseconds between frames.
    """
    for frame in frames:
        if frame is not None:
            cv.imshow("Video", frame)
            if cv.waitKey(delay) & 0xFF == ord('q'):  # Press 'q' to exit
                break
    cv.destroyAllWindows()

def write_video(frames, frame_width, frame_height, location):
    fps = 25
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format

    # Create video writer
    out = cv.VideoWriter(location, fourcc, fps, (frame_width, frame_height))

    for frame in frames:
        out.write(cv.cvtColor(frame, cv.COLOR_GRAY2BGR))  # Write frame to video

    out.release()

vid_path = "H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\video\\s6.mpg_6000.part2\\s6\\video\\mpg_6000\\pbab5n.mpg"


face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor("./training/predictors/shape_predictor_68_face_landmarks.dat")

# prepare videos normally
#prepare_videos(VIDEO_PATH, DATASET_PATH, MAX_NUM_VIDEOS)

# prepare test subset
#prepare_videos("H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\test_video", "H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\test_dataset", 10)
prepare_test_videos("H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\test_video", "H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\test_datasets", 10)