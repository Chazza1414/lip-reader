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
from lipreader.common.constants import VIDEO_PATH, DATASET_PATH, VALIDATION_FRACTION, IMAGE_HEIGHT, IMAGE_WIDTH, VIDEO_FRAME_NUM, LIPS_PATH
import concurrent
from concurrent.futures import ThreadPoolExecutor
import threading

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



class LipExtractor():
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.api_counter = 0
        self.lock = threading.Lock()
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_detector = dlib.shape_predictor("./training/predictors/shape_predictor_68_face_landmarks.dat")

    def increment_api_counter(self):
        with self.lock:
            self.api_counter += 1
            print("Calls: " + str(self.api_counter))

    def extract_lips(self, video_path, lip_vid_name):
        frames = self.get_video_frames(str(video_path))

        lip_frames = self.get_frames_mouth(frames)

        if os.path.exists(lip_vid_name):
            os.remove(lip_vid_name)

        self.write_video(lip_frames, lip_vid_name)

        self.increment_api_counter()

    def prepare_specific_videos(self, paths):
        tasks = []

        for path in paths:

            speaker_id = os.path.splitext(path)[0].split('\\')[-1].split("_")[0]
            video_name = os.path.splitext(path)[0].split('\\')[-1].split("_")[1]

            video_path1 = Path(VIDEO_PATH) / (str(speaker_id) + '.mpg_6000.part1') / str(speaker_id) / str('video') / str('mpg_6000') / (str(video_name) + '.mpg')
            video_path2 = Path(VIDEO_PATH) / (str(speaker_id) + '.mpg_6000.part2') / str(speaker_id) / str('video') / str('mpg_6000') / (str(video_name) + '.mpg')
            
            if os.path.exists(video_path1):
                tasks.append((video_path1, path))
            elif os.path.exists(video_path2):
                tasks.append((video_path2, path))
            else:
                print("can't find file")
            
        with self.executor as executor:  # Ensure proper shutdown

            futures = []
            for (video_path, lip_name) in tasks:
                
                futures.append(executor.submit(self.extract_lips, video_path=video_path, lip_vid_name=lip_name))

            #print(futures[0])
            print("starting execution")
            print(len(tasks), len(futures))

            for future in concurrent.futures.as_completed(futures):
                future.result()

    def prepare_videos(self):

        tasks = []
        for compressed_path in [d for d in Path(VIDEO_PATH).glob('*')]:
            for speaker_path in Path(compressed_path).glob('*'):
                speaker_id = os.path.splitext(speaker_path)[0].split('\\')[-1]

                for video_path1 in Path(speaker_path).glob('*'):
                    for video_path2 in Path(video_path1).glob('*'):
                        for video_path3 in Path(video_path2).glob('*'):

                            video_name = os.path.splitext(video_path3)[0].split('\\')[-1]

                            lip_vid_name = Path(LIPS_PATH) / (speaker_id + "_" + video_name + '.mp4')

                            tasks.append((video_path3, lip_vid_name))

                            # frames = self.get_video_frames(str(video_path3))

                            # lip_frames = self.get_frames_mouth(frames)

                            # self.write_video(lip_frames, lip_vid_name)

                            # print(n)
                            # n += 1
        
        with self.executor as executor:  # Ensure proper shutdown

            futures = []
            for (video_path, lip_name) in tasks:
                
                futures.append(executor.submit(self.extract_lips, video_path=video_path, lip_vid_name=lip_name))

            #print(futures[0])
            print("starting execution")
            print(len(tasks), len(futures))

            for future in concurrent.futures.as_completed(futures):
                future.result()

    def get_frames_mouth(self, frames):
        mouth_frames = []
        #print("starting detection")
        for frame in frames:
            dets = self.face_detector(frame, 1)
            shape = None
            for k, d in enumerate(dets):
                shape = self.landmark_detector(frame, d)
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

            mouth_l = int(mouth_centroid[0] - IMAGE_WIDTH / 2)
            mouth_r = int(mouth_centroid[0] + IMAGE_WIDTH / 2)
            mouth_t = int(mouth_centroid[1] - IMAGE_HEIGHT / 2)
            mouth_b = int(mouth_centroid[1] + IMAGE_HEIGHT / 2)

            mouth_crop_image = frame[mouth_t:mouth_b, mouth_l:mouth_r]

            mouth_frames.append(mouth_crop_image)
        return mouth_frames

    def get_video_frames(self, path):
        videogen = skvideo.io.vreader(path)
        frames = np.array([cv.cvtColor(frame, cv.COLOR_RGB2GRAY) for frame in videogen])
        return frames

    def write_video(self, frames, location):
        fps = 25
        fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format

        # Create video writer
        out = cv.VideoWriter(location, fourcc, fps, (IMAGE_WIDTH, IMAGE_HEIGHT))

        for frame in frames:
            out.write(cv.cvtColor(frame, cv.COLOR_GRAY2BGR))  # Write frame to video

        out.release()

lipEx = LipExtractor()

#lipEx.prepare_videos()
paths = ["H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s1_bbizzn.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s1_brwg8p.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s1_bwwuzn.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s1_lbad6n.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s1_lgbf8n.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s1_lrarzn.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s1_pbio6p.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s1_pbio7a.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s1_pbwx1s.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s2_pbwxzs.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s2_pwbd6s.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s3_bgbn9a.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s3_bgit2n.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s3_lbij5s.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s3_lgbz9s.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s3_lrbr3s.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s3_pbiu6n.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s3_pgwy7s.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s3_swiu2n.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s4_lwiy3n.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s4_sgbp1n.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s4_swwc1n.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s5_lbbk1s.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s6_bgba7p.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s6_bgbg9n.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s6_bgwu2s.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s6_sbwu2a.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s6_sran4s.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s6_srig6s.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s6_swwc1p.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s7_bbir1s.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s7_bbir2p.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s7_bgam9s.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s7_lgws5s.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s7_lrwe6p.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s7_lwak8p.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s7_pwii6n.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s7_sgio1s.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s7_sgio2p.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s8_bgitza.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s8_bgwa7n.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s8_bgwa8s.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s9_bwaf6n.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s11_pwac3s.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s15_bgam1s.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s15_bgil9a.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s15_brwf4p.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s15_bwaf3a.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s15_lbio7s.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s15_lbwd2p.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s15_lgwl5a.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s15_lrax1s.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s15_lwij4n.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s15_pbbh7a.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s15_pbin1s.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s15_pgbc9s.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s15_prih5a.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s15_pwbv7a.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s15_pwib5s.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s15_sbaf6p.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s15_sbal8n.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s15_sbiz5a.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s15_sgia7a.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s15_sgig8n.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s15_sgwi2p.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s15_swbh3a.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s17_lbib9a.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s22_lwbd3p.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s22_sbiy6s.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s24_brbq8s.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s24_brbrza.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s24_pbim1n.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s24_pgah9n.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s24_pgwi8s.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s24_prag8s.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s24_pwab2a.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s24_pwwi4a.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s24_sgbhza.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s24_sgwh1n.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s24_srwg1p.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s24_swwa4a.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s28_bwiq2s.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s28_bwix6s.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s28_bwwy7n.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s28_lbig9n.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s28_lgbd3p.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s28_lrab7p.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s28_lwbv9p.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s28_lwwd2a.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s28_pbif6a.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s28_sbwf1p.mp4",
"H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s28_sbwf2a.mp4",]
lipEx.prepare_specific_videos(paths)

cv.destroyAllWindows()