import cv2
import numpy as np
import imageio.v3 as iio
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lipreader.videos import Video

# video_path = "H:/UNI/CS/Year3/Project/Dataset/GRID/video/s1.mpg_6000.part1/s1/video/mpg_6000/bbaf2n.mpg"
# cap = cv2.VideoCapture(video_path)

# if not cap.isOpened():
#     print("Error: Could not open video file.")
# else:
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     print(f"FPS: {fps}")
#     print(f"Total Frames: {frame_count}")

# cap.release()



# path = "H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s24_pwab2a.mp4"
# path2 = "H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s8_bwbg2s.mp4"
# path3 = "H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s8_lgaz1p.mp4"
# path4 = "H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\gray_lips\\s8_bgwn5n.mp4"
# # frames = np.array(iio.imread(path, plugin='pyav'))
# video = Video().from_path(path2)
# frames = video.frames[:60]
# #frames = frames
# print(frames.shape)
# input_path = "H:\\UNI\\CS\\Year3\\Project\\lip-reader\\video-input\\AV_Clip_Weather.mp4"
# cap = cv2.VideoCapture(input_path)

# if not cap.isOpened():
#     print("Error: Could not open video.")

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#     print(frame.shape) 

# cap.release()
# cv2.destroyAllWindows()

vid = np.zeros((1,2,3,4))
vid = np.arange(1, 10)
print(vid)
vid = np.repeat(vid, 2, axis=0)
print(vid.shape)
print(vid)
#vid = np.repeat



# vid = np.empty(shape=(75, 50, 100, 3))
# vid = np.swapaxes(vid, 1, 2)
# print(vid.shape)