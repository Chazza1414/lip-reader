import cv2
import numpy as np

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

vid = np.empty(shape=(75, 50, 100, 3))
vid = np.swapaxes(vid, 1, 2)
print(vid.shape)