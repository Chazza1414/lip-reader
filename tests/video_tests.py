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

# vid = np.zeros((1,2,3,4))
# vid = np.arange(1, 10)
# print(vid)
# vid = np.repeat(vid, 2, axis=0)
# print(vid.shape)
# print(vid)
#vid = np.repeat

video = Video().from_path("H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\datasets\\train\\s5_bgaa3s.mpg")
#vid_copy = np.copy(video.frames)
video.flip_video()
print(video.frames.shape)
#print(video.frames / 255)
#print(np.all(vid_copy == np.flip(video.frames, axis=1)))
video.frames = np.swapaxes(video.frames, 1, 2)

frame_height, frame_width = 50, 100
fps=25

delay = int(1000 / fps)  # Convert FPS to milliseconds per frame

for frame in video.frames:
    #corrected_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    cv2.imshow("Video Playback", frame)  # Show frame
    if cv2.waitKey(delay) & 0xFF == ord('q'):  # Press 'q' to exit early
        break

cv2.destroyAllWindows()  # Close window when done

# Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4
# out = cv2.VideoWriter("H:\\UNI\\CS\\Year3\\Project\\lip-reader\\video-input\\flipped.mp4", fourcc, fps, (frame_height, frame_width))

# Write each frame
# for frame in video.frames:
    
#     out.write(frame)

# Release the writer
# out.release()

# vid = np.empty(shape=(75, 50, 100, 3))
# vid = np.swapaxes(vid, 1, 2)
# print(vid.shape)