import numpy as np
import cv2 as cv
import dlib
import matplotlib.pyplot as plt
from phoneme_library import PhonemeLibrary

VIDEO_PATH = 'id2_6000_swwp2s.mpg'
VIDEO_PATH = "H:/UNI/CS/Year3/Project/Dataset/GRID/s23.mpg_6000.part1/s23/video/mpg_6000/bbad1s.mpg"

cap = cv.VideoCapture(VIDEO_PATH)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0, 255, (100, 3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
# drawing mask
mask = np.zeros_like(old_frame)

frontal_face_detector = dlib.get_frontal_face_detector()
face_landmark_detector = dlib.shape_predictor("../../predictors/shape_predictor_68_face_landmarks.dat")

dets = frontal_face_detector(old_gray)

for i, d in enumerate(dets):
    face = face_landmark_detector(old_gray, d)

    section = face.part(51).y - face.part(33).y
    draw_rectangle = dlib.rectangle(left=face.part(33).x - (section * 3), 
                                    top=face.part(33).y, 
                                    right=face.part(33).x + (section * 3), 
                                    bottom=face.part(33).y + (section * 3))

feature_mask = np.zeros_like(old_gray, dtype=np.uint8)

# Define a region of interest (ROI) in the mask
#feature_mask[50:200, 50:200] = 255  # ROI is now white, rest is black
#feature_mask[face.part(33).x - (section * 3):face.part(33).x + (section * 3), face.part(33).y:face.part(33).y + (section * 3)] = 255  # ROI is now white, rest is black
feature_mask[face.part(33).y:face.part(33).y + (section * 3), face.part(33).x - (section * 3):face.part(33).x + (section * 3)] = 255

p0 = cv.goodFeaturesToTrack(old_gray, mask = feature_mask, **feature_params)

motion = [0]

while(1):
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    


    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]

    frame_motion = 0
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        frame_motion += np.sqrt(np.square(a - c) + np.square(b - d))
    
    motion.append(motion[-1] + frame_motion)
    img = cv.add(frame, mask)
    cv.imshow('frame', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
cv.destroyAllWindows()

fig, ax = plt.subplots()

TRANS_FILE_NAME = 'H:/UNI/CS/Year3/Project/Dataset/GRID/s23/align/bbad1s.align'
PhonLib = PhonemeLibrary()
transcription_array = PhonLib.create_transcription_array(TRANS_FILE_NAME, 25)

time_labels = [pair[0] for pair in transcription_array]
word_labels = [pair[2] for pair in transcription_array]

ax.set_xticks(time_labels)
ax.set_xticklabels(word_labels)

ax.scatter(np.arange(len(motion))/25, motion)
ax.vlines([pair[0] for pair in transcription_array], 
colors='black', ymin=0, ymax=motion[-1])


plt.show()