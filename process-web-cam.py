import cv2 as cv
import numpy as np

web_cam_capture = cv.VideoCapture(0)

face_classififer = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)
mouth_classifier = cv.CascadeClassifier(
    cv.data.haarcascades + 'haarcascade_smile.xml'
)
tracker = cv.TrackerMIL.create()

ret, frame = web_cam_capture.read()
frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
faces = []
mouths = []

while len(faces) == 0:
    faces, rejectLevels, levelWeights = face_classififer.detectMultiScale3(frame, outputRejectLevels=True)
    print("finding face " + str(len(faces)))

max_index = np.argmax(levelWeights)
print(max_index, levelWeights)
(fx, fy, fw, fh) = faces[max_index]
print(faces[max_index])

# face_roi = frame[fy:fy+fh, fx:fx+fw]

# while len(mouths) == 0:
#     mouths, rejectLevels, levelWeights = mouth_classifier.detectMultiScale3(face_roi, outputRejectLevels=True)
#     print("finding mouth " + str(len(mouths)))

# if len(mouths) > 1:
    

#     cv.rectangle(frame, (mouths[0][0], mouths[0][1]), (mouths[0][0] + mouths[0][2], mouths[0][1] + mouths[0][3]), (255,0,0), 2)
#     cv.imshow('web cam', frame)

# print(levelWeights)
# max_index = np.argmax(levelWeights)
# print(levelWeights[max_index])

# (mx, my, mw, mh) = mouths[max_index]
tracker.init(frame, faces[0])

while web_cam_capture.isOpened():

    if not ret:
        print("Can't receive frame")
        break

    
    
    success, bbox = tracker.update(frame)

    if success:
        x, y, w, h = [int(v) for v in bbox]
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_roi = frame[x:x+w, y+int(h/2):y+h]
        mouths, rejectLevels, levelWeights = mouth_classifier.detectMultiScale3(face_roi, outputRejectLevels=True)
        if len(mouths) > 0:
            (mx, my, mw, mh) = mouths[0]
            print((mx, my, mw, mh))
            cv.rectangle(frame, (mx+x, my+y), (mx + x + mw, my + y + mh), (0, 255, 0), 2)


    #cv.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)

    cv.imshow('web cam', frame)

    #print(gray_frame.shape)
    if cv.waitKey(1) == ord('q'):
        break

    ret, frame = web_cam_capture.read()
