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
frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
faces = []
mouths = []
pixel_tolerance = 20
confidence_min = 2
face_confident = False
frame_counter = 0

while not face_confident:
    faces, rejectLevels, levelWeights = face_classififer.detectMultiScale3(frame_gray, outputRejectLevels=True)
    print("finding face " + str(len(faces)))
    if len(faces) > 0:
        max_index = np.argmax(levelWeights)
        if levelWeights[max_index] > confidence_min:
            face_confident = True

print(max_index, levelWeights)

print(faces[max_index])

tracker.init(frame_gray, faces[max_index])

while web_cam_capture.isOpened():

    if not ret:
        print("Can't receive frame")
        break
    
    if face_confident:
        success, bbox = tracker.update(frame_gray)

        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_roi = frame_gray[x:x+w, y+int(h/2):y+h]
            mouths, rejectLevels, levelWeights = mouth_classifier.detectMultiScale3(face_roi, outputRejectLevels=True)
            if len(mouths) > 0:
                (mx, my, mw, mh) = mouths[0]
                print((mx, my, mw, mh))
                cv.rectangle(frame, (mx+x, my+y), (mx + x + mw, my + y + mh), (0, 255, 0), 2)


    if frame_counter % 30 == 0:
        faces, rejectLevels, levelWeights = face_classififer.detectMultiScale3(frame_gray, outputRejectLevels=True)
        # if we have found a face
        if not len(faces) == 0:
            # get the most confident face
            max_index = np.argmax(levelWeights)
            print(faces[max_index])
            # if we are confident enough with the detected face
            if levelWeights[max_index] > confidence_min:
                face_confident = True
                cv.rectangle(frame, (faces[max_index][0], faces[max_index][1]), 
                (faces[max_index][0] + faces[max_index][2], faces[max_index][1] + faces[max_index][3]), (0, 255, 0), 2)
                
                # if redetected face is not within tracked region
                if not (
                    bbox[0]-pixel_tolerance < faces[max_index][0] and 
                    bbox[1]-pixel_tolerance < faces[max_index][1] and
                    (bbox[0] + bbox[2] + pixel_tolerance) > (faces[max_index][0] + faces[max_index][2]) and 
                    (bbox[1] + bbox[3] + pixel_tolerance) > (faces[max_index][1] + faces[max_index][3])):

                    tracker.init(frame_gray, faces[max_index])
                    print("not in region")

                else:
                    print("in region")
                    print(levelWeights)
            else:
                print("not confident")
                face_confident = False
                frame_counter-=1
        else:
            print("no faces found on redetection")
            face_confident = False
            frame_counter-=1

    #cv.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)

    cv.imshow('web cam', frame)

    #print(gray_frame.shape)
    if cv.waitKey(1) == ord('q'):
        break
    
    frame_counter+=1
    ret, frame = web_cam_capture.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
