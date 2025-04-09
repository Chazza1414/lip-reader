import cv2 as cv
import numpy as np
from input.tracker import Tracker
from input.classifier import Classifier

web_cam_capture = cv.VideoCapture(0)

face_tracker = Tracker()
mouth_tracker = Tracker()

face_classifier = Classifier('face', 20, 2)
mouth_classifier = Classifier('mouth', 10, 0.6)

#ret, frame = web_cam_capture.read()
#frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
frame_counter = 0

while not face_classifier.confident:
    ret, frame = web_cam_capture.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    face_classifier.detect(frame_gray)

    if face_classifier.max_index == -1:
        print("No faces detected on init")

face_tracker.create(frame_gray, face_classifier.objects[face_classifier.max_index])
face_tracker.set_tracking(True)

while not mouth_classifier.confident:
    ret, frame = web_cam_capture.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face_tracker.update(frame_gray)

    if (face_tracker.track_success):
        face_roi = frame_gray[face_tracker.y+int(face_tracker.h/2):face_tracker.y+face_tracker.h, 
                              face_tracker.x:face_tracker.x+face_tracker.w]

        est_mouth_size = (int(face_tracker.w/3), int(face_tracker.h/4))

        #mouth_classifier.detect_with_size(face_roi, min_size=est_mouth_size, max_size=est_mouth_size)
        mouth_classifier.detect(face_roi)

        if mouth_classifier.max_index == -1:
            print("No mouths detected on init")

mouth_tracker.create(face_roi, mouth_classifier.objects[mouth_classifier.max_index])
mouth_tracker.set_tracking(True)

while web_cam_capture.isOpened():

    ret, frame = web_cam_capture.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    if not ret:
        print("Can't receive frame")
        break

    if cv.waitKey(1) == ord('q'):
        break

    if frame_counter % 15 == 0:
        face_classifier.check_accuracy = True
        mouth_classifier.check_accuracy = True
    
    #if face_classifier.confident:
    if face_tracker.tracking:
        face_tracker.update(frame_gray)

        if face_tracker.track_success:
            est_mouth_size = (int(face_tracker.w/3), int(face_tracker.h/4))
            cv.rectangle(
                frame, 
                (face_tracker.x, face_tracker.y), 
                (face_tracker.x + face_tracker.w, face_tracker.y + face_tracker.h), 
                (255, 0, 0), 2)
            cv.circle(
                frame, 
                (face_tracker.x,face_tracker.y), 
                radius=5, color=(0,0,255), thickness=-1)
            #face_roi = frame_gray[x:x+w, y+int(h/2):y+h]
            face_roi = frame_gray[
                face_tracker.y+int(face_tracker.h/2):face_tracker.y+face_tracker.h, 
                face_tracker.x:face_tracker.x+face_tracker.w]
    else:
        print("not face confident in loop")
        face_classifier.check_accuracy = True

    #if mouth_classifier.confident:
    if mouth_tracker.tracking:
        mouth_tracker.update(face_roi)

        if mouth_tracker.track_success:
            cv.rectangle(
                frame, 
                (face_tracker.x + mouth_tracker.x, face_tracker.y + mouth_tracker.y + int(face_tracker.h/2)), 
                (face_tracker.x + mouth_tracker.x + mouth_tracker.w, face_tracker.y + mouth_tracker.y + mouth_tracker.h + int(face_tracker.h/2)), 
                (255, 0, 0), 2)
            #print(mx, my, mw, mh)
            #print(fx, fy, fw, fh)
            cv.circle(
                frame, 
                (face_tracker.x + mouth_tracker.x, face_tracker.y + mouth_tracker.y + int(face_tracker.h/2)), 
                radius=5, color=(0,0,255), thickness=-1)
    else:
        print("not mouth confident in loop")
        mouth_classifier.check_accuracy = True

    # check face tracking
    if face_classifier.check_accuracy:
        face_classifier.detect(frame_gray)
        # if we have found a face

        if face_classifier.max_index == -1:
            print("no faces found on redetection")
            # no faces
        elif not face_classifier.confident:
            print("face not confident")
        else:
            face_tracker.set_tracking(True)
            cv.rectangle(
                frame, 
                (face_classifier.x, face_classifier.y), 
                (face_classifier.x + face_classifier.w, face_classifier.y + face_classifier.h), 
                (0, 255, 0), 2)
            face_roi = frame_gray[
                face_classifier.y + int(face_classifier.h/2):face_classifier.y + face_classifier.h, 
                face_classifier.x:face_classifier.x + face_classifier.w]
            
            if face_classifier.check_in_region(face_tracker.x, face_tracker.y, face_tracker.w, face_tracker.h):
                print("face in region")
            else:
                face_tracker.create(frame_gray, face_classifier.objects[face_classifier.max_index])
                print("re-init face tracker")

    # check mouth tracking
    if mouth_classifier.check_accuracy:
        #mouth_classifier.detect_with_size(face_roi, min_size=est_mouth_size, max_size=est_mouth_size)
        mouth_classifier.detect(face_roi)

        # if we have found a mouth
        if mouth_classifier.max_index == -1:
            print("no mouths found on redetection")
        elif not mouth_classifier.confident:
            print("mouth not confident on redetection")
        else:
            mouth_tracker.set_tracking(True)
            cv.rectangle(
                frame, 
                (face_tracker.x + mouth_classifier.x, face_tracker.y + int(face_tracker.h/2) + mouth_classifier.y), 
                (face_tracker.x + mouth_classifier.w + mouth_classifier.x, face_tracker.y + int(face_tracker.h/2)+ mouth_classifier.y + mouth_classifier.h), 
                (0, 255, 0), 2)
            
            if mouth_classifier.check_in_region(mouth_tracker.x, mouth_tracker.y, mouth_tracker.w, mouth_tracker.h):
                print("mouth in region")
            else:
                mouth_tracker.create(frame_gray, mouth_classifier.objects[mouth_classifier.max_index])
                print("re-init mouth tracker")


    cv.imshow('web cam', frame)

    frame_counter+=1
