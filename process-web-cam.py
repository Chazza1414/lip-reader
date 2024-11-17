import cv2 as cv
import numpy as np

web_cam_capture = cv.VideoCapture(0)

face_classifier = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)
mouth_classifier = cv.CascadeClassifier(
    cv.data.haarcascades + 'haarcascade_smile.xml'
)

face_tracker = cv.TrackerMIL.create()
mouth_tracker = cv.TrackerMIL.create() 

ret, frame = web_cam_capture.read()
frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
faces = []
mouths = []
face_pixel_tolerance = 20
mouth_pixel_tolerance = 10
face_confidence_min = 2
mouth_confidence_min = 0.8
face_confident = False
mouth_confident = False
frame_counter = 0

while not face_confident:
    faces, rejectLevels, levelWeights = face_classifier.detectMultiScale3(frame_gray, outputRejectLevels=True)
    print("finding face " + str(len(faces)))
    if len(faces) > 0:
        max_index = np.argmax(levelWeights)
        print(levelWeights[max_index])
        if levelWeights[max_index] > face_confidence_min:
            face_tracker.init(frame_gray, faces[max_index])
            face_confident = True
        else:
            ret, frame = web_cam_capture.read()
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    else:
        ret, frame = web_cam_capture.read()
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

ret, frame = web_cam_capture.read()
frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
face_track_success, face_bbox = face_tracker.update(frame_gray)

while not mouth_confident:
    if (face_track_success):
        (fx, fy, fw, fh) = face_bbox
        face_roi = frame_gray[fy+int(fh/2):fy+fh, fx:fx+fw]

        mouths, rejectLevels, levelWeights = mouth_classifier.detectMultiScale3(
        face_roi, outputRejectLevels=True)
        print("finding mouth " + str(len(mouths)))
        
        if len(mouths) > 0:
            max_index = np.argmax(levelWeights)
            print(levelWeights[max_index])
            if levelWeights[max_index] > mouth_confidence_min:
                #(mx,my,mw,mh) = (mouths[max_index][0] + fx, mouths[max_index][1] + fy, mouths[max_index][2], mouths[max_index][3])
                mouth_tracker.init(face_roi, mouths[max_index])
                
                mouth_confident = True
            else:
                ret, frame = web_cam_capture.read()
                frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                face_track_success, face_bbox = face_tracker.update(frame_gray)
        else:
            ret, frame = web_cam_capture.read()
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            face_track_success, face_bbox = face_tracker.update(frame_gray)

    else:
        ret, frame = web_cam_capture.read()
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        face_track_success, face_bbox = face_tracker.update(frame_gray)

# (fx, fy, fw, fh) = faces[max_index]
# face_roi = frame_gray[fy+int(fh/2):fy+fh, fx:fx+fw]
# max_mouth_size = (int(fw/2), int(fh/4))
# min_mouth_size = (int(fw/4), int(fh/8))
# cv.imshow("test", face_roi)
# while not mouth_confident:
#     mouths, rejectLevels, levelWeights = mouth_classifier.detectMultiScale3(
#         face_roi, outputRejectLevels=True)
#     print("finding mouth " + str(len(mouths)))
    
#     if len(mouths) > 0:
#         max_index = np.argmax(levelWeights)
#         print(levelWeights[max_index])
#         if levelWeights[max_index] > mouth_confidence_min:
#             #(mx,my,mw,mh) = (mouths[max_index][0] + fx, mouths[max_index][1] + fy, mouths[max_index][2], mouths[max_index][3])
#             mouth_tracker.init(frame_gray, mouths[max_index])
            
#             mouth_confident = True

#print(max_index, levelWeights)

#print(faces[max_index])

while web_cam_capture.isOpened():

    if not ret:
        print("Can't receive frame")
        break
    
    if face_confident:
        face_track_success, face_bbox = face_tracker.update(frame_gray)

        if face_track_success:
            fx, fy, fw, fh = [int(v) for v in face_bbox]
            cv.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
            cv.circle(frame, (fx,fy), radius=5, color=(0,0,255), thickness=-1)
            #face_roi = frame_gray[x:x+w, y+int(h/2):y+h]
            face_roi = frame_gray[fy+int(fh/2):fy+fh, fx:fx+fw]
            #print(face_roi.shape)
            #cv.imshow("cam", face_roi)
            #cv.rectangle(frame, (x, y + int(h/2)), (x + w, y + h), (150, 150, 150), 2)
            
            # max_mouth_size = (int(w/2), int(h/4))
            # min_mouth_size = (int(w/4), int(h/8))
            # mouths, rejectLevels, levelWeights = mouth_classifier.detectMultiScale3(
            #     face_roi, maxSize=max_mouth_size, minSize=min_mouth_size, outputRejectLevels=True)
            # if len(mouths) > 0:
            #     max_index = np.argmax(levelWeights)
            #     (mx, my, mw, mh) = mouths[max_index]
            #     print((mx, my, mw, mh))
            #     cv.rectangle(frame, (mx+x, my+y+int(h/2)), (mx + x + mw, my + y + mh+int(h/2)), (0, 255, 0), 2)
            #     cv.circle(frame, (mx+x,my+y+int(h/2)), radius=5, color=(0,0,255), thickness=-1)
    else:
        print("not face confident l")

    if mouth_confident:
        mouth_track_success, mouth_bbox = mouth_tracker.update(face_roi)

        if mouth_track_success:
            print("tracking mouth")
            mx, my, mw, mh = [int(v) for v in mouth_bbox]
            #print((mx+fx, my+fy+int(fh/2)), (mx + fx + mw, my + fy + mh+int(fh/2)))
            #cv.rectangle(frame, (mx+fx, my+fy+int(fh/2)), (mx + fx + mw, my + fy + mh+int(fh/2)), (0, 255, 0), 2)
            cv.rectangle(frame, (fx + mx, fy + my + int(fh/2)), (fx + mx + mw, fy + my + mh + int(fh/2)), (255, 0, 0), 2)
            print(mx, my, mw, mh)
            print(fx, fy, fw, fh)
            cv.circle(frame, (fx + mx, fy + my + int(fh/2)), radius=5, color=(0,0,255), thickness=-1)
            
            #face_roi = frame_gray[x:x+w, y+int(h/2):y+h]
            #face_roi = frame_gray[y+int(h/2):y+h, x:x+w]

    # check face tracking
    if frame_counter % 15 == 0:
        faces, rejectLevels, levelWeights = face_classifier.detectMultiScale3(frame_gray, outputRejectLevels=True)
        # if we have found a face
        if not len(faces) == 0:
            # get the most confident face
            max_index = np.argmax(levelWeights)
            print(faces[max_index])
            # if we are confident enough with the detected face
            if levelWeights[max_index] > face_confidence_min:
                face_confident = True

                (fx, fy, fw, fh) = faces[max_index]
                cv.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
                face_roi = frame_gray[fy+int(fh/2):fy+fh, fx:fx+fw]
                
                # if redetected face is not within tracked region
                if not (
                    face_bbox[0]-face_pixel_tolerance < faces[max_index][0] and 
                    face_bbox[1]-face_pixel_tolerance < faces[max_index][1] and
                    (face_bbox[0] + face_bbox[2] + face_pixel_tolerance) > (fx + fw) and 
                    (face_bbox[1] + face_bbox[3] + face_pixel_tolerance) > (fy + fh)):

                    face_tracker.init(frame_gray, faces[max_index])
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

    # check mouth tracking
    if frame_counter % 15 == 0:
        mouths, rejectLevels, levelWeights = mouth_classifier.detectMultiScale3(face_roi, outputRejectLevels=True)
        # if we have found a face
        if not len(mouths) == 0:
            # get the most confident face
            max_index = np.argmax(levelWeights)
            print(mouths[max_index])
            # if we are confident enough with the detected face
            if levelWeights[max_index] > mouth_confidence_min:
                mouth_confident = True

                (mx, my, mw, mh) = mouths[max_index]
                cv.rectangle(frame, (mx, my), (mx + mw, my + mh), (0, 255, 0), 2)
                
                # if redetected face is not within tracked region
                if not (
                    mouth_bbox[0]-mouth_pixel_tolerance < mx and 
                    mouth_bbox[1]-mouth_pixel_tolerance < my and
                    (mouth_bbox[0] + mouth_bbox[2] + mouth_pixel_tolerance) > (mx + mw) and 
                    (mouth_bbox[1] + mouth_bbox[3] + mouth_pixel_tolerance) > (my + mh)):

                    mouth_tracker.init(face_roi, mouths[max_index])
                    print("mouth not in region")

                else:
                    print("mouth in region")
                    print(levelWeights[max_index])
            else:
                print("mouth not confident")
                mouth_confident = False
                frame_counter-=1
        else:
            print("no mouths found on redetection")
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
