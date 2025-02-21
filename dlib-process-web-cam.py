import dlib
import cv2 as cv
import keyboard
import numpy as np
import keras.backend as K
from evaluation.predict import predict

recording = False
record_frames = np.empty((0,100,50,3))
#print(record_frames.shape)

frontal_face_detector = dlib.get_frontal_face_detector()
face_landmark_detector = dlib.shape_predictor("../predictors/shape_predictor_68_face_landmarks.dat")

web_cam_capture = cv.VideoCapture(0)
win = dlib.image_window()

margin=10
mouth_box_width = 60
mouth_box_height = 35

width_scale = 0.75
height_scale = 0.75

do_prediction = False

while web_cam_capture.isOpened():

    ret, frame = web_cam_capture.read()
    #frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #frame = dlib.load_rgb_image(frame)
    #frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    print(frame.shape)

    

    if not ret:
        print("Can't receive frame")
        break

    if keyboard.is_pressed('q'):
        break

    if keyboard.is_pressed('p'):
        do_prediction = True
        break

    if keyboard.is_pressed('r'):
        recording = not recording
        if (recording):
            print("recording")
        else:
            print("recording stopped")

    dets = frontal_face_detector(frame)

    for i, d in enumerate(dets):
        face = face_landmark_detector(frame, d)

        section = face.part(51).y - face.part(33).y
        draw_rectangle = dlib.rectangle(left=face.part(33).x - (section * 3), 
                                        top=face.part(33).y, 
                                        right=face.part(33).x + (section * 3), 
                                        bottom=face.part(33).y + (section * 3)) 
        #rect = dlib.rectangle(left=face.left(), top=face.top(), right=face.right(), bottom=face.bottom())
        # win.add_overlay(face.rect, dlib.rgb_pixel(255,0,0))
        # win.add_overlay(face.parts(), dlib.rgb_pixel(0,0,255))
        #win.add_overlay(draw_rectangle, dlib.rgb_pixel(0,0,255))
        #print(face.part(51).y)

        # for n in range(68):  # Draw all 68 points
            # x, y = face.part(n).x, face.part(n).y
            # cv.circle(frame, (x, y), 2, (0, 255, 0), -1)
            # win.add_overlay(face.parts(), dlib.rgb_pixel(0,0,255))
        #print(frame.shape)
        #cv.imshow("Facial Landmarks", frame)


        win.clear_overlay()

        #frame = np.array(frame[draw_rectangle.left():draw_rectangle.right(), draw_rectangle.top():draw_rectangle.bottom()])
        frame = np.array(frame[draw_rectangle.top():draw_rectangle.bottom(), draw_rectangle.left():draw_rectangle.right()])


        win.set_image(frame)



        # if (recording):
        #     if (K.image_data_format() == 'channels_last'):
        #         lips = frame[draw_rectangle.left():draw_rectangle.right(), draw_rectangle.top():draw_rectangle.bottom()]
        #         rescaled_lips = cv.resize(lips, (100, 50), interpolation=cv.INTER_LINEAR)
        #         rescaled_lips = rescaled_lips.swapaxes(0,1)
        #         #print(rescaled_lips.shape)
        #         record_frames = np.append(record_frames, [np.array(rescaled_lips)], axis=0)
        #         #print(np.array(frame).shape)

web_cam_capture.release()
cv.destroyAllWindows()

if (do_prediction):
    print("beginning prediction")

    predict("evaluation/models/unseen-weights178.h5", record_frames)

#print(record_frames.shape)