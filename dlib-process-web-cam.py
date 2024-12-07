import dlib
import cv2 as cv
import keyboard
import numpy as np
import keras.backend as K

recording = False
record_frames = np.empty((0,100,50,3))
print(record_frames.shape)

frontal_face_detector = dlib.get_frontal_face_detector()
face_landmark_detector = dlib.shape_predictor("../predictors/shape_predictor_68_face_landmarks.dat")

web_cam_capture = cv.VideoCapture(0)
win = dlib.image_window()

margin=10
mouth_box_width = 60
mouth_box_height = 35

width_scale = 0.75
height_scale = 0.75

while web_cam_capture.isOpened():

    ret, frame = web_cam_capture.read()
    #frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #frame = dlib.load_rgb_image(frame)
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

    win.clear_overlay()
    win.set_image(frame)

    if not ret:
        print("Can't receive frame")
        break

    if keyboard.is_pressed('q'):
        break

    if keyboard.is_pressed('r'):
        print("recording changed")
        recording = not recording

    dets = frontal_face_detector(frame)

    for i, d in enumerate(dets):
        #cv.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 2)

        face = face_landmark_detector(frame, d)
        #print(face.parts())
        
        # for j, p in enumerate(face.parts()):
        #     cv.circle(frame, (p.x, p.y), radius=5, color=(255,0,0), thickness=-1)
        #     win.add_overlay_circle(p, radius=3, color=dlib.rgb_pixel(255,0,0))
        # mouth_rectangle = dlib.rectangle(left=face.part(48).x - margin, 
        #                            top=face.part(51).y - margin, 
        #                            right=face.part(54).x + margin, 
        #                            bottom=face.part(57).y + margin)

        # mouth_rectangle = dlib.rectangle(left=face.part(48).x, 
        #                            top=face.part(51).y, 
        #                            right=face.part(54).x, 
        #                            bottom=face.part(57).y)
        #print(mouth_rectangle.center())

        # draw_rectangle = dlib.rectangle(left=mouth_rectangle.center().x - int(mouth_rectangle.width() * width_scale), 
        #                                 top=mouth_rectangle.center().y - int(mouth_rectangle.height() * height_scale), 
        #                                 right=mouth_rectangle.center().x + int(mouth_rectangle.width() * width_scale), 
        #                                 bottom=mouth_rectangle.center().y + int(mouth_rectangle.height() * height_scale))
        # draw_rectangle = dlib.rectangle(left=int(mouth_rectangle.left() * width_scale), 
        #                                 top=int(mouth_rectangle.top() * height_scale), 
        #                                 right=int(mouth_rectangle.right() * width_scale), 
        #                                 bottom=int(mouth_rectangle.bottom() * height_scale))

        section = face.part(51).y - face.part(33).y
        #print(face.part(51), face.part(33))

        draw_rectangle = dlib.rectangle(left=face.part(33).x - (section * 3), 
                                        top=face.part(33).y, 
                                        right=face.part(33).x + (section * 3), 
                                        bottom=face.part(33).y + (section * 3)) 
        #print(frame.shape)
        #print(K.image_data_format())
        if (recording):
            if (K.image_data_format() == 'channels_last'):
                lips = frame[draw_rectangle.left():draw_rectangle.right(), draw_rectangle.top():draw_rectangle.bottom()]
                rescaled_lips = cv.resize(lips, (100, 50), interpolation=cv.INTER_LINEAR)
                rescaled_lips = rescaled_lips.swapaxes(0,1)
                print(rescaled_lips.shape)
                record_frames = np.append(record_frames, [np.array(rescaled_lips)], axis=0)
                print(np.array(frame).shape)
                #record_frames = np.concatenate((record_frames, frame), axis=0)
            

        #print(rectangle.height, rectangle.width)
        win.add_overlay(draw_rectangle, dlib.rgb_pixel(0,0,255))

        #print("\n\n")

        #cv.rectangle(frame, (face.part(0)[0], face.part(0)[1]), (face.part(1)[0], face.part(1)[1]), (255, 0, 0), 2)
        #win.add_overlay(face)

print(record_frames.shape)