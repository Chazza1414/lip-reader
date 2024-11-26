import dlib
import cv2 as cv
import keyboard

frontal_face_detector = dlib.get_frontal_face_detector()
face_landmark_detector = dlib.shape_predictor("../predictors/shape_predictor_68_face_landmarks.dat")

web_cam_capture = cv.VideoCapture(0)
win = dlib.image_window()

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

    dets = frontal_face_detector(frame)

    for i, d in enumerate(dets):
        cv.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 2)

        face = face_landmark_detector(frame, d)
        print(face.part(0))
        print("\n\n")

        #cv.rectangle(frame, (face.part(0)[0], face.part(0)[1]), (face.part(1)[0], face.part(1)[1]), (255, 0, 0), 2)
        win.add_overlay(face)