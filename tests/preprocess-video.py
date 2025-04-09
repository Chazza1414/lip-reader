import cv2 as cv

video_input_path = 'video-input/AV_Clip_Weather.mp4'

input_video_capture = cv.VideoCapture(video_input_path)

face_classififer = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)
mouth_classifier = cv.CascadeClassifier(
    cv.data.haarcascades + 'haarcascade_smile.xml'
)

if not input_video_capture.isOpened():
    print("Error: Could not open video")
    exit()

while input_video_capture.isOpened():
    ret, frame = input_video_capture.read()

    if not ret:
        print("Can't receive frame")
        break
    
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    frame_face = face_classififer.detectMultiScale(gray_frame)

    frame_mouth = mouth_classifier.detectMultiScale(gray_frame)

    for (x,y,w,h) in frame_face:
        cv.rectangle(gray_frame, (x,y), (x+w, y+h), (255,0,0), 2)

        frame_face_mouth = gray_frame[y + int(h/2):y + h, x:x + w]

    cv.imshow('frame',gray_frame)
    #print(gray_frame.shape)
    if cv.waitKey(5) == ord('q'):
        break
