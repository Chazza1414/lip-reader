import cv2 as cv

class Tracker:
    def __init__(self):
        self.track_success = False
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0
        self.tracker = cv.TrackerMIL.create()

    def create(self, frame, rectangle):
        self.tracker.init(frame, rectangle)

    def update(self, frame):
        self.track_success, (self.x, self.y, self.w, self.h) = self.tracker.update(frame)
    