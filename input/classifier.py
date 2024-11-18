import cv2 as cv
import numpy as np

face_classifier = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)
mouth_classifier = cv.CascadeClassifier(
    cv.data.haarcascades + 'haarcascade_smile.xml'
)

class Classifier:
    def __init__(self, type, pixel_tolerance=20, confidence_min=0, confident=False, check_accuracy=False):
        self.type = type
        self.pixel_tolerance = pixel_tolerance
        self.confidence_min = confidence_min
        self.confident = confident
        self.check_accuracy = check_accuracy

        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0
        self.max_index = -1

        self.objects = []
        self.reject_levels = []
        self.level_weights = []

        if type == 'face':
            self.classifier = face_classifier
        elif type == 'mouth':
            self.classifier = mouth_classifier
        else:
            raise ValueError
        
    def __get_most_confident(self):
        if len(self.objects) > 0:
            self.max_index = np.argmax(self.level_weights)
            
            if self.level_weights[self.max_index] > self.confidence_min:
                self.confident = True
                self.check_accuracy = False

                (self.x, self.y, self.w, self.h) = self.objects[self.max_index]
            else:
                self.confident = False
                self.check_accuracy = True
        else:
            self.confident = False
            self.check_accuracy = True
            self.max_index = -1
        
    def detect(self, frame, output_reject_levels=True):
        self.objects, self.reject_levels, self.level_weights = self.classifier.detectMultiScale3(
            frame, 
            outputRejectLevels=output_reject_levels)

        self.__get_most_confident()

    def detect_with_size(self, frame, min_size, max_size, output_reject_levels=True):
        self.objects, self.reject_levels, self.level_weights = self.classifier.detectMultiScale3(
            frame, 
            outputRejectLevels=output_reject_levels,
            minSize=min_size,
            maxSize=max_size)
        
        self.__get_most_confident()

    # if detected object is not within tracked region
    def check_in_region(self, track_x, track_y, track_w, track_h):
        if (self.max_index == -1):
            raise ValueError("No objects detected using " + str(self.type) + " classifier")
        
        (detect_x, detect_y, detect_w, detect_h) = self.objects[self.max_index]
        if (track_x - self.pixel_tolerance > detect_x or 
            track_y - self.pixel_tolerance > detect_y or
            (track_x + track_w + self.pixel_tolerance) < (detect_x + detect_w) or 
            (track_y + track_h + self.pixel_tolerance) < (detect_y + detect_h)):
            return False
        else:
            return True