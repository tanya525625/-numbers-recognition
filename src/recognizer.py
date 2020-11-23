import cv2
import imutils

import src.utils as utils


class Recognizer:
    def __init__(self, feature_extractions_methods):
        self.feature_extractions_methods = feature_extractions_methods

    @staticmethod
    def detect_number(img):
        edged = cv2.Canny(img, 50, 200, 255)
        thresh = cv2.adaptiveThreshold(edged, 255, 1, 1, 11, 1)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
        return img[y:y + h, x:x + w]

    @staticmethod
    def make_grayscale(img):
        src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        src_gray = cv2.GaussianBlur(src_gray, (5, 5), 0)
        return src_gray

    def recognize(self, img_path):
        image = cv2.imread(str(img_path), 1)
        gray_img = self.make_grayscale(image)
        detected_img = self.detect_number(gray_img)
        features = []
        for method in self.feature_extractions_methods:
            features.extend(method(detected_img))


