from pathlib import Path
from joblib import dump, load

import os
import cv2
import imutils
import numpy as np
from tqdm import tqdm

import src.utils as utils


class Recognizer:
    def __init__(self, feature_extractions_methods, classifiers):
        self.feature_extractions_methods = feature_extractions_methods
        self.classifiers = classifiers

    def train(self, train_X, train_y, models_path):
        train_data = []
        for x in train_X:
            train_data.append(self.apply_features(x))
        for classifier in tqdm(self.classifiers):
            classifier.fit(X=train_data, y=train_y)
            dump(classifier, models_path / f'{str(classifier).replace(")", "").replace("(", "")}.joblib')

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

    def apply_features(self, detected_img):
        features = []
        for method in self.feature_extractions_methods:
            features.extend(method(detected_img))
        return features

    def recognize(self, image, models_path):
        # gray_img = self.make_grayscale(image)
        # detected_img = self.detect_number(image)
        predictions = []
        for model_name in os.listdir(models_path):
            model = load(models_path / model_name)
            features = self.apply_features(image)
            features = np.array(features).reshape(1, -1)
            predictions.append(model.predict(features))
        pred = max(predictions, key=predictions.count)
        if predictions.count(pred) < 2:
            return None
        return pred


