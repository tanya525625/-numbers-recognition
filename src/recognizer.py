from pathlib import Path
from joblib import dump, load

import os
import cv2
import time
import imutils
import numpy as np
from PIL import ImageOps, Image
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from src.digits_extraction import letters_extract
import src.utils as utils


class Recognizer:
    def __init__(self, feature_extractions_methods, classifiers):
        self.feature_extractions_methods = feature_extractions_methods
        self.classifiers = classifiers

    def train(self, train_X, train_y, models_path):
        train_data = []
        print("Features' applying process")
        time.sleep(1)
        # train_X = train_X[:10]
        # train_y = train_y[:10]
        for x, y in tqdm(zip(train_X, train_y)):
            x = cv2.cvtColor(np.array(x), cv2.COLOR_RGB2BGR)
            x = self.make_grayscale(x)
            x = utils.resize_image(x, 4, 7, 7)
            _, image = cv2.threshold(x, 127, 255, cv2.THRESH_BINARY)
            # utils.show_image(image)
            train_data.append(self.apply_features(image))
        for classifier in tqdm(self.classifiers):
            classifier.fit(X=train_data, y=train_y)
            dump(classifier, models_path / f'{str(classifier).replace(")", "").replace("(", "")}.joblib')

    def test_models(self, models_path, X_test, y_test):
        curr_predictions = []
        test_data = []
        print('Models testing applying features process')
        time.sleep(1)
        for x in tqdm(X_test):
            test_data.append(self.apply_features(x))
        print('Models predictions process')
        time.sleep(1)
        for model_name in tqdm(os.listdir(models_path)):
            curr_predictions.clear()
            model = load(models_path / model_name)
            for x in test_data:
                x = np.array(x).reshape(1, -1)
                pred = model.predict(x)
                if pred:
                    pred = int(pred)
                else:
                    pred = 0
                curr_predictions.append(pred)
            acc = accuracy_score(y_test, curr_predictions)
            print(f'Accuracy for {model_name}: {acc}')

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
            # print(method(detected_img))
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
            # print(f'{model_name}: {model.predict(features)}')
        pred = max(predictions, key=predictions.count)
        if predictions.count(pred) < 2:
            return None
        return pred


