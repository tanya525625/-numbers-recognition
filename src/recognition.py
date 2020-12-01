import os
from pathlib import Path

import cv2
import time
from tqdm import tqdm
from sklearn.svm import SVC
from keras.datasets import mnist
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier


from src.recognizer import Recognizer
import src.feature_extractions_methods as fem


def main():
    data_dir = Path('../data')
    models_path = Path('../models')
    is_train_mode = False
    is_prediction_mode = False
    is_test_mode = True

    methods = [fem.make_mean_value_in_square, fem.haar_features]
    classifiers = [SVC(), DecisionTreeClassifier(), SGDClassifier(), GradientBoostingClassifier()]
    # arguments = []

    recognizer = Recognizer(methods, classifiers)

    if is_train_mode or is_test_mode:
        (train_X, train_y), (test_X, test_y) = mnist.load_data()
        if is_train_mode:
            recognizer.train(train_X, train_y, models_path)
        elif is_test_mode:
            print(f'Test progress: ')
            time.sleep(1)
            y_pred = []
            for X in tqdm(test_X):
                y_pred.append(recognizer.recognize(X, models_path))
            score = accuracy_score(test_y, y_pred)
            print(f'Accuracy: {score}')
    if is_prediction_mode:
        for image_name in os.listdir(data_dir):
            image = cv2.imread(str(data_dir / image_name), 1)
            image = recognizer.make_grayscale(image)
            pred = recognizer.recognize(image, models_path)
            print(f'Recognized digit: {pred}')


if __name__ == '__main__':
    main()