import os
from pathlib import Path

import cv2
from PIL import ImageOps
import numpy as np
import time
from tqdm import tqdm
from sklearn.svm import SVC
from keras.datasets import mnist
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

import src.utils as utils
from src.digits_extraction import letters_extract
from src.recognizer import Recognizer
import src.feature_extractions_methods as fem


def main():
    data_dir = Path('../data')
    models_path = Path('../models')
    is_train_mode = False
    is_prediction_mode = True
    is_test_mode = False
    file_path = data_dir / 'var.jpg'

    methods = [fem.make_mean_value_in_square, fem.haar_features, fem.diag_prizn_1, fem.diag_prizn_2]
    classifiers = [SVC(), DecisionTreeClassifier(), GradientBoostingClassifier(), KNeighborsClassifier]
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
        img_lists = letters_extract(str(file_path))
        # (train_X, train_y), (test_X, test_y) = mnist.load_data()
        # img = train_X[0]
        # image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        # image = recognizer.make_grayscale(image)
        # width, height = image.shape[:2]
        # my_img = ImageOps.invert(img_lists[0])
        # my_img = cv2.cvtColor(np.array(my_img), cv2.COLOR_RGB2BGR)
        # my_img = recognizer.make_grayscale(my_img)
        # my_img = utils.resize_image(my_img, 4, 7, 7)
        # width, height = my_img.shape[:2]
        for image in img_lists:
            image = ImageOps.invert(image)
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            image = recognizer.make_grayscale(image)
            image = utils.resize_image(image, 4, 7, 7)
            utils.show_image(image)
            pred = recognizer.recognize(image, models_path)
            print(f'Recognized digit: {pred}')


if __name__ == '__main__':
    main()