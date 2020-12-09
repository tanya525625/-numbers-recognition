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
from sklearn.neighbors import KNeighborsClassifier

import src.utils as utils
from src.output import App
from src.digits_extraction import letters_extract
from src.recognizer import Recognizer
import src.feature_extractions_methods as fem


def main():
    data_dir = Path('../data')
    models_path = Path('../models')
    is_train_mode = True
    is_prediction_mode = False
    is_test_mode = False
    is_models_test_mode = False
    file_path = data_dir / 'sample_2.jpg'

    methods = [fem.make_mean_value_in_square, fem.haar_features, fem.diag_prizn_1, fem.diag_prizn_2,
               fem.make_square_proportion]
    classifiers = [KNeighborsClassifier(), GradientBoostingClassifier(), SVC(), DecisionTreeClassifier(), SGDClassifier()]
    # arguments = []

    recognizer = Recognizer(methods, classifiers)

    if is_train_mode or is_test_mode or is_models_test_mode:
        (train_X, train_y), (test_X, test_y) = mnist.load_data()
        if is_train_mode:
            recognizer.train(train_X, train_y, models_path)
        elif is_test_mode:
            print(f'Test progress: ')
            time.sleep(1)
            y_pred = []
            # test_X = test_X[:10]
            # test_y = test_y[:10]
            for X in tqdm(test_X):
                pred = recognizer.recognize(X, models_path)
                if pred:
                    pred = int(pred)
                else:
                    pred = 0
                y_pred.append(pred)
            score = accuracy_score(test_y, y_pred)
            print(f'Accuracy: {score}')
        elif is_models_test_mode:
            recognizer.test_models(models_path, test_X, test_y)
    if is_prediction_mode:
        img_lists = letters_extract(str(file_path), is_read=True)
        for i in img_lists:
            # image = cv2.cvtColor(np.array(i), cv2.COLOR_RGB2BGR)
            # image = recognizer.make_grayscale(image)
            img = cv2.bitwise_not(i)
            img = cv2.blur(img, (2, 2))
            ret, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
            pred = recognizer.recognize(img, models_path)
            app = App(pred, img)
            # print(f'Recognized digit: {pred}')


if __name__ == '__main__':
    main()