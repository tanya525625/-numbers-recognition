from pathlib import Path

from src.recognizer import Recognizer
import src.feature_extractions_methods as fem


def main():
    data_dir = Path('../data')
    img_path = data_dir / 'test_5.jpg'

    methods = [fem.make_mean_value_in_square, fem.haar_features]
    recognizer = Recognizer(methods)
    recognizer.recognize(img_path)


if __name__ == '__main__':
    main()