import cv2
from PIL import Image
import numpy as np

import src.utils as utils


def make_mean_value_in_square(image, window=4):
    image = utils.resize_image(image, window)
    squares_mean_values = []
    h, w = image.shape
    for i in range(0, h, window):
        for j in range(0, w, window):
            square = image[i:i + window, j:j + window]
            squares_mean_values.append(np.mean(square))
    return squares_mean_values


# def haar_features(image, window=4):
def haar_features(img, size=(90, 60)):
    # image = utils.resize_image(image, window)
    img_2 = cv2.resize(img, size)
    ret, image = cv2.threshold(img_2, 127, 255, cv2.THRESH_BINARY)
    # h, w = image.shape
    w = size[1]
    h = size[0]
    half = int(w / 2)
    transposed_img = np.transpose(image)
    left_part = transposed_img[:half]
    right_prt = transposed_img[half:]
    return [np.mean(left_part), np.mean(right_prt)]


def process_img(img, size=(90, 60)):
    img_2 = cv2.resize(img, size)
    ret, thresh = cv2.threshold(img_2, 127, 255, cv2.THRESH_BINARY)
    width_2 = size[1]
    height_2 = size[0]
    pix_2 = [[0 for j in range(height_2)] for i in range(width_2)]
    for i in range(width_2):
        for j in range(height_2):
            if thresh[i, j] != 0:
                pix_2[i][j] = 1
    return pix_2


def diag_prizn_1(img, size=(90, 60), h=10):
    pix_2 = process_img(img, size)
    width_2 = size[1]
    height_2 = size[0]
    sr_diag = []
    sr_s = []
    prizn_1 = []
    for l in range(0, width_2, h):
        for m in range(0, height_2, h):
            sr_s.append(pix_2[l][m])
            sr_s.append(pix_2[l + h - 1][m + h - 1])
            for j in range(l + 1, l + h):
                i = j % h
                if i % 2 == 0:
                    for k in range(i // 2):
                        sr_diag.append(pix_2[l + k][m + i - k])
                        sr_diag.append(pix_2[l + i - k][m + k])
                    sr_diag.append(pix_2[l + (i // 2)][m + (i // 2)])
                    sr_s.append(np.mean(sr_diag))
                    sr_diag = []
                else:
                    for k in range(i // 2 + 1):
                        sr_diag.append(pix_2[l + i - k][m + k])
                        sr_diag.append(pix_2[l + k][m + i - k])
                    sr_s.append(np.mean(sr_diag))
                    sr_diag = []
            for j in range(l + 1, l + h - 1):
                i = j % h
                if i % 2 != 0:
                    for k in range((h - 1 - i) // 2):
                        sr_diag.append(pix_2[l + 1 + k][m + h - 1 - k])
                        sr_diag.append(pix_2[l + h - 1 - k][m + 1 + k])
                    sr_diag.append(pix_2[l + 5 + (i // 2)][m + 5 + (i // 2)])
                    sr_s.append(np.mean(sr_diag))
                    sr_diag = []
                else:
                    for k in range((h - 1 - i) // 2 + 1):
                        sr_diag.append(pix_2[l + 1 + k][m + h - 1 - k])
                        sr_diag.append(pix_2[l + h - 1 - k][m + 1 + k])
                    sr_s.append(np.mean(sr_diag))
                    sr_diag = []
            prizn_1.append(np.mean(sr_s))
            sr_s = []
    return prizn_1


def diag_prizn_2(img, size = (90, 60), h = 10):
    pix_2 = process_img(img, size)
    width_2 = size[1]
    height_2 = size[0]
    prizn_2 = []
    sum_pix = 0
    for l in range(0, width_2, h):
        for m in range(0, height_2, h):
            for i in range(l, l + h):
                for j in range(m, m + h):
                    sum_pix += pix_2[i][j]
            prizn_2.append(sum_pix / 100)
            sum_pix = 0
    return prizn_2
