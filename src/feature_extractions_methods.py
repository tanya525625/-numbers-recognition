import numpy as np
from PIL import Image


def process_img(img, size = (90,60)):
    img_2 = img.resize(size, Image.ANTIALIAS)
    pix_22 = img_2.load()
    width_2 = size[0]
    height_2 = size[1]
    pix_2 = [[0 for j in range(height_2)] for i in range(width_2)]
    for i in range(width_2):
        for j in range(height_2):
            if pix_22[i, j] != (255, 255, 255):
                pix_2[i][j] = 1
    return pix_2


def diag_prizn_1(img, size = (90,60), h = 10):
    pix_2 = process_img(img, size)
    width_2 = size[0]
    height_2 = size[1]
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


def diag_prizn_2(img, size = (90,60), h = 10):
    pix_2 = process_img(img, size)
    width_2 = size[0]
    height_2 = size[1]
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
