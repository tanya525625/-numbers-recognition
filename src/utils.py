import cv2


def show_image(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize_image(img, window, st_h=7, st_w=7):
    width = int(st_w * window)
    height = int(st_h * window)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

