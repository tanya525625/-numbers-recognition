from typing import List

import numpy as np
import cv2
from PIL import Image

# image_file = "C:\\Users\\Home\\Desktop\\alg\\source.png"


def letters_extract(image_file: str, out_size=28) -> List[Image.Image]:
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

    # Get contours
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    output = img.copy()

    letters = []
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        if hierarchy[0][idx][3] == 0:
            cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
            letter_crop = img_erode[y:y + h, x:x + w]

            # Resize letter canvas to square
            size_max = max(w, h)
            letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
            if w > h:
                # Enlarge image top-bottom
                # ------
                # ======
                # ------
                y_pos = size_max // 2 - h // 2
                letter_square[y_pos:y_pos + h, 0:w] = letter_crop
            elif w < h:
                # Enlarge image left-right
                # --||--
                x_pos = size_max // 2 - w // 2
                letter_square[0:h, x_pos:x_pos + w] = letter_crop
            else:
                letter_square = letter_crop

            # Resize letter to 28x28 and add letter and its X-coordinate
            letters.append((x, w, cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA)))
    cv2.imshow("Output", output)
    cv2.waitKey(0)
    # Sort array in place by X-coordinate
    letters.sort(key=lambda x: x[0], reverse=False)

    letters_as_image = [Image.fromarray(np.uint8(letter[2])).convert('RGB') for letter in letters]

    return letters_as_image


# letters = letters_extract(image_file)
# k = 0
# for letter in letters:
#     letter.save(f"C:\\Users\\Home\\Desktop\\alg\\{k}.png")
#     k += 1
