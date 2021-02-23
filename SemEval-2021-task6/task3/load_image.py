import numpy as np
from load_data_task3 import load_train_data, load_dev_data, load_test_data
import cv2


def load_image(image_list):
    x = []
    for image in image_list:
        img = cv2.imread(image)
        # cv2.imshow('resize0', img)
        # cv2.waitKey()
        img = cv2.resize(img, (224, 224))
        img = img / 255.
        # print(img.shape)
        # cv2.imshow('resize0', img)
        # cv2.waitKey()
        x.append(img)
        # print('loading : ', image)

    # x = np.array(x)
    return x


if __name__ == '__main__':
    id, texts, techs, image_path = load_dev_data()
    img = load_image(image_path)
    print(len(img))
    print(img[0].shape)
    print(type(img[0]))
