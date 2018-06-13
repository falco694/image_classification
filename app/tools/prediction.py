#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
from keras.models import model_from_json
from keras.preprocessing.image import load_img, img_to_array
from keras.backend import tensorflow_backend as backend
import matplotlib.pyplot as plt
import cv2

def main(uploadfile = ""):
    x_test = []
    # データの読み込み
    # filepath = "dataset/cat/cat.0.jpg"
    filepath = ""
    if uploadfile is None:
        raise NameError("写真が選択されていません")
    else:
        filepath = uploadfile
    image = np.array(Image.open(filepath))
    image = load_img(filepath, target_size=(512, 512))
    image = image.convert("L")
    image = img_to_array(image)
    x_test.append(image / 255.)
    x_test = np.array(x_test)

    # 機械学習器を復元
    model = model_from_json(open('model_convertcolor', 'r').read())
    model.load_weights('model_convertcolor.hdf5')
    encoded_imgs = model.predict(x_test)
    # print(encoded_imgs[0])
    # cv2.imwrite("encoded_imgs.jpg", encoded_imgs[0])
    plt.imshow(encoded_imgs[0].reshape(512, 512, 3))
    plt.show()

    # 評価終了時に明示的にセッションをクリア
    backend.clear_session()

if __name__ == "__main__":
    main("./dataset/9053.jpg")
    # main("C:/Users/falco/Desktop/dataset/docorcat/test/9053.jpg")