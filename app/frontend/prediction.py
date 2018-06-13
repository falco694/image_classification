#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
from keras.models import model_from_json
from keras.preprocessing.image import load_img, img_to_array
from keras.backend import tensorflow_backend as backend

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
    image = load_img(filepath, target_size=(128, 128))
    image = img_to_array(image)
    x_test.append(image / 255.)
    x_test = np.array(x_test)

    # 機械学習器を復元
    model = model_from_json(open('model', 'r').read())
    model.load_weights('model.hdf5')

    # 予測結果を保存
    classes = model.predict(x_test, batch_size=1)
    # index = model.predict_classes(x_test, batch_size=1)
    # index = model.predict_proba(x_test, batch_size=1)

    # 評価終了時に明示的にセッションをクリア
    backend.clear_session()
    # ラベルを読み込み
    labels = []
    with open("label.txt", "r") as f:
        for label in f:
            labels.append(label.rstrip("\n"))

    # 評価をラベルと紐づける
    result = []
    for i in range(0, len(labels)):
        result.append([labels[i], classes[0][i]])
    result = sorted(result, key=lambda x: x[1], reverse=True)
    print(result)
    return result

if __name__ == "__main__":
    main("cat.0.jpg")
