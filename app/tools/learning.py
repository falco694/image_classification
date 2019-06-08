#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
kerasでの画像分類の学習
"""

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Flatten, Input, GlobalAveragePooling2D, Conv2D, MaxPooling2D
from keras.optimizers import SGD, adadelta
from keras.utils import np_utils, plot_model
from keras.utils.np_utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split

# 各種データ
x = []
y = []
dictlabel = {}  # ラベル格納
width = 128  # 立幅
side = 128  # 横幅
# color = 3 # モノクロ:1,カラー:3
test_size = 0.7  # 訓練データの割合
datadir = "dataset/"  # データ格納先
batch_size = 32  # 学習毎のデータ数
epoch = 40  # 学習回数

# 学習用のデータを読み込む。
# datadir以下の画像を読み込む。
index = 0
for dir in os.listdir(datadir):
    if dir == ".DS_Store":
        continue
    # ラベル
    dictlabel[dir] = index

    dirlabel = datadir + dir
    for file in os.listdir(dirlabel):
        if file != ".DS_Store" and file != "desktop.ini":
            filepath = dirlabel + "/" + file
            # 画像を縦横指定pixelに変換して読み込む。0-255の配列。
            # image = np.array(Image.open(filepath))
            image = load_img(filepath, target_size=(width, side))
            # 配列を変換し、[[Redの配列],[Greenの配列],[Blueの配列]] のような形にする。
            # image = image.reshape(width, side, color)
            # image = image.astype("float32")
            image = img_to_array(image)
            # それぞれが0~1になるように正規化して学習or検証データに格納
            x.append(image / 255.)
            # 正解データを追加
            y.append(index)
    index += 1

# ラベルを保存
with open('label.txt', 'w') as f:
    for label in dictlabel:
        f.write(str(label) + "\n")
    f.close()

# kerasに渡すためにnumpy配列に変換。
x = np.array(x)

# ラベルの配列を1と0からなるラベル配列に変更
# 0 -> [1,0], 1 -> [0,1] という感じ。
y = to_categorical(y)

# ランダムにシャッフルし、学習データと検証データに振り分け
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=train_size)

# add for TensorBoard

old_session = KTF.get_session()

session = tf.Session('')
KTF.set_session(session)
KTF.set_learning_phase(1)
###

# モデル構築
# 転移学習
input_tensor = Input(shape=(width, side, 3))
base_model = VGG16(weights='imagenet', include_top=False,
                   input_tensor=input_tensor)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(dictlabel), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers[:15]:
    layer.trainable = False

# 通常学習
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(width, side, color)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(len(dictlabel), activation='softmax'))

# モデルコンパイル
# model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(loss="categorical_crossentropy",
              optimizer='adadelta', metrics=["accuracy"])
# モデルを表示
model.summary()

# 学習
open("model", "w").write(model.to_json())
# add for TensorBoard
log_filepath = "./logs/"
tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=1,
                    write_graph=True, write_images=True)
###
check = ModelCheckpoint("model.hdf5", save_best_only="true")
earlystop = EarlyStopping(
    monitor='val_loss', patience=10, verbose=0, mode='auto')
callbacks = [check, earlystop, tb_cb]
# 指定した割合に応じて、trainとtestデータに振り分ける。
history = model.fit(x, y,
                    batch_size=batch_size,
                    epochs=epoch,
                    verbose=1,
                    callbacks=callbacks,
                    validation_split=0.3)

# 学習状況を表示
loss_and_metrics = model.evaluate(x_test, y_test, verbose=0)
print("\nloss:{} accuracy:{}".format(loss_and_metrics[0], loss_and_metrics[1]))

# モデルを写真形式で保存
# エラーとなるため、無効化している。
# plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)

# def plot_history(history):
#     """精度/損失をプロット
#     """

#     # 精度の履歴をプロット
#     plt.plot(history.history['acc'],"o-", label="accuracy")
#     plt.plot(history.history['val_acc'],"o-", label="val_acc")
#     plt.title('model accuracy')
#     plt.xlabel('epoch')
#     plt.ylabel('accuracy')
#     plt.legend(loc="lower right")
#     plt.show()

#     # 損失の履歴をプロット
#     plt.plot(history.history['loss'],"o-", label="loss",)
#     plt.plot(history.history['val_loss'],"o-", label="val_loss")
#     plt.title('model loss')
#     plt.xlabel('epoch')
#     plt.ylabel('loss')
#     plt.legend(loc='lower right')
#     plt.show()

# # modelに学習させた時の変化の様子をplot
# plot_history(history)

fig = plt.figure(figsize=(2, 9))
fig.subplots_adjust(left=0, right=1, bottom=0,
                    top=0.5, hspace=0.05, wspace=0.05)
for i in range(18):
    ax = fig.add_subplot(2, 9, i + 1, xticks=[], yticks=[])
    ax.imshow(x_train[i].reshape((128, 128)), cmap='rgb')
