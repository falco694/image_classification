#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
kerasで白黒画像をカラー化する
"""

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Flatten, Input, Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D
from keras.optimizers import SGD, adadelta
from keras.utils import np_utils, plot_model
from keras.preprocessing.image import load_img, img_to_array
# import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split

# 各種データ
x = []
y = []
dictlabel = {} # ラベル格納
width = 128 # 立幅
side = 128 # 横幅
train_size = 0.7 # 訓練データの割合
datadir = "dataset/" # データ格納先
batch_size = 32 # 学習毎のデータ数
epochs = 40 # 学習回数

# 学習用のデータを作る。
# datadir以下の画像を読み込む。
# 指定した割合に応じて、trainとtestデータに振り分ける。
for file in os.listdir(datadir):
    if file != ".DS_Store" and file != "desktop.ini":
        filepath = datadir + "/" + file
        # 画像を縦横指定pixelに変換して読み込む。0-255の配列。
        # image = np.array(Image.open(filepath))
        image = load_img(filepath, target_size=(width, side))
        grayimage = image.convert("L")
        # 配列を変換し、[[Redの配列],[Greenの配列],[Blueの配列]] のような形にする。
        # image = image.reshape(width, side, color)
        # image = image.astype("float32")
        image = img_to_array(image)
        grayimage = img_to_array(grayimage)
        # それぞれが0~1になるように正規化して学習or検証データに格納
        x.append(grayimage / 255.)
        # 正解データを追加
        y.append(image / 255.)

# kerasに渡すためにnumpy配列に変換。
x = np.array(x)
y = np.array(y)

# ランダムにシャッフルし、学習データと検証データに振り分け
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size)

### add for TensorBoard
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

old_session = KTF.get_session()

session = tf.Session('')
KTF.set_session(session)
KTF.set_learning_phase(1)

# モデル構築
input_img = Input(shape=(width, side, 1))
x = Conv2D(64, (3, 3), padding='same')(input_img)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(32, (3, 3), padding='same')(encoded)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(3, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
decoded = Activation('sigmoid')(x)

# モデルコンパイル
# model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
model = Model(input_img, decoded)
model.compile(optimizer='adam', loss='binary_crossentropy')
# モデルを表示
model.summary()

# 学習
open("model_convertcolor", "w").write(model.to_json())
### add for TensorBoard
log_filepath = "./logs/"
tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=1, write_graph=True, write_images=True)
###
check = ModelCheckpoint("model_convertcolor.hdf5", save_best_only="true")
earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
callbacks = [check, earlystop, tb_cb]
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks = callbacks,
                    shuffle=True)

# 学習状況を表示
loss_and_metrics = model.evaluate(x_test, y_test, verbose=0)
print("\nloss:{} accuracy:{}".format(loss_and_metrics[0],loss_and_metrics[1]))

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
