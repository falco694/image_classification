#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
kerasで白黒画像をカラー化する
"""

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Flatten, Input, Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D
from keras.optimizers import SGD, adadelta
from keras.utils import np_utils
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import numpy as np
import os
# from sklearn.model_selection import train_test_split

# 各種データ
x = []
y = []
dictlabel = {} # ラベル格納
width = 512 # 立幅
side = 512 # 横幅
test_size = 0.3 # 訓練データの割合
datadir = "dataset/" # データ格納先
traindir = datadir + "train/" #訓練データ格納先
testdir = datadir + "test/" #テストデータ格納先
batch_size = 32 # 学習毎のデータ数
epoch = 40 # 学習回数

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
model = Model(input_img, decoded)
model.compile(optimizer='adam', loss='binary_crossentropy')
# モデルを表示
model.summary()

# 学習
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'dataset/train',
        target_size=(width, side),
        batch_size=batch_size,
        class_mode='identical')

validation_generator = test_datagen.flow_from_directory(
        'dataset/test',
        target_size=(width, side),
        batch_size=batch_size,
        class_mode='identical')

# steps_per_epoch 算出するために、訓練テストディレクトリそれぞれのファイル数（画像データ数）を算出する
traindatacount = 0
testdatacount = 0
for f in os.listdir(traindir):
    traindatacount = traindatacount + 1
for f in os.listdir(testdir):
    testdatacount = testdatacount + 1

steps_per_epoch = traindatacount // batch_size
validation_steps = testdatacount // batch_size

open("model_convertcolor", "w").write(model.to_json())
### add for TensorBoard
log_filepath = "./logs/"
tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=1, write_graph=True, write_images=True)
###
check = ModelCheckpoint("model_convertcolor.hdf5", save_best_only="true")
earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
callbacks = [check, earlystop, tb_cb]

# 学習
history = model.fit_generator(train_generator,
                                steps_per_epoch=steps_per_epoch,
                                epochs=epoch,
                                verbose=1,
                                validation_data=validation_generator,
                                validation_steps=validation_steps,
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
