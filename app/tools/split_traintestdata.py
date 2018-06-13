"""
画像データを指定構成比をもとに訓練データとテストデータに振り分けます
画像処理ライブラリは使用せず、ファイル移動で実装しなおす。
import shutil
shutil.move("./test2/test1.txt", ".")
"""

import random
import os
import shutil

datadir = "dataset/" # データ格納先
traindir = datadir + "train/"
testdir = datadir + "test/"
test_size = 0.3

# train/testディレクトリが作成されていない場合は、作成する
# 既に作成されている場合は、一旦削除する
if  os.path.exists(traindir):
    shutil.rmtree(traindir)
else:
    os.makedirs(traindir)
if not os.path.exists(testdir):
    shutil.rmtree(testdir)
else:
    os.makedirs(testdir)

#クラスごとのフォルダを作成する
for labeldir in os.listdir(datadir):
    os.makedirs(traindir + "/" + labeldir)
    os.makedirs(testdir + "/" + labeldir)

# 指定の構成比に従って、画像を分割する
for labeldir in os.listdir(datadir):
    if labeldir == "train" or labeldir == "test":
        continue
    for f in os.listdir(datadir + labeldir):
        src = datadir + labeldir + "/" + f
        if random.random() >= test_size:
            dst = traindir + labeldir + "/" + f
        else:
            dst = testdir + labeldir + "/" + f
        shutil.move(src, dst)
