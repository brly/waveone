from net import generator, discriminator

from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, concatenate
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import scipy
import glob
import matplotlib.pyplot as plt
import sys
import datetime
import pickle
import os

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import copy
import time

from util import id, get_training_image, imsave

def main():
    train()
    #train_gan()

def imread(path):
    return scipy.misc.imread(path).astype(np.float)

def imwrite(path, arr):
    return scipy.misc.imsave(path, arr)

def train_gan():
    identifier = id()
    dir_path = "%s-gan-w" % identifier
    os.mkdir(dir_path)

    G = generator()
    D = discriminator()
    # TODO optimizer 調節
    D.compile(optimizer=Adam(lr=3e-04), loss='binary_crossentropy', metrics=['accuracy'])

    # build GAN
    G_input = Input(shape=(128, 128, 3))
    G_output = G(G_input)
    raw_image_input = Input(shape=(128, 128, 3))
    D_output = D([raw_image_input, G_output])
    GAN = Model(inputs=[G_input, raw_image_input], outputs=[G_output, D_output])
    gan_loss = ['mae', 'binary_crossentropy']
    gan_loss_weight = [1e2, 1]
    D.trainable = False

    # TODO optimizer 調節
    GAN.compile(optimizer=Adam(lr=1e-06), loss=gan_loss, loss_weights=gan_loss_weight)

    X = get_training_image(100000)
    X_size = int(len(X) * 0.5)
    train_size = int(X_size * 0.8)
    test_size = X_size - train_size
    X_train = X[0:train_size,:,:,:]
    X_test = X[train_size:train_size+test_size,:,:,:]

    epochs = 400000
    batch_size = 16
    num_batch = int(train_size / batch_size)
    checkpoint = 1

    for epoch in range(epochs):
        start_time = time.time()

        batch_x = 0

        # D の学習を行うか判定
        # acc が一定以上ならば学習しない
        X_test_from_G = G.predict(X_test)
        y_test_D = np.tile([1, 0], [test_size, 1])
        loss_D = D.evaluate([X_test, X_test_from_G], y_test_D, verbose=0)
        acc_D = loss_D[1]
        trainable_D = True
        if acc_D > 0.95:
            trainable_D = False

        for _ in range(num_batch):
            lb = batch_x
            ub = min([batch_x + batch_size, train_size - 1])
            sz = ub - lb

            # D の学習データを用意する
            # G の生成データと生画像データのペアを用意しシャッフルして学習
            X_from_G = G.predict(X_train[lb:ub,:,:,:])
            X_as_Gen = np.zeros(shape=(sz,128,128,3))
            X_as_raw = np.zeros(shape=(sz,128,128,3))
            swap_labels = np.random.randint(2, size=sz)
            y_train_D = np.zeros(shape=(sz,2))
            for i, bit in enumerate(swap_labels):
                if bit == 0:
                    X_as_Gen[i] = X_from_G[i]
                    X_as_raw[i] = X_train[sz+i]
                    y_train_D[i] = np.asarray([1, 0])
                else:
                    X_as_Gen[i] = X_train[sz+i]
                    X_as_raw[i] = X_from_G[i]
                    y_train_D[i] = np.asarray([0, 1])

            # GAN の学習
            y_train_GAN = np.tile([1, 0], [sz, 1])
            GAN.train_on_batch([X_train[lb:ub,:,:,:], X_train[lb:ub,:,:,:]],
                [X_from_G, y_train_GAN])

            # D の学習
            if trainable_D:
                D.train_on_batch([X_as_raw, X_as_Gen], y_train_D)

            # 状態更新
            batch_x += batch_size

        # logging
        elapsed = time.time() - start_time
        X_test_from_G = G.predict(X_test)
        y_test_D = np.tile([1, 0], [test_size, 1])
        loss_D = D.evaluate([X_test, X_test_from_G], y_test_D, verbose=0)
        loss_GAN = GAN.evaluate([X_test, X_test], [X_test_from_G, y_test_D], verbose=0)
        now = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        print("epoch %06d: D loss %.3f, acc %.3f: GAN loss %.3f, G loss %.3f D loss %.3f: trainable_D %d: %.3fs: %s" %
            (epoch, loss_D[0], loss_D[1], loss_GAN[0], loss_GAN[1], loss_GAN[2], trainable_D, elapsed, now))

        # checkpoint
        if epoch > 0 and (epoch % checkpoint == 0):
            # 画像の保存
            pred_image_path = "%s/G-predict-%06d.png" % (dir_path, epoch)
            pred = G.predict(X_test)
            tests = np.hstack(X_test[0:10])
            preds = np.hstack(pred[0:10])
            if epoch == checkpoint:
                imsave("%s/tests-%06d.png" % (dir_path, epoch), tests)
            imsave("%s/preds-%06d.png" % (dir_path, epoch), preds)

            # モデルの保存
            G.save("%s/G.h5" % dir_path)
            D.save("%s/D.h5" % dir_path)
            GAN.save("%s/GAN.h5" % dir_path)

def train():
    identifier = id()

    # directory 作成
    os.mkdir(identifier)

    # generator 単体で学習
    G = generator()
    G.compile(optimizer=Adam(lr=3e-04), loss='mae', metrics=['mae'])

    encoder = Model(inputs=G.get_input_at(0), outputs=G.get_layer(name='code').output)

    # 学習する画像の準備
    X = get_training_image(250000)
    np.random.shuffle(X)

    # training-set, test-set を分割
    X_size = int(len(X))
    train_size = int(X_size * 0.8)
    test_size = X_size - train_size
    X_train = X[0:train_size,:,:,:]
    X_test = X[train_size:train_size+test_size,:,:,:]

    epochs = 400000
    batch_size = 16
    num_batch = int(train_size / batch_size)

    # TODO 調節
    checkpoint = 10

    for epoch in range(epochs):
        start_time = time.time()

        batch_x = 0
        for _ in range(num_batch):
            lb = batch_x
            ub = min([batch_x + batch_size, train_size - 1])
            G.train_on_batch(X_train[lb:ub,:,:,:],
                X_train[lb:ub,:,:,:])
            batch_x += batch_size

        # 訓練/テスト誤差の計算
        train_score = G.evaluate(x=X_train, y=X_train, verbose=0)
        test_score = G.evaluate(x=X_test, y=X_test, verbose=0)

        now = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        elapsed = time.time() - start_time

        print("train epoch %06d: train loss %f: test loss %f: %.fs: %s" %
            (epoch, train_score[0], test_score[0], elapsed, now))

        if epoch % checkpoint == 0 and epoch > 0:
            filename = "%s/input-predict-%06d.png" % (identifier, epoch)
            Y_pred = G.predict(X_train[0:1,:,:,:])
            scipy.misc.imsave(filename, np.concatenate([X_train[0], Y_pred[0]]))
            # モデルの保存
            G.save("%s/G.h5" % (identifier))

if __name__ == '__main__':
    main()
