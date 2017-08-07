from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from net import *
from util import get_training_image

import argparse
import pickle
import sys
import math
import copy
import time

def main():
    parser = argparse.ArgumentParser(description='waveone flavor codec')
    parser.add_argument('--gmodel', '-g', help='Generator model (.h5)')
    parser.add_argument('--lr_output', '-l', help='Output LinearRegression model (.pkl)')
    args = parser.parse_args()

    G = generator()
    G.load_weights(args.gmodel)

    X = get_training_image(50000)
    y = G.predict(X[0:200])
    lr = train_bit_predictor(y)

    with open(args.lr_output, "wb") as f:
        pickle.dump(lr, f)

def train_bit_predictor(y):
    num, c, h, w = y.shape

    # 正規化/量子化/整数化
    # 要素毎に正規化
    y_norm = np.zeros(shape=(num, c, h, w))
    for i in range(num):
        mn = np.amin(y[i])
        mx = np.amax(y[i])
        mx_abs = max([math.fabs(mn), math.fabs(mx)])
        norm_rate = math.ceil(mx_abs / 31)
        y[i] /= norm_rate
    # 量子化
    y_hat = np.ceil(y * 32) / 32
    y_hat = y_hat.astype("int8")

    # context の定義
    # 処理順序を考慮して [(-1,0), (-1,-1), (0,-1), (1,-1)] にする
    context = []
    B = 6
    # aa bb cc
    # dd []
    # 2列分データを保持しておく必要がある
    uw = w * B
    pl = [-1] * (uw)
    cl = [-1] * (uw)
    sz = num * c * h * w * B
    X_data = np.zeros(shape=(sz,4))
    y_data = np.zeros(shape=(sz,1))
    idx = 0
    prev = time.time()
    for i in range(num):
        for j in range(c):
            for k in range(h):
                lw = 0
                for l in range(w):
                    value = y_hat[i][j][k][l]
                    for m in range(B):
                        # 符号 bit かどうか
                        if m == (B-1):
                            bit = [0, 1][value >= 0]
                        else:
                            bit = ((value >> m) & 1)
                        cw = lw + m
                        aa = -1
                        bb = -1
                        cc = -1
                        dd = -1
                        # context 変数の設定
                        if k > 0:
                            bb = pl[cw]
                            if cw < (uw-1):
                                cc = pl[cw+1]
                            if cw > 0:
                                aa = pl[cw-1]
                        if cw > 0:
                            dd = cl[cw-1]
                        X_data[idx] = np.array([aa,bb,cc,dd])
                        y_data[idx] = bit
                        idx += 1
                        # 状態更新
                        cl[cw] = bit
                    # 状態更新
                    lw += B
                # 状態更新
                pl = copy.deepcopy(cl)
                cl = [0] * (uw)
        # 経過
        print("%04d/%04d load: %.3fs" % (i, num, time.time() - prev))
        prev = time.time()
    # データシャッフルはモデル側でやってくれるらしいので放置
    # データ分割
    data_size = len(X_data)
    train_size = int(data_size * 0.8)
    test_size = data_size - train_size
    X_train = X_data[0:train_size,:]
    y_train = y_data[0:train_size]
    X_test = X_data[train_size:train_size+test_size,:]
    y_test = y_data[train_size:train_size+test_size]
    # 学習
    lr_model = LogisticRegression(solver='sag', verbose=1, n_jobs=12)
    lr_model.fit(X_train, y_train)
    # 学習誤差/テスト誤差
    train_loss = lr_model.score(X_train, y_train)
    test_loss = lr_model.score(X_test, y_test)
    print("lr-model train_acc %f: test_acc %f" % (train_loss, test_loss))
    return lr_model

if __name__ == '__main__':
    main()
