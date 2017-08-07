from keras.models import Model, load_model
from util import *
from bac import BACEncoder, BACDecoder
from net import generator
import numpy as np
import argparse
import pickle
import copy
import struct
import sys
import math
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras import backend as K

def main():
    parser = argparse.ArgumentParser(description='waveone flavor codec')
    parser.add_argument('--gmodel', '-g', help='Generator model (.h5)', required=True)
    parser.add_argument('--lrmodel', '-l', help='LinearRegression model (.pkl)', required=True)
    parser.add_argument('--mode', '-m', type=int, default=0, help='Codec mode [enc=0, dec=1]')
    parser.add_argument('--input_path', '-i', help='Input image or Input code', required=True)
    parser.add_argument('--output_path', '-o', help='Output image or Output code', required=True)

    args = parser.parse_args()

    mode = args.mode
    gmodel = args.gmodel
    lrmodel = args.lrmodel
    input_path = args.input_path
    output_path = args.output_path

    G = generator()
    G.load_weights(gmodel)
    with open(lrmodel, "rb") as f:
        LR = pickle.load(f)

    if mode == 0:
        encode(G, LR, input_path, output_path)
    else:
        decode(G, LR, input_path, output_path)

def get_context(k, cw, uw, pl, cl):
    # context 変数の設定
    aa = -1
    bb = -1
    cc = -1
    dd = -1
    if k > 0:
        bb = pl[cw]
        if cw < (uw-1):
            cc = pl[cw+1]
        if cw > 0:
            aa = pl[cw-1]
    if cw > 0:
        dd = cl[cw-1]
    return (aa, bb, cc, dd)

def encode(G, LR, input_path, output_path):
    encoder = Model(inputs=G.get_input_at(0), outputs=G.get_layer(name='code').output)
    image = imread(input_path)
    code = encoder.predict(np.asarray([image]))[0]
    code_norm, norm_rate = normalize(code)

    # 中間層をちゃんと渡せているかテスト
    f = K.function([G.get_layer(name='code').output], [G.output])

    qcode = quantize(code_norm)

    c, h, w = qcode.shape
    print("dim is")
    print(c, h, w)
    B = definedB()
    uw = w * B
    pl = [-1] * uw
    cl = [-1] * uw

    bac_encoder = BACEncoder()

    dbg = []
    dbg_p = []
    for i in range(c):
        for j in range(h):
            lw = 0
            for k in range(w):
                value = round(math.fabs(qcode[i][j][k]))
                for l in range(B):
                    # 符号 bit かどうか
                    if l == (B-1):
                        bit = [1, 0][qcode[i][j][k] >= 0]
                    else:
                        bit = ((value >> l) & 1)
                    dbg.append(bit)
                    cw = lw + l
                    aa, bb, cc, dd = get_context(k, cw, uw, pl, cl)
                    # 確率推定と符号化
                    prob = LR.predict_proba(np.asarray([[aa, bb, cc, dd]]))
                    zero_prob = prob[0][0]
                    bac_encoder.encode(bit, zero_prob)
                    dbg_p.append([int(zero_prob * 1000000), aa, bb, cc, dd, i, j, k, l, bit])
                    #bac_encoder.encode(bit, 0.2)
                    # 状態更新
                    cl[cw] = bit
                    if i == 0 and j == 0 and (k == 0 or k == 1):
                        print("(0,0,%d,%d) %d %f (%d,%d,%d,%d)" % (k, l, bit, zero_prob, aa, bb, cc, dd))
                # 状態更新
                lw += B
                if i == 0 and j == 0:
                    print("(i,j,k)=(%d,%d,%d) %d (%f)" % (i,j,k,value, qcode[i][j][k]))
            # 状態更新
            pl = copy.deepcopy(cl)
            cl = [0] * uw

    # 終了処理
    bac_encoder.flush()
    # バイト列を保存する
    # 最初の 1byte で norm_rate を保存
    if norm_rate >= 255:
        print("Warning: norm_rate should be lesser 255.")
    with open(output_path, "wb") as f:
        f.write(struct.pack("B", norm_rate))
        for byte in bac_encoder.cod:
            f.write(struct.pack("B", byte))
    print(bac_encoder.cod[0:10])

def decode(G, LR, input_path, output_path):
    with open(input_path, "rb") as f:
        binary = f.read()
        cod = []
        first_flag = False
        norm_rate = 1
        for b in binary:
            if first_flag == False:
                norm_rate = b
                first_flag = True
            else:
                cod.append(b)

    bac_decoder = BACDecoder(cod)

    c, h, w = definedCHW()
    B = definedB()
    uw = w * B
    pl = [-1] * uw
    cl = [-1] * uw

    # ???
    code = np.zeros(shape=(14, 14, 16))

    for i in range(c):
        for j in range(h):
            lw = 0
            for k in range(w):
                value = 0
                for l in range(B):
                    cw = lw + l
                    aa, bb, cc, dd = get_context(k, cw, uw, pl, cl)
                    # 確率推定
                    prob = LR.predict_proba(np.asarray([[aa, bb, cc, dd]]))
                    zero_prob = prob[0][0]
                    bit = bac_decoder.decode(zero_prob)
                    #bit = bac_decoder.decode(0.2)
                    #bac_decoder.decode
                    #bit = bac_decoder.decode(int((1 << 12) / 5))
                    # 状態更新
                    if i == 0 and j == 0 and k == 0:
                        print("(0,0,0,%d) %d" % (l, bit))
                    if l == (B-1):
                        if bit == 1:
                            value = -value
                    else:
                        value |= (bit << l)
                    cl[cw] = bit
                # 状態更新
                lw += B
                code[i][j][k] = value
                if i == 0 and j == 0:
                    print("(i,j,k)=(%d,%d,%d) %d" % (i,j,k,value))

            # 状態更新
            pl = copy.deepcopy(cl)
            cl = [0] * uw

    # 復号時は逆量子化ではスケーリング処理のみ行う
    # denormalize
    code *= norm_rate

    f = K.function([G.get_layer(name='code').output], [G.output])
    decode = f([ np.array([code]) ])[0]
    imsave(output_path, decode[0])

if __name__ == '__main__':
    main()
