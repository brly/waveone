import datetime
import scipy
import numpy as np
import pickle
import math

def imread(path):
    return scipy.misc.imread(path).astype(np.float)

def imsave(path, arr):
    return scipy.misc.imsave(path, arr)

def id():
    today = datetime.datetime.today()
    y = today.year
    mo = today.month
    d = today.day
    h = today.hour
    mi = today.minute
    s = today.second
    return "%04d-%02d-%02d-%02d-%02d-%02d" % (y, mo, d, h, mi, s)

def normalize(y):
    mn = np.amin(y)
    mx = np.amax(y)
    mx_abs = max([math.fabs(mn), math.fabs(mx)])
    norm_rate = math.ceil(mx_abs / 31)
    y /= norm_rate
    return (y, norm_rate)

def quantize(y):
    # quantize
    y = np.ceil(y * 32) / 32
    return y

def definedB():
    return 6

def definedCHW():
    #return (16, 16, 16)
    return (14, 14, 16)

def get_training_image(num):
    source_path = "/home/brly/pyenvs/v3.6.1/waveone-clone/datasets/X-2017-17-28-23-17-35-00.pkl"
    with open(source_path, "rb") as f:
        X = pickle.load(f)
        X = np.asarray(X)

    idx = 1
    while len(X) < num:
        with open("/home/brly/pyenvs/v3.6.1/waveone-clone/datasets/X-2017-17-28-23-17-35-%02d.pkl" % (idx), "rb") as f:
            t = pickle.load(f)
            t = np.asarray(t)
            X = np.vstack((X, t))
        idx += 1

    np.random.shuffle(X)
    return X
