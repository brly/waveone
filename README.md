# waveone minimal imp

[SIGNICO](http://signico.hi-king.me/) vol4 で解説された「ニューラルネットワークで画像圧縮」の記事での実装です.

動作させるためには別途データセットが必要となります.
そのまま動かすのであれば util.py の get_training_image を書き換えて入力となる `(N,128,128,3)` 次元の numpy.array を返すようにしてください.

## 依存ライブラリ

```
backports.weakref==1.0rc1
bleach==1.5.0
cycler==0.10.0
Django==1.11.3
h5py==2.7.0
html5lib==0.9999999
image==1.5.7
Keras==2.0.5
Markdown==2.6.8
matplotlib==2.0.2
numpy==1.13.1
olefile==0.44
Pillow==4.2.1
protobuf==3.3.0
pydot==1.2.3
pyparsing==2.2.0
python-dateutil==2.6.1
python-magic==0.4.13
pytz==2017.2
PyYAML==3.12
scikit-learn==0.18.2
scipy==0.19.1
six==1.10.0
tensorflow-gpu==1.2.1
Theano==0.9.0
Werkzeug==0.12.2
```

恐らく必要のないライブラリも列挙されているかと思いますが, あくまで開発時の環境の情報として
`pip freeze` の情報を載せておきます.

## encoder-decoder ネットワークの学習

```
python main.py
```
実行すると実行時の日付名のディレクトリが作成され, ソース内で指定されているチェックポイント毎にモデルとネットワークの出力画像を保存します.

## ロジスティック回帰の学習

```
python lr.py -g [input encoder-decoder model h5] -l [output lr-model pickle]
```
割と時間がかかります...

## エンコード, デコード

```
# encode
python codec.py -g [input encoder-decoder model h5] -l [input lr-model pickle] -i [input image] -o [output code]
ex) python codec.py -g G.h5 -l lr.pkl -i input.jpg -o code.bin

# decode
python codec.py -m 1 -g [input encoder-decoder model h5] -l [input lr-model pickle] -i [input code] -o [output decode image]
ex) python codec.py -m 1 -g G.h5 -l lr.pkl -i code.bin -o decode.png

```
