### Prepare Dataset


global_stat.pth：80次元の入力特徴量スペクトログラムにおける、各次元の平均と分散。この値を用いて正規化する。

    (ave, std)というタプルで、ave, stdのそれぞれがshape = (80,)のtorchのtensor。

train.pkl：訓練データをpickle化したもの。具体的には、(pathes, labels)というタプルのリスト。１つ１つの(pathes, labels)が１つのバッチに対応する。

    pathes=[path1, path2, ..., pathn], labels=[label1, label2, ..., labeln]

    pahtiとlabeliが対応する。pathiは入力特徴量のスペクトログラムをtorchのtensorで格納したpthファイル。labeliは正解テキストの文字列。