# huggingface-template

Huggingfaceで適当に何かしたいときのテンプレート．

- モデルの定義（およびモデルのためのコンフィグの定義）とDatasetの定義を書くだけ
- AccelerateによるマルチGPUでの訓練に対応．
- 継続した訓練に対応．例えば，一旦5エポック目まで訓練して保存されたモデルがあったとき，それを読み込んで，エポックや最小lossの情報を維持しながら6エポック目以降を訓練可能．

### どこ変えたらいいの

次の2つのファイルを自分用に書き換えたら，単純なものはほとんど実装できるはず．

- `model/configuration.py`
ModelConfigが定義されているので必要なコンフィグを追記する．ベースとなるtransformerモデルのidとか，分類ならクラスの数とか．このコンフィグを渡せばモデルを初期化できるようにする．

- `model/modeling.py`  
モデルの定義を書く．モデルの初期化の際にはModelConfigのみを受け取り，そのメンバ変数を参照しながら初期化していく．forward()は引数のlabels=が与えられたらlossまで計算し，ModelOutputのインスタンスに入れて返す，

- `model/dataset.py`  
タスクに応じてDatasetクラスを書き換える．generate_dataset()関数は，手元のデータのフォーマットを整形してDatasetクラスに渡すために使う．

- `train.py`
argparseの設定を適宜追記する．特に，ModelConfigを変えた場合にはこれのインスタンスを作るところを追記する．  
訓練のループに関しては特に変えなくて良いはず．

### 訓練中の表示
モデルの訓練が回っているときには，tqdmによるプログレスが表示される．

フォーマットは
```
[Epoch 0] [TRAIN]:  12%|█▏        | 520/4387 [02:31<20:56,  3.08it/s, loss=2.56, lr=7.8e-6]
もしくは
[Epoch 2] [VALID]:  17%|█▋        | 3/18 [00:00<00:01,  9.66it/s, loss=1.83]
```
のようになっている．現状のエポックと，訓練データか開発データのどちらを処理中か，ミニバッチ単位の損失はいくらか，現在の学習率はいくらか，を知ることができる．学習率を表示しているのは，learning rate schedulerが所望の通りに動作しているかを確認できれば嬉しいかなという動機からである．

### 訓練の結果はどうやって保存されるの
訓練済みモデルは2種類保存される．開発データで最小lossを達成したcheckpointである`best/`と，指定のエポック終了時点のcheckpointである`last/`が保存される．その下にどのようなファイルはモデルによって変わるとは思う．

`training_state.json`には，訓練を実行した際のargparseの情報が含まれているため，どのようなオプション（入力データのパス・シード値など）で訓練を実行したかを後から確認可能である．

```
model/
├── best
│   ├── config.json
│   ├── merges.txt
│   ├── training_state.json
│   ├── pytorch_model.bin
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── vocab.json
├── last
│   ├── 上に同じ
└── log.json
```

また，`best/`や`last/`と同じ階層に`log.json`が保存される．これは訓練データおよび開発データに対するlossを示した簡単なサマリである．
```json
// 例
{
    "Epoch 0": {
        "train_log": {
            "loss": 2.3448873911573482
        },
        "valid_log": {
            "loss": 1.7927383250660367
        }
    },
    // ...<中略>...
    "Epoch 4": {
        "train_log": {
            "loss": 2.0223096971879224
        },
        "valid_log": {
            "loss": 1.7126508156458538
        }
    }
}
```

### 訓練したモデルをどうやって読み込むの

モデルには`from_pretrained()`が定義されており，訓練によって出力された`model/best`や`model/last`へのパスを渡せば読み込めるようになっている．トークナイザも同じである．
```py
from model.modeling import Model
from transformers import AutoTokenizer

path = 'model/best'
model = Model.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained(path)
```
