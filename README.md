# huggingface-template

Huggingfaceで適当に何かしたいときのテンプレート．

- モデルの定義とDatasetの定義を書くだけ
- AccelerateによるマルチGPUでの訓練に対応．
- 継続した訓練に対応．例えば，一旦5エポック目まで訓練して保存されたモデルがあったとき，それを読み込んで，エポックや最小lossの情報を維持しながら6エポック目以降を訓練可能．

### どこ変えたらいいの

次の2つのファイルを自分用に書き換えたら，単純なものはほとんど実装できるはず．

- `modeling.py`  
モデルの定義を書く．モデルの`forward()`ではModelOutputクラスのインスタンスが返り値になっている．train.pyではこのクラスのメンバ変数`.loss`を使うことになるので，必ずModelOutputクラスのlossに値を入れて返すようにする．

- `dataset.py`  
Datasetクラスと，`generate_dataset()`関数が定義されている．
タスクに応じてDatasetクラスを書き換えて，手元のデータのフォーマットに応じて`generate_dataset()`を書き換えるイメージ．
  
Datasetクラスはよくある`__len__()`と`__getitem__()`を定義するもの．  
`generate_dataset()`は，データのファイルのパスを入力とし，Datasetのインスタンスを返す関数．

※ ベースの実装はローカルのファイルからデータを読み込むことを想定したものになっている．Huggingface datasetsから読み込む場合に対応していないが，おそらく大した修正にはならないはず（普段使わないのでよく分からない）．

### 訓練中の表示
モデルの訓練が回っているときには，tqdmによるプログレスが表示される．

フォーマットの例を次に示す．
```
[Epoch 0] [TRAIN]:  12%|█▏        | 520/4387 [02:31<20:56,  3.08it/s, loss=2.56, lr=7.8e-6]
もしくは
[Epoch 2] [VALID]:  17%|█▋        | 3/18 [00:00<00:01,  9.66it/s, loss=1.83]
```
のようになっている．現状のエポックと，訓練データか開発データのどちらを処理中か，ミニバッチ単位の損失はいくらか，現在の学習率はいくらか，を知ることができる．学習率を表示しているのは，learning rate schedulerが所望の通りに動作しているかを確認できれば嬉しいかなという動機からである．


### 訓練の結果はどうやって保存されるの
訓練済みモデルは2種類保存される．開発データで最小lossを達成したcheckpointである`best/`と，指定のエポック終了時点のcheckpointである`last/`が保存される．ファイルはモデルによって変わるとは思う．

`my_config.json`には，訓練を実行した際のargparseの情報が含まれているため，どのようなオプション（入力データのパス・シード値など）で訓練を実行したかを後から確認可能である．

```
model/
├── best
│   ├── config.json
│   ├── lr.bin
│   ├── merges.txt
│   ├── my_config.json
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

`best/`や`last/`のディレクトリに生成されるファイルは，`my_config.json`を除いてHuggingfaceの標準的な命名に従う．したがって，トークナイザは`AutoTokenizer.from_pretrained()`に直接ディレクトリのパスを渡せば読み込める．

モデルについても，`modeling.py`を見ればわかるように，クラスメソッドの`from_pretrained()`が定義されている．これにより，`BertModel`などのHuggingface標準のモデルと同じインターフェイスで読み込むことが可能である（`save_pretrained()`も同様）．

