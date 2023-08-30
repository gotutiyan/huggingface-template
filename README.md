# huggingface-template

Huggingfaceで適当に何かしたいときのテンプレート．

- モデルの定義（およびモデルのためのコンフィグの定義）とDatasetの定義を書くだけ
- AccelerateによるマルチGPUでの訓練に対応．
- 継続した訓練に対応．例えば，一旦5エポック目まで訓練して保存されたモデルがあったとき，それを読み込んで，エポックや最小lossの情報を維持しながら6エポック目以降を訓練可能．

# どこ書き換えたらいいの，各ファイルの役割は何

`train.py`と`model/`以下を良い感じに書き換える．  
注意：`model/`以下にある3つのモジュールは相対パスで参照し合っているため，pythonコマンドで直接実行できない．必ず親ディレクトリのファイル（このリポジトリで言うとtrain.pyみたいなファイル）からimportする形で利用する．

- `model/configuration.py`  
    ModelConfigが定義されているので必要なコンフィグを追記する．ベースとなるtransformerモデルのidとか，分類ならクラスの数とか．このコンフィグを渡せばモデルを初期化できるようにする．
    ```py
    from model.configuratoin import ModelConfig
    config = ModelConfig(arguments)
    ...
    ```

- `model/modeling.py`  
    モデルの定義を書く．モデルの初期化の際にはModelConfigのみを受け取り，ModelConfigのメンバ変数を参照しながら初期化する．forward()は引数のlabels=が与えられたらlossまで計算し，ModelOutputのインスタンスに入れて返すようにする，
    ```py
    from model.configuratoin import ModelConfig
    from model.modeling import Model
    config = ModelConfig(arguments)
    model = Model(config)
    outputs = model(**{'input_ids':..., 'attention_mask':..., 'labels':...})
    ouputs.loss.backward()
    ...
    ```

- `model/dataset.py`  
    タスクに応じてDatasetクラスを書き換える．generate_dataset()関数は，手元のデータのフォーマットを整形してDatasetクラスに渡すために使う．基本，外からはgenerate_dataset()を呼ぶ．  
    扱うのはあくまでDatasetに相当するものなので，DataLoaderに入れるのを忘れないこと．
    ```py
    from model.dataset import generate_dataset
    from torch.utils.data import DataLoader
    train_dataset = generate_dataset(train_file_path)
    valid_dataset = generate_dataset(valid_file_path)
    train_loader = DataLoader(train_dataset)
    valid_loader = DataLoader(valid_dataset)
    ...
    ```

- `train.py`  
    argparseの設定，ModelConfigを初期化する時の引数，generate_dataset()に渡す引数，の辺りを適宜追記する．
    訓練のループに関しては特に変えなくて良いはず．

    実行時は，Accelerateに基づいているため`accelerate launch train.py <arguments>`のようにして実行する．  
    複数のGPUで分散学習する場合は，`accelerate config`を実行して必要な設定を入力するか，直接コマンドライン引数を渡して実行する．

    <details>
    <summary>参考：accelerate configで入力する時</summary>
    GPU2枚使うとき

    ```
    In which compute environment are you running? ([0] This machine, [1] AWS (Amazon SageMaker)): 0
    Which type of machine are you using? ([0] No distributed training, [1] multi-CPU, [2] multi-GPU, [3] TPU [4] MPS): 2
    How many different machines will you use (use more than 1 for multi-node training)? [1]: 1
    Do you want to use DeepSpeed? [yes/NO]: NO
    Do you want to use FullyShardedDataParallel? [yes/NO]: NO
    How many GPU(s) should be used for distributed training? [1]:2
    What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]:
    Do you wish to use FP16 or BF16 (mixed precision)? [NO/fp16/bf16]: bf16
    ```

    </details>

    <details>
    <summary>参考：コマンドライン引数で渡す時</summary>
    `--num_processes=`がGPUの枚数に相当．

    ```sh
    accelerate launch \
    --multi_gpu \
    --num_processes 2 \
    train.py \
    <arguments for train.py>
    ```

    </details>

    

# 訓練中の表示
モデルの訓練が回っているときには，tqdmによるプログレスが表示される．

フォーマットは
```
[Epoch 0] [TRAIN]:  12%|█▏        | 520/4387 [02:31<20:56,  3.08it/s, loss=2.56, lr=7.8e-6]
もしくは
[Epoch 2] [VALID]:  17%|█▋        | 3/18 [00:00<00:01,  9.66it/s, loss=1.83]
```
のようになっている．現状のエポックと，訓練データか開発データのどちらを処理中か，ミニバッチ単位の損失はいくらか，現在の学習率はいくらか，を知ることができる．学習率を表示しているのは，learning rate schedulerが所望の通りに動作しているかを確認できれば嬉しいかなという動機からである．

# 訓練の結果はどうやって保存されるの
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

# 訓練したモデルをどうやって読み込むの

モデルには`from_pretrained()`が定義されており，訓練によって出力された`model/best`や`model/last`へのパスを渡せば読み込めるようになっている．トークナイザも同じである．
```py
from model.modeling import Model
from transformers import AutoTokenizer

path = 'model/best'
model = Model.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained(path)
```

