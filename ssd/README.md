# SSD実行

## 事前準備
### openVINOのインストール  

とりあえず、pyenvで仮想環境作ってインストールしてみる。  
[Intel® Distribution of OpenVINO™ Toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html)
でDevtools、Linux、2022.1、PIP、Caffe/Tensorflow2.X/ONNXを選ぶと
「Download Intel® Distribution of OpenVINO™ Toolkit」の欄に必要なpipコマンドが表示されるので、それを実行すれば良い。  
今回はC/C++使わないので、Runtimeの方はインストールしなくても大丈夫。
(NCS2等を使いたい場合はドライバインストールに必要だけど)  

> 「Don’t miss out」の欄にメールアドレスを入力してSubmitをクリックすると色々なメールが送られてくるようになるけど、必須ではない。  
> (以前は登録しないとダウンロードさせてくれなかったけど、今は完全フリーになったらしい)    

以下ではonnx,tensorflow2,caffeを選択した場合。  
openVINO 2022.1.0はpython 3.10をサポートしてないので、現時点でのそれ以下の最新版3.9．13を使うことにする。  

```bash
pyenv install 3.9.13
pyenv virtualenv 3.9.13 openvino_2022.1
pyenv local openvino_2022.1 
pip install --upgrade pip setuptools wheel
pip install openvino-dev[onnx,tensorflow2,caffe]==2022.1.0
```


## ファイル構成

| ファイル                     | 内容                      |
|------------------------------|---------------------------|
| ov_object_detection_ssd.py   | SSD処理スクリプト本体     |
| test.sh                      | テストスクリプト          |
| _result                      | 結果格納用ディレクトリ    |

## ``ov_object_detection_ssd.py``

SSD認識処理本体。  

USAGEは以下の通り。  

```
usage: ov_object_detection_ssd.py [-h] -m MODEL -i INPUT [--labels LABELS]
                                  [-d DEVICE] [-l CPU_EXTENSION]
                                  [-pt PROB_THRESHOLD] [--sync] [--save SAVE]
                                  [--time TIME] [--log LOG] [--no_disp]

optional arguments:
  -h, --help            Show this help message and exit.

Input Options:
  -m MODEL, --model MODEL
                        Required.
                        Path to an .xml file with a trained model.
  -i INPUT, --input INPUT
                        Required.
                        Path to a image/video file. 
                        (Specify 'cam' to work with camera)
  --labels LABELS       Optional.
                        Labels mapping file
                        Default is to change the extension of the modelfile
                        to '.labels'.
  -d DEVICE, --device DEVICE
                        Optional
                        Specify the target device to infer on; 
                        CPU, GPU, FPGA, HDDL or MYRIAD is acceptable.
                        The demo will look for a suitable plugin 
                        for device specified.
                        Default value is CPU
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        Optional.
                        Required for CPU custom layers. 
                        Absolute path to a shared library
                        with the kernels implementations.
                        以前はlibcpu_extension_avx2.so 指定が必須だったけど、
                        2020.1から不要になった

Output Options:
  --save SAVE           Optional.
                        Save result to specified file
  --time TIME           Optional.
                        Save time log to specified file
  --log LOG             Optional.
                        Save console log to specified file
  --no_disp             Optional.
                        without image display

Execution Options:
  -pt PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                        Optional.
                        Probability threshold for detections filtering
  --sync                Optional.
                        Start in sync mode
```

## ``test.sh``

``test.sh`` を実行するとパラメータに応じた設定で ``ov_object_detection_ssd.py`` を実行する。  
オプション/パラメータは以下の通り。

```
  ./test.sh [option_model] [option_log] [model_number | list | all | allall ] [input_file]
    ---- option_model ----
      -c | --cpu : CPUを使用
      -n | --ncs : NCS2を使用
    ---- option_log ----
      -l | --log : 実行ログを保存(model_number指定時のみ有効
                       all/allall指定時は指定の有無に関わらずログを保存
      --no_disp  : 表示を省略
                       all/allall指定時は指定の有無に関わらず表示を省略
      --sync     : 同期モードで実行
                       省略時は非同期モードで実行
    input_file 省略時はデフォルトの入力ファイルを使用
```

ログを保存する場合は、``_result`` ディレクトリに以下の形式で保存される。

| ファイル            | 内容                                            |
|---------------------|-------------------------------------------------|
| «モデル名»\_(cpu\|ncs2)\_(async\|sync).log      | 認識結果                                        |
| «モデル名»\_(cpu\|ncs2)\_(async\|sync).time     | フレーム毎の処理時間                            |
| «モデル名»\_(cpu\|ncs2)\_(async\|sync).time.average     | 各処理時間の平均値                     |
| «モデル名»\_(cpu\|ncs2)\_(async\|sync).«拡張子» | 認識結果画像(入力ファイルと同じフォーマット)    |

### 注意事項  
- ログファイル名はモデル名に応じて付与されるので、同じモデルで入力ファイルを変えて実行すると上書きされる。  
  
