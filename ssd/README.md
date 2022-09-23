# SSD実行

## ファイル構成

| ファイル                     | 内容                                   |
|------------------------------|----------------------------------------|
| ov_object_detection_ssd.py   | SSD処理スクリプト本体                  |
| DispFrame.py                 | 表示/保存関連処理                      |
| model/async_model_base.py    | 非同期処理用モデルラッパの基底クラス   |
| model/model_ssd_detect.py    | SSDモデルラッパクラス                  |
| test.sh                      | テストスクリプト                       |
| _result                      | 結果格納用ディレクトリ                 |

## ``ov_object_detection_ssd.py``

SSD認識処理本体。  

USAGEは以下の通り。  

```
usage: ov_object_detection_ssd.py [-h] -i INPUT [-l CPU_EXTENSION] -m MODEL
                                  [--labels LABELS] [-d DEVICE]
                                  [--queue_num QUEUE_NUM]
                                  [-t_detect THRESHOLD_DETECT] [--save SAVE]
                                  [--time TIME] [--log LOG] [--no_disp]

optional arguments:
  -h, --help            Show this help message and exit.
  -i INPUT, --input INPUT
                        Required.
                        Path to a image/video file.
                        (Specify 'cam' to work with camera)
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        Optional.
                        Required for CPU custom layers.
                        Absolute path to a shared library
                        with the kernels implementations.

SSD Options:
  -m MODEL, --model MODEL
                        Required.
                        Path to an .xml file with a trained model.
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
  --queue_num QUEUE_NUM
                        Optional.
                        Number of async infer queues
  -t_detect THRESHOLD_DETECT, --threshold_detect THRESHOLD_DETECT
                        Optional.
                        Probability threshold for detections filtering

Output Options:
  --save SAVE           Optional.
                        Save result to specified file
  --time TIME           Optional.
                        Save time log to specified file
  --log LOG             Optional.
                        Save console log to specified file
  --no_disp             Optional.
                        without image display
```

## ``test.sh``

``test.sh`` を実行するとパラメータに応じた設定で ``ov_object_detection_ssd.py`` を実行する。  
オプション/パラメータは以下の通り。

```
==== USAGE ====
  ./test.sh [option_model] [option_log] [model_number | list | all | allall] [input_file]
    ---- option_model ----
      -c | --cpu : CPUを使用
      -n | --ncs : NCS2を使用
    ---- option_log ----
      --no_disp  : 表示を省略
                       all/allall指定時は指定の有無に関わらず表示を省略
      -l | --log : 実行ログを保存(model_number指定時のみ有効
                       --no_disp指定時は指定の有無に関わらずログを保存
                       all/allall指定時は指定の有無に関わらずログを保存
    ---- option_other ----
      --allow_long_proc  : 実行時間の長いモデルの実行を許可
      --queue_num <queue_num>  : 推論キューの数を設定(default:2)
      --check            : 実行コマンドのチェックのみ行う
    input_file 省略時はデフォルトの入力ファイルを使用

==== MODEL LIST ====
0 : mobilenet-ssd
1 : ssd300
2 : ssd512
3 : ssd_mobilenet_v1_coco
4 : ssd_mobilenet_v1_fpn_coco
5 : ssdlite_mobilenet_v2
6 : person-vehicle-bike-detection-2000
7 : person-vehicle-bike-detection-2001
8 : person-vehicle-bike-detection-2002
9 : person-vehicle-bike-detection-2003
10 : person-vehicle-bike-detection-2004
11 : person-vehicle-bike-detection-crossroad-0078
12 : person-vehicle-bike-detection-crossroad-1016
```

ログを保存する場合は、``_result`` ディレクトリに以下の形式で保存される。

| ファイル                                                  | 内容                                          |
|-----------------------------------------------------------|-----------------------------------------------|
| «モデル名»\-(cpu\|ncs2)\-queue_«queue_num».log            | 認識結果                                      |
| «モデル名»\-(cpu\|ncs2)\-queue_«queue_num».time           | フレーム毎の処理時間                          |
| «モデル名»\-(cpu\|ncs2)\-queue_«queue_num».time.average   | 各処理時間の平均値                            |
| «モデル名»\-(cpu\|ncs2)\-queue_«queue_num».«拡張子»       | 認識結果画像(入力ファイルと同じフォーマット)  |

### 注意事項  
- ログファイル名はモデル名に応じて付与されるので、同じモデルで入力ファイルを変えて実行すると上書きされる。  
  
