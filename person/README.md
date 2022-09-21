# 顔認識実行

## ファイル構成

| ファイル                       | 内容                                |
|--------------------------------|-------------------------------------|
| ov_person_detection.py         | 人物認識処理スクリプト本体          |
| DispFrame.py                   | 表示/保存関連処理                   |
| model/sync_model_base.py       | 同期処理用モデルラッパの基底クラス  |
| model/model_person_detect.py   | 人物認識モデルラッパクラス          |
| model/model_person_reid.py     | 人物同定モデルラッパクラス          |
| model/model_person_attr.py     | 属性検出モデルラッパクラス          |
| test.sh                        | テストスクリプト                    |
| _result                        | 結果格納用ディレクトリ              |

## ``ov_person_detection.py``

人物認識処理本体。  

USAGEは以下の通り。  

```
usage: ov_person_detection.py [-h] -i INPUT [-l CPU_EXTENSION] -m MODEL
                              [-d DEVICE] [-t_detect THRESHOLD_DETECT]
                              [-m_reid MODEL_REID] [-d_reid DEVICE_REID]
                              [-t_reid THRESHOLD_REID] [-m_attr MODEL_ATTR]
                              [-d_attr DEVICE_ATTR] [-t_attr THRESHOLD_ATTR]
                              [--save SAVE] [--time TIME] [--log LOG]
                              [--no_disp]

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

person detect Options:
  -m MODEL, --model MODEL
                        Required.
                        Path to an .xml file with a trained model.
  -d DEVICE, --device DEVICE
                        Optional
                        Specify the target device to infer on;
                        CPU, GPU, FPGA, HDDL or MYRIAD is acceptable.
                        The demo will look for a suitable plugin
                        for device specified.
                        Default value is CPU
  -t_detect THRESHOLD_DETECT, --threshold_detect THRESHOLD_DETECT
                        Optional.
                        Probability threshold for detections filtering

person reidentification Options:
  -m_reid MODEL_REID, --model_reid MODEL_REID
                        Optional.
                        Path to an .xml file for person reidentification model.
  -d_reid DEVICE_REID, --device_reid DEVICE_REID
                        Optional
                        Specify the target device to infer for person reidentification
                        Default value is CPU
  -t_reid THRESHOLD_REID, --threshold_reid THRESHOLD_REID
                        Optional.
                        Probability threshold for person reidentification

person attributes Options:
  -m_attr MODEL_ATTR, --model_attr MODEL_ATTR
                        Optional.
                        Path to an .xml file for person attributes model.
  -d_attr DEVICE_ATTR, --device_attr DEVICE_ATTR
                        Optional
                        Specify the target device to infer for person attributes
                        Default value is CPU
  -t_attr THRESHOLD_ATTR, --threshold_attr THRESHOLD_ATTR
                        Optional.
                        Probability threshold for person attributes

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

``test.sh`` を実行するとパラメータに応じた設定で ``ov_person_detection.py`` を実行する。  
オプション/パラメータは以下の通り。

```
==== USAGE ====
  ./test.sh [option_model] [option_log] [model_number | list | all | allall ] [input_file]
    ---- option_model ----
      -c | --cpu : CPUを使用
      -n | --ncs : NCS2を使用
    ---- option_append_model ----
      --reid <model_no>     : 人物同定処理をCPUで実行
      --reid_ncs <model_no> : 人物同定処理をNCS2で実行
      --attr <model_no>     : 属性検出をCPUで実行
      --attr_ncs <model_no> : 属性検出をNCS2で実行
    ---- option_log ----
      --no_disp  : 表示を省略
                       all/allall指定時は指定の有無に関わらず表示を省略
      -l | --log : 実行ログを保存(model_number指定時のみ有効
                       --no_disp指定時は指定の有無に関わらずログを保存
                       all/allall指定時は指定の有無に関わらずログを保存
    ---- option_other ----
      --allow_long_proc  : 実行時間の長いモデルの実行を許可
    input_file 省略時はデフォルトの入力ファイルを使用

==== MODEL LIST ====
0 : person-detection-0200
1 : person-detection-0201
2 : person-detection-0202
3 : person-detection-0203
4 : person-detection-0301
5 : person-detection-0302
6 : person-detection-0303
7 : person-detection-0106
8 : person-detection-asl-0001
9 : person-detection-retail-0002
10 : person-detection-retail-0013

==== --model_reid MODEL LIST ====
0 : person-reidentification-retail-0277
1 : person-reidentification-retail-0286
2 : person-reidentification-retail-0287
3 : person-reidentification-retail-0288

==== --model_attr MODEL LIST ====
0 : person-attributes-recognition-crossroad-0230
1 : person-attributes-recognition-crossroad-0234
2 : person-attributes-recognition-crossroad-0238

```

ログを保存する場合は、``_result`` ディレクトリに以下の形式で保存される。

| ファイル            | 内容                                            |
|---------------------|-------------------------------------------------|
| «モデル名»\-«人物検出実行デバイス»-reid\_«人物同定モデル番号»_«人物同定実行デバイス»-attr\_«属性検出モデル番号»_«属性検出実行デバイス».log          | 認識結果                                        |
| «モデル名»\-«人物検出実行デバイス»-reid\_«人物同定モデル番号»_«人物同定実行デバイス»-attr\_«属性検出モデル番号»_«属性検出実行デバイス».time         | フレーム毎の処理時間                            |
| «モデル名»\-«人物検出実行デバイス»-reid\_«人物同定モデル番号»_«人物同定実行デバイス»-attr\_«属性検出モデル番号»_«属性検出実行デバイス».time.average | 各処理時間の平均値                              |
| «モデル名»\-«人物検出実行デバイス»-reid\_«人物同定モデル番号»_«人物同定実行デバイス»-attr\_«属性検出モデル番号»_«属性検出実行デバイス».«拡張子»     | 認識結果画像(入力ファイルと同じフォーマット)    |

### 注意事項  
- ログファイル名はモデル名に応じて付与されるので、同じモデルで入力ファイルを変えて実行すると上書きされる。  
  
