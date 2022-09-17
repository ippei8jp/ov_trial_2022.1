# 顔認識実行

## ファイル構成

| ファイル                       | 内容                                |
|--------------------------------|-------------------------------------|
| ov_face_detection.py           | 顔認識処理スクリプト本体            |
| DispFrame.py                   | 表示/保存関連処理                   |
| model/sync_model_base.py       | 同期処理用モデルラッパの基底クラス  |
| model/model_face_detect.py     | 顔認識モデルラッパクラス            |
| model/model_face_landmark5.py  | 特徴点検出(5点)モデル ラッパクラス  |
| model/model_face_landmark35.py | 特徴点検出(35点)モデル ラッパクラス |
| model/model_face_headpose.py   | 顔向き推定モデル ラッパクラス       |
| test.sh                        | テストスクリプト                    |
| _result                        | 結果格納用ディレクトリ              |

## ``ov_face_detection.py``

顔認識処理本体。  

USAGEは以下の通り。  

```
usage: ov_face_detection.py [-h] -m MODEL [-m_lm5 MODEL_LM5]
                            [-m_lm35 MODEL_LM35] [-m_hp MODEL_HP] -i INPUT
                            [-d DEVICE] [-d_lm5 DEVICE_LM5]
                            [-d_lm35 DEVICE_LM35] [-d_hp DEVICE_HP]
                            [-l CPU_EXTENSION] [-pt PROB_THRESHOLD]
                            [--save SAVE] [--time TIME] [--log LOG]
                            [--no_disp]

optional arguments:
  -h, --help            Show this help message and exit.

Input Options:
  -m MODEL, --model MODEL
                        Required.
                        Path to an .xml file with a trained model.
  -m_lm5 MODEL_LM5, --model_lm5 MODEL_LM5
                        Optional.
                        Path to an .xml file for landmark detection (5point) model.
  -m_lm35 MODEL_LM35, --model_lm35 MODEL_LM35
                        Optional.
                        Path to an .xml file for landmark detection (35point) model.
  -m_hp MODEL_HP, --model_hp MODEL_HP
                        Optional.
                        Path to an .xml file for head pose estimation model.
  -i INPUT, --input INPUT
                        Required.
                        Path to a image/video file. 
                        (Specify 'cam' to work with camera)
  -d DEVICE, --device DEVICE
                        Optional
                        Specify the target device to infer on; 
                        CPU, GPU, FPGA, HDDL or MYRIAD is acceptable.
                        The demo will look for a suitable plugin 
                        for device specified.
                        Default value is CPU
  -d_lm5 DEVICE_LM5, --device_lm5 DEVICE_LM5
                        Optional
                        Specify the target device to infer for landmark detection (5point)
                        Default value is CPU
  -d_lm35 DEVICE_LM35, --device_lm35 DEVICE_LM35
                        Optional
                        Specify the target device to infer for landmark detection (35point)
                        Default value is CPU
  -d_hp DEVICE_HP, --device_hp DEVICE_HP
                        Optional
                        Specify the target device to infer for head pose estimation
                        Default value is CPU
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        Optional.
                        Required for CPU custom layers. 
                        Absolute path to a shared library
                        with the kernels implementations.

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
```

## ``test.sh``

``test.sh`` を実行するとパラメータに応じた設定で ``ov_face_detection.py`` を実行する。  
オプション/パラメータは以下の通り。

```
==== USAGE ====
  ./test.sh [option_model] [option_log] [model_number | list | all | allall ] [input_file]
    ---- option_model ----
      -c | --cpu : CPUを使用
      -n | --ncs : NCS2を使用
    ---- option_append_model ----
      --lm5      : 特徴点検出(5点)をCPUで実行
      --lm5_ncs  : 特徴点検出(5点)をNCS2で実行
      --lm35     : 特徴点検出(35点)をCPUで実行
      --lm35_ncs : 特徴点検出(35点)をNCS2で実行
      --hp       : 顔の向き推定をCPUで実行
      --hp_ncs   : 顔の向き推定をNCS2で実行
    ---- option_log ----
      --no_disp  : 表示を省略
                       all/allall指定時は指定の有無に関わらず表示を省略
      -l | --log : 実行ログを保存(model_number指定時のみ有効
                       --no_disp指定時は指定の有無に関わらずログを保存
                       all/allall指定時は指定の有無に関わらずログを保存
    input_file 省略時はデフォルトの入力ファイルを使用
 
==== MODEL LIST ====
0 : face-detection-retail-0004
1 : face-detection-retail-0005
2 : face-detection-retail-0044
3 : face-detection-adas-0001
4 : face-detection-0200
5 : face-detection-0202
6 : face-detection-0204
7 : face-detection-0205
8 : face-detection-0206

```

ログを保存する場合は、``_result`` ディレクトリに以下の形式で保存される。

| ファイル            | 内容                                            |
|---------------------|-------------------------------------------------|
| «モデル名»\-«顔検出実行デバイス»-lm5\_«5点検出実行デバイス»-lm35\_«35点検出実行デバイス»-hp\_«顔向き推定実行デバイス».log          | 認識結果                                        |
| «モデル名»\-«顔検出実行デバイス»-lm5\_«5点検出実行デバイス»-lm35\_«35点検出実行デバイス»-hp\_«顔向き推定実行デバイス».time         | フレーム毎の処理時間                            |
| «モデル名»\-«顔検出実行デバイス»-lm5\_«5点検出実行デバイス»-lm35\_«35点検出実行デバイス»-hp\_«顔向き推定実行デバイス».time.average | 各処理時間の平均値                     |
| «モデル名»\-«顔検出実行デバイス»-lm5\_«5点検出実行デバイス»-lm35\_«35点検出実行デバイス»-hp\_«顔向き推定実行デバイス».«拡張子»     | 認識結果画像(入力ファイルと同じフォーマット)    |

### 注意事項  
- ログファイル名はモデル名に応じて付与されるので、同じモデルで入力ファイルを変えて実行すると上書きされる。  
  
