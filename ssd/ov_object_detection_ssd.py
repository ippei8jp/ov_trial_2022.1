#!/usr/bin/env python
import sys
import os
import time
import logging as log
from argparse import ArgumentParser, SUPPRESS, RawTextHelpFormatter
import cv2
import numpy as np

# openvino.inference_engine のバージョン取得
from openvino.runtime import get_version as ov_get_version
ov_vession_str = ov_get_version()
print(ov_vession_str)               # バージョン2019には '2.1.custom_releases/2019/R～'という文字列が入っている
                                    # バージョン2020には '～-releases/2020/～'という文字列が入っている
                                    # バージョン2021には '～-releases/2021/～'という文字列が入っている
                                    # バージョン2022には '～-releases/2022/～'という文字列が入っている

from openvino.runtime import Core as ov_Core
from openvino.runtime import AsyncInferQueue as ov_AsyncInferQueue


# 表示フレームクラス ==================================================================
class DispFrame() :
    # カラーパレット(8bitマシン風。ちょっと薄目)
    COLOR_PALETTE = [   #   B    G    R 
                    ( 128, 128, 128),         # 0 (灰)
                    ( 255, 128, 128),         # 1 (青)
                    ( 128, 128, 255),         # 2 (赤)
                    ( 255, 128, 255),         # 3 (マゼンタ)
                    ( 128, 255, 128),         # 4 (緑)
                    ( 255, 255, 128),         # 5 (水色)
                    ( 128, 255, 255),         # 6 (黄)
                    ( 255, 255, 255)          # 7 (白)
                ]
    
    # ステータス領域サイズ
    STATUS_LINE_HIGHT   = 15                            # ステータス行の1行あたりの高さ
    STATUS_LINES        =  6                            # ステータス行数
    STATUS_PADDING      =  8                            # ステータス領域の余白
    STATUS_AREA_HIGHT   = STATUS_LINE_HIGHT * STATUS_LINES + STATUS_PADDING # ステータス領域の高さ
    
    def __init__(self, image, frame_number, all_frames) :
        # 画像にステータス表示領域を追加
        self.image = cv2.copyMakeBorder(image, 0, self.STATUS_AREA_HIGHT, 0, 0, cv2.BORDER_CONSTANT, (0,0,0))
        
        self.img_height = image.shape[0]
        self.img_width  = image.shape[1]
        
        self.frame_number = frame_number
        self.all_frames   = all_frames
        
    # 画像フレーム表示
    def disp_image(self) :
        cv2.imshow("Detection Results", self.image)                  # 表示
    
    # 検出枠の描画
    def draw_box(self, str, class_id, pt1, pt2) :
        left,  top    = pt1
        right, bottom = pt2
        
        # 対象物の枠とラベルの描画
        color = self.COLOR_PALETTE[class_id & 0x7]       # 表示色(IDの下一桁でカラーパレットを切り替える)
        cv2.rectangle(self.image,    (left, top     ), (right,      bottom), color,  2)
        cv2.rectangle(self.image,    (left, top + 20), (left + 160, top   ), color, -1)
        cv2.putText(self.image, str, (left, top + 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        
        return
    
    def STATUS_LINE_Y(self, line) : 
        return self.img_height + self.STATUS_LINE_HIGHT * (line + 1)
    
    def status_puts(self, message, line) :
        cv2.putText(self.image, message, (10, self.STATUS_LINE_Y(line)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 128, 128), 1)
    
    def disp_status(self, frame_time, inf_time, render_time, parse_time, is_async_mode) :
        frame_number_message    = f'frame_number   : {self.frame_number:5d} / {self.all_frames}'
        if frame_time == 0 :
            frame_time_message  =  'Frame time     : ---'
        else :
            frame_time_message  = f'Frame time     : {(frame_time * 1000):.3f} ms    {(1/frame_time):.2f} fps'  # ここは前のフレームの結果
        render_time_message     = f'Rendering time : {(render_time * 1000):.3f} ms'                             # ここは前のフレームの結果
        inf_time_message        = f'Inference time : {(inf_time * 1000):.3f} ms'
        parsing_time_message    = f'parse time     : {(parse_time * 1000):.3f} ms'
        async_mode_message      = f"Async mode is {' on' if is_async_mode else 'off'}"
        
        # 結果の書き込み
        self.status_puts(frame_number_message, 0)
        self.status_puts(inf_time_message,     1)
        self.status_puts(parsing_time_message, 2)
        self.status_puts(render_time_message,  3)
        self.status_puts(frame_time_message,   4)
        self.status_puts(async_mode_message,   5)

# ================================================================================
# 画像保存クラス ==================================================================
class ImageSave() :
    # 初期化
    def __init__(self, img_height, img_width) :
        # イメージ領域サイズ
        self.disp_width  = img_width
        self.disp_height = img_height + DispFrame.STATUS_AREA_HIGHT

        # 保存用ライタ
        self.writer = None
    
    # JPEGファイル書き込み
    def save_jpeg(self, jpeg_file, frame) :
        if jpeg_file :
            cv2.imwrite(jpeg_file, frame.image)
    
    # 動画ファイルのライタ生成
    def create_writer(self, filename, frame_rate) :
        # フォーマット
        fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.writer = cv2.VideoWriter(filename, fmt, frame_rate, (self.disp_width, self.disp_height))
    
    # 動画ファイル書き込み
    def write_image(self, frame) :
        if self.writer:
            self.writer.write(frame.image)
    
    # 動画ファイルのライタ解放
    def release_writer(self) :
        if self.writer:
            self.writer.release()

# ================================================================================

# コマンドラインパーサの構築 =====================================================
def build_argparser():
    parser = ArgumentParser(add_help=False, formatter_class=RawTextHelpFormatter)
    input_args = parser.add_argument_group('Input Options')
    output_args = parser.add_argument_group('Output Options')
    exec_args = parser.add_argument_group('Execution Options')
    parser.add_argument('-h', '--help', action='help', default=SUPPRESS, 
                        help='Show this help message and exit.')
    input_args.add_argument("-m", "--model", required=True, type=str, 
                        help="Required.\n"
                             "Path to an .xml file with a trained model.")
    input_args.add_argument("-i", "--input", required=True, type=str, 
                        help="Required.\n"
                             "Path to a image/video file. \n"
                             "(Specify 'cam' to work with camera)")
    input_args.add_argument("--labels", default=None, type=str, 
                        help="Optional.\n"
                             "Labels mapping file\n"
                             "Default is to change the extension of the modelfile\n"
                             "to '.labels'.")
    input_args.add_argument("-d", "--device", default="CPU", type=str, 
                        help="Optional\n"
                             "Specify the target device to infer on; \n"
                             "CPU, GPU, FPGA, HDDL or MYRIAD is acceptable.\n"
                             "The demo will look for a suitable plugin \n"
                             "for device specified.\n"
                             "Default value is CPU")
    input_args.add_argument("-l", "--cpu_extension", type=str, default=None, 
                        help="Optional.\n"
                             "Required for CPU custom layers. \n"
                             "Absolute path to a shared library\n"
                             "with the kernels implementations.\n"
                             "以前はlibcpu_extension_avx2.so 指定が必須だったけど、\n"
                             "2020.1から不要になった")
    exec_args.add_argument("-pt", "--prob_threshold", default=0.5, type=float, 
                        help="Optional.\n"
                             "Probability threshold for detections filtering")
    exec_args.add_argument("--sync", action='store_true', 
                        help="Optional.\n"
                             "Start in sync mode")
    output_args.add_argument("--save", default=None, type=str, 
                        help="Optional.\n"
                             "Save result to specified file")
    output_args.add_argument("--time", default=None, type=str, 
                        help="Optional.\n"
                             "Save time log to specified file")
    output_args.add_argument("--log", default=None, type=str,  
                        help="Optional.\n"
                             "Save console log to specified file")
    output_args.add_argument("--no_disp", action='store_true', 
                        help="Optional.\n"
                             "without image display")
    return parser
# ================================================================================

# コンソールとログファイルへの出力 ===============================================
def console_print(log_f, message, both=False, end=None) :
    if not (log_f and (not both)) :
        print(message,end=end)
    if log_f :
        log_f.write(message + '\n')

# 結果の取り出し
def parse_result(res, parse_params) :
    # tuple を個別の変数にバラす
    output_blob, prob_threshold, frame, all_results = parse_params
    
    # output tensorの取り出し
    res = res.get_tensor(output_blob).data[:]

    # print(res)
    #  -> 例：(1, 1, 100, 7)        100:バウンディングボックスの数
    #                               7: [image_id, label, conf, x_min, y_min, x_max, y_max]
    # データ構成は
    # https://docs.openvino.ai/2022.1/omz_models_model_ssdlite_mobilenet_v2.html
    # 等の「outputs」の 「Converted Model」を参照

    # バウンディングボックス毎の結果を取得
    res_array = res[0][0]

    raw_results = []
    for obj in res_array:
        conf = obj[2]                       # confidence for the predicted class(スコア)
        if conf > prob_threshold:           # 閾値より大きいものだけ処理
            class_id = int(obj[1])                      # クラスID
            left     = int(obj[3] * frame.img_width)    # バウンディングボックスの左上のX座標
            top      = int(obj[4] * frame.img_height)   # バウンディングボックスの左上のY座標
            right    = int(obj[5] * frame.img_width)    # バウンディングボックスの右下のX座標
            bottom   = int(obj[6] * frame.img_height)   # バウンディングボックスの右下のY座標
            
            pt1 = (left,  top   )
            pt2 = (right, bottom)
            
            raw_results.append({"class_id":class_id, "conf":conf, "pt1":pt1, "pt2":pt2})
    all_results[frame.frame_number] = {"frame": frame, "result":raw_results}
    
    inf_end = time.time()                               # 推論処理終了時刻          --------------------------------
    frame.inf_time = inf_end - frame.inf_start          # 推論処理時間
    return
# ================================================================================

# 表示&入力フレームの作成 =======================================================
def prepare_disp_and_input(cap, input_shape, frame_number, all_frames) :
    preprocess_start = time.time()                          # 前処理開始時刻            --------------------------------
    
    ret, img_frame = cap.read()    # フレームのキャプチャ
    if not ret :
        # キャプチャ失敗
        return ret, None, None
    
    # 入力用フレームの作成
    input_n, input_height, input_width, input_colors = input_shape
    in_frame = cv2.resize(img_frame, (input_width, input_height))       # リサイズ
    in_frame = in_frame.reshape(input_shape)                            # HWC → BHWC   ※ 旧バージョンはBCHWだった
    
    # 表示用フレームの作成
    disp_frame = DispFrame(img_frame, frame_number, all_frames)

    preprocess_end = time.time()                            # 前処理終了時刻            --------------------------------
    
    # 前処理時間を表示フレームに格納しておく
    disp_frame.preprocess_time = preprocess_end - preprocess_start     # 前処理時間
    
    
    return ret, in_frame, disp_frame
# ================================================================================

def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    
    # コマンドラインオプションの解析
    args = build_argparser().parse_args()
    
    # モデルファイル
    model_xml = args.model                                      # モデルファイル名(xml)
    # model_bin = os.path.splitext(model_xml)[0] + ".bin"         # モデルファイル名(bin)
    
    # 非表示設定
    no_disp = args.no_disp
    
    # ラベルファイル
    model_label = None
    if args.labels:
        model_label = args.labels
    else:
        model_label = os.path.splitext(model_xml)[0] + ".labels"
    if not os.path.isfile(model_label)  :
        log.warning("label file is not specified")
        model_label = None
    
    # 入力ファイル
    if args.input == 'cam':
        # カメラ入力の場合
        input_stream = 0
    else:
        input_stream = os.path.abspath(args.input)
        assert os.path.isfile(input_stream), "Specified input file doesn't exist"
    
    # ログファイル類の初期化
    time_f = None
    if args.time :
        time_f = open(args.time, mode='w')
        time_f.write(f'frame_number, frame_time, preprocess_time, inf_time, parse_time, render_time, wait_request, wait_time\n')     # 見出し行
    
    log_f = None
    if args.log :
        log_f = open(args.log, mode='w')
        log_f.write(f'command: {" ".join(sys.argv)}\n')     # 見出し行
    
    # 初期状態のsync/asyncモード切替
    is_async_mode = not args.sync
    
    # ラベルファイル読み込み
    labels_map = None
    if model_label:
        log.info(f"Loading label files: {model_label}")
        # ラベルファイルの読み込み
        with open(model_label, 'r') as f:
            labels_map = [x.strip() for x in f]
    
    # 推論エンジンの初期化
    log.info("Creating Inference Engine...")
    core = ov_Core()
    
    # 拡張ライブラリのロード(CPU使用時のみ)
    if args.cpu_extension and 'CPU' in args.device:
        log.info("Loading Extension Library...")
        core.add_extension(args.cpu_extension)
    
    # IR(Intermediate Representation ;中間表現)ファイル(.xml & .bin) の読み込み
    log.info(f"Loading model files: {model_xml}")
    model = core.read_model(model_xml)      # xmlとbinが同名ならbinは省略可能
    
    # 出力レイヤ数のチェックと名前の取得
    log.info("Check outputs")
    outputs = model.outputs
    assert len(outputs) == 1, "Demo supports only single output topologies" # 出力レイヤ数は1のみ対応
    output_blob = model.outputs[0].get_any_name()       # 出力レイヤ名

    # 入力レイヤ数のチェックと名前の取得
    log.info("Check inputs")
    inputs = model.inputs
    
    img_input_blob_name = None                  # 入力画像レイヤ
    img_info_input_blob_name = None             # 入力画像情報レイヤ
    
    for blob in inputs:
        blob_name = blob.get_any_name()         # 名前
        blob_shape = blob.shape                 # サイズ
        print(f'{blob_name}   {blob_shape}')
        if len(blob_shape) == 4:                # 4次元なら入力画像レイヤ
            img_input_blob_name = blob_name
            img_input_blob_shape = blob_shape
            input_n, img_input_height, img_input_width, input_colors = blob_shape
        elif len(blob_shape) == 2:              # 2次元なら入力画像情報レイヤ
           img_info_input_blob_name = blob_name
        else:                                   # それ以外のレイヤが含まれていたらエラー
            raise RuntimeError(f"Unsupported {len(blob_shape)} input layer '{blob_name}'. Only 2D and 4D input layers are supported")
    
    # キャプチャデバイス
    cap = cv2.VideoCapture(input_stream)
    
    # 幅と高さを取得
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # フレームレート(1フレームの時間単位はミリ秒)の取得
    org_frame_rate = int(cap.get(cv2.CAP_PROP_FPS))                 # オリジナルのフレームレート
    org_frame_time = 1.0 / cap.get(cv2.CAP_PROP_FPS)                # オリジナルのフレーム時間
    # フレーム数
    all_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_frames = 1 if all_frames != -1 and all_frames < 0 else all_frames   # -1なら静止画
    
    # 画像保存インスタンスの作成
    img_save = ImageSave(img_height, img_width)
    
    # 画像保存オプション
    # writer = None
    jpeg_file = None
    if args.save :
        if all_frames == 1 :
            jpeg_file = args.save
        else :
            img_save.create_writer(args.save, org_frame_rate)
    
    # 初期状態のsync/asyncモードを表示
    log.info(f"Starting inference in {'async' if  is_async_mode else 'sync'} mode...")
    
    # 1フレーム表示後の待ち時間
    wait_key_time = 1
    
    # 動画か静止画かをチェック
    if all_frames == 1:
        # 1フレーム -> 静止画
        is_async_mode = False       # 同期モードにする
        wait_key_time = 0           # 永久待ち
    
    cur_request_id = 0          # 同期モードで初期化(ID=0のみ使用)
    next_request_id = 0
    if is_async_mode:
        # 非同期モードでは0と1を入れ替えて使う
        cur_request_id = 0
        next_request_id = 1
    
    # 表示フレーム保持用
    disp_frame = [None, None]
    
    # モデルのコンパイル
    log.info("Loading model to the plugin...")
    compiled_model = core.compile_model(model, args.device)

    # 推論キューの作成
    if is_async_mode: 
        # 非同期モードなら2面
        async_queue = ov_AsyncInferQueue(compiled_model, 2)
    else :
        # 同期モードなら1面
        async_queue = ov_AsyncInferQueue(compiled_model, 1)
    
    # 推論開始
    log.info("Starting inference...")
    print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")
    
    # 実行時間測定用変数の初期化
    frame_time = 0
    preprocess_time = 0
    inf_time = 0
    parse_time = 0
    render_time = 0
    
    # 現在のフレーム番号
    infer_frame_number = 1      # 推論用フレーム番号
    disp_frame_number  = 1      # 表示用フレーム番号
    
    if is_async_mode:
        # 非同期モード時は最初のフレームの推論を予約しておく
        feed_dict = {}
        ret, feed_dict[img_input_blob_name], disp_frame[cur_request_id] = prepare_disp_and_input(cap, img_input_blob_shape, infer_frame_number, all_frames)
        if not ret :
            print("failed to capture first frame.")
            sys.exit(1)
        if img_info_input_blob_name:
            feed_dict[img_info_input_blob_name] = np.array([[img_input_height, img_input_width, 1]])
        
        # 推論開始時刻を表示フレームに格納しておく
        disp_frame[cur_request_id].inf_start = time.time()                                 # 推論処理開始時刻          --------------------------------
        
        # 推論予約 =============================================================================
        async_queue[cur_request_id].start_async(feed_dict)
        
        # フレーム番号更新
        infer_frame_number += 1
        
    
    # フレーム測定用タイマ
    prev_time = time.time()
    
    # すべての結果
    all_results = {}
    
    while True:
        # 画像の前処理 =============================================================================
        # 現在のフレーム番号表示
        print(f'frame_number: {infer_frame_number:5d} / {all_frames}', end='\r')
        # 画像キャプチャと表示/入力用画像を作成
        # 非同期モード時は次のフレームとして
        feed_dict = {}
        ret, feed_dict[img_input_blob_name], disp_frame[next_request_id] = prepare_disp_and_input(cap, img_input_blob_shape, infer_frame_number, all_frames)
        if not ret:
            # キャプチャ失敗
            break
        if img_info_input_blob_name:
            feed_dict[img_info_input_blob_name] = np.array([[img_input_height, img_input_width, 1]])
        
        # 推論開始時刻を表示フレームに格納しておく
        disp_frame[next_request_id].inf_start = time.time()                                 # 推論処理開始時刻          --------------------------------
        
        # 推論予約 =============================================================================
        parse_params = (output_blob, args.prob_threshold, disp_frame[next_request_id], all_results)
        async_queue[next_request_id].start_async(feed_dict, parse_params)
        
        # フレーム番号更新
        infer_frame_number += 1
        
        # 推論結果待ち =============================================================================
        if async_queue[cur_request_id].wait_for(-1) :           # -1: 永久待ち
            parse_params = (output_blob, args.prob_threshold, disp_frame[cur_request_id], all_results)
            parse_result(async_queue[cur_request_id], parse_params)
            frame_result = all_results.pop(disp_frame_number)       # 辞書から要素を取り出して削除
            
            # 検出結果の解析 =============================================================================
            parse_start = time.time()                           # 解析処理開始時刻          --------------------------------
            
            # 現在の表示フレーム
            cur_frame = frame_result["frame"]
            
            # 処理時間
            preprocess_time = cur_frame.preprocess_time
            inf_time        = cur_frame.inf_time
            
            for rst in frame_result["result"] :
                # 結果を個別の変数にバラす
                class_id = rst["class_id"]
                conf     = rst["conf"]
                pt1      = rst["pt1"]
                pt2      = rst["pt2"]
                
                # 検出結果の文字列化
                # ラベルが定義されていればラベルを読み出し、なければclass ID
                if labels_map :
                    if len(labels_map) > class_id :
                        class_name = labels_map[class_id]
                    else :
                        class_name = str(class_id)
                else :
                    class_name = str(class_id)
                
                # 結果をログファイルorコンソールに出力
                console_print(log_f, f'{disp_frame_number:3}:Class={class_name:15}({class_id:3}) Confidence={conf:4f} Location=({pt1[0]},{pt1[1]})-({pt2[0]},{pt2[1]})', False)
                
                # 検出枠の描画
                box_str = f'{class_name} {round(conf * 100, 1)}%'
                cur_frame.draw_box(box_str, class_id, pt1, pt2)
            
            parse_end = time.time()                             # 解析処理終了時刻          --------------------------------
            parse_time = parse_end - parse_start                # 解析処理開始時間
            
            # 結果の表示 =============================================================================
            render_start = time.time()                          # 表示処理開始時刻          --------------------------------
            # 測定データの表示
            cur_frame.disp_status(frame_time, inf_time, render_time, parse_time, is_async_mode)
            
            # 表示
            if not no_disp :
                cur_frame.disp_image()        # 表示
            
            # 画像の保存
            if jpeg_file :
                img_save.save_jpeg(jpeg_file, cur_frame)
            # 保存が設定されていか否かはメソッド内でチェック
            img_save.write_image(cur_frame)
            render_end = time.time()                            # 表示処理終了時刻          --------------------------------
            render_time = render_end - render_start             # 表示処理時間
        
        # 非同期モードではフレームバッファ入れ替え
        if is_async_mode:
            cur_request_id, next_request_id = next_request_id, cur_request_id
        
        # キー入力取得 =============================================================================
        wait_start = time.time()                            # キー待ち開始時刻    --------------------------------
        key = cv2.waitKey(wait_key_time)
        if key == 27:
            # ESCキー
            break
        wait_end = time.time()                              # キー待ち終了時刻    --------------------------------
        wait_time = wait_end - wait_start                   # キー待ち時間
        
        # フレーム処理終了 =============================================================================
        cur_time = time.time()                              # 現在のフレーム処理完了時刻
        frame_time = cur_time - prev_time                   # 1フレームの処理時間
        prev_time = cur_time
        if time_f :
            time_f.write(f'{disp_frame_number:5d}, {frame_time * 1000:.3f}, {preprocess_time * 1000:.3f}, {inf_time * 1000:.3f}, {parse_time * 1000:.3f}, {render_time * 1000:.3f}, {wait_key_time}, {wait_time * 1000:.3f}\n')
        
        # 表示フレーム更新
        disp_frame_number += 1
    
    # 後片付け
    if time_f :
        time_f.close()
    
    if log_f :
        log_f.close()
    
    # 保存が設定されていか否かはメソッド内でチェック
    img_save.release_writer()
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    sys.exit(main() or 0)
