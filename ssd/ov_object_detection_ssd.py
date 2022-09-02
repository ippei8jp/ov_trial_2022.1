#!/usr/bin/env python3
import sys
import os
import time
import logging as log
from argparse import ArgumentParser, SUPPRESS, RawTextHelpFormatter
import cv2
import numpy as np

# openVINOモジュール
from openvino.runtime import get_version        as ov_get_version
from openvino.runtime import Core               as ov_Core
from openvino.runtime import AsyncInferQueue    as ov_AsyncInferQueue


# コンソールとログファイルへの出力 ===============================================
def console_print(log_f, message, both=False, end=None) :
    if (not log_f) or both :
        print(message,end=end)
    if log_f :
        print(message,end=end, file=log_f)
# ================================================================================

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
    
    def __init__(self, image, frame_number, all_frames, is_async_mode, queue_num) :
        # 画像にステータス表示領域を追加
        self.image = cv2.copyMakeBorder(image, 0, self.STATUS_AREA_HIGHT, 0, 0, cv2.BORDER_CONSTANT, (0,0,0))
        
        # イメージサイズ
        self.img_height = image.shape[0]
        self.img_width  = image.shape[1]
        
        # フレーム番号
        self.frame_number = frame_number
        self.all_frames   = all_frames
        
        # 同期モード/キュー数
        self.is_async_mode = is_async_mode
        self.queue_num     = queue_num
        
        # 処理時間計測用変数
        self.preprocess_start     = 0
        self.infer_start          = 0
        self.postprocess_start    = 0
        
        self.frame_time          = 0
        self.preprocess_time     = 0
        self.infer_time          = 0
        self.postprocess_time    = 0
        
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
    
    # ステータス表示行座標
    def STATUS_LINE_Y(self, line) : 
        return self.img_height + self.STATUS_LINE_HIGHT * (line + 1)
    
    # ステータス文字列出力
    def status_puts(self, message, line) :
        cv2.putText(self.image, message, (10, self.STATUS_LINE_Y(line)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 128, 128), 1)
    
    # ステータス表示
    def disp_status(self) :
        # ステータス文字列生成
        frame_number_message    = f'frame_number     : {self.frame_number:5d} / {self.all_frames}'
        if self.frame_time == 0 :
            frame_time_message  =  'Frame time       : ---'
        else :
            frame_time_message      = f'Frame time       : {      self.frame_time:.3f} ms'
        preprocess_time_message     = f'preprocess time  : { self.preprocess_time:.3f} ms'
        infer_time_message          = f'Inference time   : {      self.infer_time:.3f} ms'
        postprocess_time_message    = f'postprocess time : {self.postprocess_time:.3f} ms'
        if self.is_async_mode : 
            async_mode_message      = f"Async mode       queue_num : {self.queue_num}"
        else :
            async_mode_message      = f"Sync mode"
        
        # 文字列の書き込み
        self.status_puts(frame_number_message,      0)
        self.status_puts(frame_time_message,        1)
        self.status_puts(preprocess_time_message,   2)
        self.status_puts(infer_time_message,        3)
        self.status_puts(postprocess_time_message,  4)
        self.status_puts(async_mode_message,        5)

    # 処理時間関連処理
    def set_frame_time(self, frame_time) :
        self.frame_time = frame_time * 1000     # msec単位に変換
    
    def start_preprocess(self, cur_time=None) :
        if cur_time is None :
            cur_time = time.perf_counter()
        self.preprocess_start   = cur_time
    def end_preprocess(self, cur_time=None) :
        if cur_time is None :
            cur_time = time.perf_counter()
        self.preprocess_time     = (cur_time - self.preprocess_start) * 1000     # msec単位に変換
    
    def start_infer(self, cur_time=None) :
        if cur_time is None :
            cur_time = time.perf_counter()
        self.infer_start        = cur_time
    def end_infer(self, cur_time=None) :
        if cur_time is None :
            cur_time = time.perf_counter()
        self.infer_time         = (cur_time - self.infer_start) * 1000           # msec単位に変換
    
    def start_postprocess(self, cur_time=None) :
        if cur_time is None :
            cur_time = time.perf_counter()
        self.postprocess_start  = cur_time
    def end_postprocess(self, cur_time=None) :
        if cur_time is None :
            cur_time = time.perf_counter()
        self.postprocess_time    = (cur_time - self.postprocess_start) * 1000    # msec単位に変換
    
    # 処理時間記録
    def write_time_data(self, time_f) :
        if time_f :
            time_f.write(f'{self.frame_number:5d}, {self.frame_time:.3f}, {self.preprocess_time:.3f}, {self.infer_time:.3f}, {self.postprocess_time:.3f}\n')

# ================================================================================

# 画像保存クラス =================================================================
class ImageSave() :
    # 初期化
    def __init__(self, img_height, img_width) :
        # イメージ領域サイズ
        self.disp_width  = img_width
        self.disp_height = img_height + DispFrame.STATUS_AREA_HIGHT # ステータス領域分を加算しておく
        
        # JPEGファイル名
        self.jpeg_file = None
        # 保存用ライタ
        self.writer    = None
    
    # JPEGファイル名の設定
    def set_jpeg(self, filename) :
        self.jpeg_file = filename
    
    # 動画ファイルのライタ生成
    def create_writer(self, filename, frame_rate) :
        # フォーマット
        fmt = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(filename, fmt, frame_rate, (self.disp_width, self.disp_height))
    
    # 動画ファイル書き込み
    def write_image(self, frame) :
        if self.jpeg_file :
            cv2.imwrite(self.jpeg_file, frame.image)
        if self.writer:
            self.writer.write(frame.image)
    
    # 動画ファイルのライタ解放
    def release_writer(self) :
        if self.writer:
            self.writer.release()
# ================================================================================

# 前処理 =======================================================
class InferPreProcess() :
    # 初期化
    def __init__(self, image_blob_name, image_blob_shape, image_blob_format_NCHW, image_info_blob_name, all_frames, is_async_mode, queue_num) :
        self.image_blob_shape       = image_blob_shape       
        self.image_blob_format_NCHW = image_blob_format_NCHW 
        if image_blob_format_NCHW :
            _, _, self.input_height, self.input_width = image_blob_shape
        else :
            _, self.input_height, self.input_width, _ = image_blob_shape
        
        self.image_blob_name         = image_blob_name
        self.image_info_blob_name    = image_info_blob_name
        
        self.all_frames        = all_frames        
        self.is_async_mode     = is_async_mode     
        self.queue_num         = queue_num         
    
    def __call__(self, img_frame, frame_number) :
        # 表示用フレームの作成
        disp_frame = DispFrame(img_frame, frame_number, self.all_frames, self.is_async_mode, self.queue_num)
        
        # 入力用フレームの作成
        in_frame = cv2.resize(img_frame, (self.input_width, self.input_height))     # リサイズ
        if self.image_blob_format_NCHW :
             in_frame = in_frame.transpose((2, 0, 1))                               # HWC → CHW
        in_frame = in_frame.reshape(self.image_blob_shape)                               # HWC → BHWC or CHW → BCHW
        
        feed_dict = {self.image_blob_name: in_frame}
        if self.image_info_blob_name:
            feed_dict[self.image_info_blob_name] = np.array([[self.img_input_height, self.img_input_width, 1]])
        
        return feed_dict, disp_frame
# ================================================================================

# 後処理 =======================================================
class InferPostProcess() :
    # 初期化
    def __init__(self, labels_map, log_f) :
        self.labels_map  = labels_map
        self.log_f       = log_f
    
    def __call__(self, infer_rst) :
        # 現在の表示フレーム
        cur_frame = infer_rst["frame"]
        
        for rst in infer_rst["result"] :
            # 結果を個別の変数にバラす
            class_id = rst["class_id"]
            conf     = rst["conf"]
            pt1      = rst["pt1"]
            pt2      = rst["pt2"]
            
            # 検出結果の文字列化
            # ラベルが定義されていればラベルを読み出し、なければclass ID
            if self.labels_map :
                if len(self.labels_map) > class_id :
                    class_name = self.labels_map[class_id]
                else :
                    class_name = str(class_id)
            else :
                class_name = str(class_id)
            
            # 結果をログファイルorコンソールに出力
            console_print(self.log_f, f'{cur_frame.frame_number:3}:Class={class_name:15}({class_id:3}) Confidence={conf:4f} Location=({pt1[0]},{pt1[1]})-({pt2[0]},{pt2[1]})', False)
            
            # 検出枠の描画
            box_str = f'{class_name} {round(conf * 100, 1)}%'
            cur_frame.draw_box(box_str, class_id, pt1, pt2)
# ================================================================================

# 推論処理のコールバック ===============================================
class InferCallback() :
    # 初期化
    def __init__(self, output_blob, prob_threshold, infer_results) :
        self.output_blob    = output_blob
        self.prob_threshold = prob_threshold
        self.infer_results  = infer_results
    
    def __call__(self, res, parse_params) :
        # tuple を個別の変数にバラす
        frame = parse_params
        
        # output tensorの取り出し
        res = res.get_tensor(self.output_blob).data[:]
    
        # print(res.shape)
        #  -> 例：(1, 1, 100, 7)        100:バウンディングボックスの数
        #                               7: [image_id, label, conf, x_min, y_min, x_max, y_max]
        # データ構成は
        # https://docs.openvino.ai/2022.1/omz_models_model_ssdlite_mobilenet_v2.html
        # 等の「outputs」の 「Converted Model」を参照
    
        # バウンディングボックス毎の結果を取得
        res_array = res[0][0]
    
        results = []
        for obj in res_array:
            conf = obj[2]                       # confidence for the predicted class(スコア)
            if conf > self.prob_threshold:           # 閾値より大きいものだけ処理
                class_id = int(obj[1])                      # クラスID
                left     = int(obj[3] * frame.img_width)    # バウンディングボックスの左上のX座標
                top      = int(obj[4] * frame.img_height)   # バウンディングボックスの左上のY座標
                right    = int(obj[5] * frame.img_width)    # バウンディングボックスの右下のX座標
                bottom   = int(obj[6] * frame.img_height)   # バウンディングボックスの右下のY座標
                
                pt1 = (left,  top   )
                pt2 = (right, bottom)
                
                results.append({"class_id":class_id, "conf":conf, "pt1":pt1, "pt2":pt2})
        self.infer_results[frame.frame_number] = {"frame": frame, "result":results}
        return
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
                             "with the kernels implementations.")
    exec_args.add_argument("-pt", "--prob_threshold", default=0.5, type=float, 
                        help="Optional.\n"
                             "Probability threshold for detections filtering")
    exec_args.add_argument("--sync", action='store_true', 
                        help="Optional.\n"
                             "Start in sync mode")
    exec_args.add_argument("--queue_num", default=2, type=int, 
                        help="Optional.\n"
                             "Number of async infer queues")
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

# メイン処理 =====================================================================
def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    
    # openvino.inference_engine のバージョン取得
    ov_vession_str = ov_get_version()
    log.info(f"openVINO vertion : {ov_vession_str}")

    # コマンドラインオプションの解析
    args = build_argparser().parse_args()
    
    # モデルファイル
    model_xml = args.model      # モデルファイル名(xml)
    
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
    
    # 初期状態のsync/asyncモード切替
    is_async_mode = not args.sync
    
    # queue数のチェック
    queue_num = args.queue_num
    if is_async_mode :
        if queue_num < 2 :
            log.warning("queue_num option must be greater than or equal to 2. use default value(2)")
            queue_num = 2
    else :
        if queue_num != 2 :
            log.warning("queue_num option is ignored in sync mode")

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
        print(f'command :          {" ".join(sys.argv)}', file=time_f)
        print(f'openVINO vertion : {ov_vession_str}', file=time_f)
        print(f' frame_number, frame_time, preprocess_time, infer_time, postprocess_time', file=time_f)

    log_f = None
    if args.log :
        log_f = open(args.log, mode='w')
        console_print(log_f, f'command :          {" ".join(sys.argv)}')
        console_print(log_f, f'openVINO vertion : {ov_vession_str}')
    
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
    log.info(f"Loading model file: {model_xml}")
    model = core.read_model(model_xml)      # xmlとbinが同名ならbinは省略可能
    
    # 出力レイヤ数のチェックと名前の取得
    log.info("Check outputs")
    outputs = model.outputs
    assert len(outputs) == 1, "Demo supports only single output topologies" # 出力レイヤ数は1のみ対応
    output_blob = model.outputs[0].get_any_name()       # 出力レイヤ名
    
    # 入力レイヤ数のチェックと名前の取得
    log.info("Check inputs")
    inputs = model.inputs
    
    image_blob_name = None                  # 入力画像レイヤ
    image_info_blob_name = None             # 入力画像情報レイヤ
    
    for blob in inputs:
        blob_name = blob.get_any_name()         # 名前
        blob_shape = blob.shape                 # サイズ
        print(f'{blob_name}   {blob_shape}')
        if len(blob_shape) == 4:                # 4次元なら入力画像レイヤ
            image_blob_name = blob_name
            image_blob_shape = blob_shape
            if blob_shape[1] in range(1, 5) :
                # CHW
                image_blob_format_NCHW = True
                img_input_n, img_input_colors, img_input_height, img_input_width = blob_shape
            else :
                # HWC
                image_blob_format_NCHW = False
                img_input_n, img_input_height, img_input_width, img_input_colors = blob_shape
        elif len(blob_shape) == 2:              # 2次元なら入力画像情報レイヤ
           image_info_blob_name = blob_name
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
    if args.save :
        if all_frames == 1 :
            img_save.set_jpeg(args.save)
        else :
            img_save.create_writer(args.save, org_frame_rate)
    
    # 1フレーム表示後の待ち時間
    wait_key_time = 1
    
    # 動画か静止画かをチェック
    if all_frames == 1:
        # 1フレーム -> 静止画
        is_async_mode = False       # 同期モードにする
        wait_key_time = 0           # 永久待ち
    
    # 初期状態のsync/asyncモードを表示
    if is_async_mode : 
        log.info(f"Starting inference in async mode and queue_num is {queue_num}...")
    else :
        log.info(f"Starting inference in sync mode...")
    
    # モデルのコンパイル
    log.info("Loading model to the plugin...")
    compiled_model = core.compile_model(model, args.device)

    # 推論キューの作成
    if is_async_mode: 
        # 非同期モードならqueue_num面
        async_queue = ov_AsyncInferQueue(compiled_model, queue_num)
    else :
        # 同期モードなら1面
        async_queue = ov_AsyncInferQueue(compiled_model, 1)
    
    # 推論結果格納用辞書
    infer_results = {}
    
    # 前後処理/callbackの設定
    pre_process = InferPreProcess(image_blob_name, image_blob_shape, image_blob_format_NCHW, image_info_blob_name, all_frames, is_async_mode, queue_num)
    post_process = InferPostProcess(labels_map, log_f)
    infer_callback = InferCallback(output_blob, args.prob_threshold, infer_results)

    # callbackの設定
    async_queue.set_callback(infer_callback)
    
    # 推論開始
    log.info("Starting inference...")
    print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")
    
    # 現在のフレーム番号
    infer_frame_number = 1      # 推論用フレーム番号
    disp_frame_number  = 1      # 表示用フレーム番号
    
    
    # キャプチャフラグ
    capture_flag = True
    
    # フレーム測定用タイマ
    prev_time = time.perf_counter()
    
    while True:
        if capture_flag and async_queue.is_ready() :
            # 画像の前処理 =============================================================================
            # 現在のフレーム番号表示
            capture_time = time.perf_counter()
            if infer_frame_number == 1 :
                first_capture_time = capture_time
            capture_time = (capture_time - first_capture_time) * 1000
            print(f'frame_number: {infer_frame_number:5d} / {all_frames}', end='\r', flush=True)
            if log_f :
                console_print(log_f, f'frame_number: {infer_frame_number:5d} / {all_frames}     @{capture_time:10.3f}')
            
            # 画像キャプチャ
            preprocess_start_time = time.perf_counter()                         # 前処理開始時刻        --------------------------------
            ret, img_frame = cap.read()    # フレームのキャプチャ
            if not ret:
                # キャプチャ失敗
                capture_flag = False        # 次からキャプチャしない
                if is_async_mode :
                    # ASYNCモードではキューに残った結果を処理するまでループ継続
                    continue
                else :
                    # SYNCモード時はここで終了
                    break;
            
            # 画像キャプチャと表示/入力用画像を作成
            feed_dict, disp_frame = pre_process(img_frame, infer_frame_number)
            disp_frame.start_preprocess(preprocess_start_time)
            disp_frame.end_preprocess()                                 # 前処理終了時刻        --------------------------------
            
            # 推論予約 =============================================================================
            disp_frame.start_infer()                                    # 推論処理開始時刻      --------------------------------
            parse_params = (disp_frame)     # 必要なら追加
            async_queue.start_async(feed_dict, parse_params)
            
            # フレーム番号更新
            infer_frame_number += 1
        
        # 推論結果待ち =============================================================================
        infer_rst = infer_results.pop(disp_frame_number, None)           # 辞書から要素を取り出して削除、要素がなければNone
        if infer_rst :        # callbackでinfer_resultsに結果が入ったらNone以外が返る
            cur_frame = infer_rst["frame"]
            cur_frame.end_infer()                                       # 推論処理終了時刻      --------------------------------
            
            # 検出結果の解析 =============================================================================
            cur_frame.start_postprocess()                               # 後処理開始時刻            --------------------------------
            post_process(infer_rst)
            cur_frame.end_postprocess()                                 # 後処理終了時刻            --------------------------------
            
            # フレーム処理時間を保存
            cur_time = time.perf_counter()                              # 現在のフレーム処理完了時刻
            frame_time = cur_time - prev_time                   # 1フレームの処理時間
            cur_frame.set_frame_time(frame_time)
            prev_time = cur_time
            
            # 結果の表示 =============================================================================
            # 測定データの表示
            cur_frame.disp_status()
            
            # 処理時間記録
            cur_frame.write_time_data(time_f)
            
            # 画面表示
            if not no_disp :
                cur_frame.disp_image()        # 表示
            
            # 画像の保存
            # 保存が設定されているか否か、MPEGかJPEGかはメソッド内でチェック
            img_save.write_image(cur_frame)
            
            # 表示フレーム更新
            disp_frame_number += 1
            
            # キー入力取得
            key = cv2.waitKey(wait_key_time)
            if key == 27:
                # ESCキー
                break
            
            # 最後のフレームチェック(ASYNCモード時)
            if is_async_mode and (disp_frame_number >= infer_frame_number) :
                # 最後のフレームを表示した
                break;
                # NOTE : SYNCモード時はキャプチャ失敗で終了なのでここではチェックしない
    
    # キュー内の残りのデータが処理されるのを待つ(これをやらないと中断時にプログラムが終了しない)
    async_queue.wait_all()
    
    # ESCキーで中断したときの残りは表示しない
    
    # 後片付け
    if time_f :
        time_f.close()
    
    if log_f :
        log_f.close()
    
    # 保存が設定されていか否かはメソッド内でチェック
    img_save.release_writer()
    
    # 表示ウィンドウを破棄
    cv2.destroyAllWindows()
# ================================================================================

if __name__ == '__main__':
    sys.exit(main() or 0)
