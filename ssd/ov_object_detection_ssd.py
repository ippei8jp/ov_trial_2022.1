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
class DisplayFrame() :
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
    # 初期化
    def __init__(self, img_height, img_width, num_frame) :
        # インスタンス変数の初期化
        self.STATUS_LINE_HIGHT    = 15                              # ステータス行の1行あたりの高さ
        self.STATUS_AREA_HIGHT    =  self.STATUS_LINE_HIGHT * 6 + 8 # ステータス領域の高さは6行分と余白
        
        self.img_height = img_height
        self.img_width = img_width
        
        self.writer = None
        
        # 表示用フレームの作成   (2面(current,next)×高さ×幅×色)
        self.disp_height = self.img_height + self.STATUS_AREA_HIGHT                    # 情報表示領域分を追加
        self.disp_frame = np.zeros((num_frame, self.disp_height, img_width, 3), np.uint8)
    
    def STATUS_LINE_Y(self, line) : 
        return self.img_height + self.STATUS_LINE_HIGHT * (line + 1)
    
    def status_puts(self, frame_id, message, line) :
        cv2.putText(self.disp_frame[frame_id], message, (10, self.STATUS_LINE_Y(line)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 128, 128), 1)
    
    # 画像フレーム初期化
    def init_image(self, frame_id, frame) :
        self.disp_frame[frame_id].fill(0)
        self.disp_frame[frame_id, :self.img_height, :self.img_width] = frame
   
    # 画像フレーム表示
    def disp_image(self, frame_id) :
        cv2.imshow("Detection Results", self.disp_frame[frame_id])                  # 表示
    
    # 検出枠の描画
    def draw_box(self, frame_id, str, class_id, left, top, right, bottom) :
        # 対象物の枠とラベルの描画
        color = self.COLOR_PALETTE[class_id & 0x7]       # 表示色(IDの下一桁でカラーパレットを切り替える)
        cv2.rectangle(self.disp_frame[frame_id], (left, top), (right, bottom), color, 2)
        cv2.rectangle(self.disp_frame[frame_id], (left, top+20), (left+160, top), color, -1)
        cv2.putText(self.disp_frame[frame_id], str, (left, top + 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        
        return
    
    # JPEGファイル書き込み
    def save_jpeg(self, jpeg_file, frame_id) :
        if jpeg_file :
            cv2.imwrite(jpeg_file, self.disp_frame[frame_id])
    
    # 動画ファイルのライタ生成
    def create_writer(self, filename, frame_rate) :
        # フォーマット
        fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.writer = cv2.VideoWriter(filename, fmt, frame_rate, (self.img_width, self.disp_height))
    
    # 動画ファイル書き込み
    def write_image(self, frame_id) :
        if self.writer:
            self.writer.write(self.disp_frame[frame_id])

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

# 結果の解析と表示
def parse_result(res, disp_frame, request_id, labels_map, prob_threshold, frame_number, log_f=None) :
    # print(res)
    #  -> 例：(1, 1, 100, 7)        100:バウンディングボックスの数
    #                               7: [image_id, label, conf, x_min, y_min, x_max, y_max]
    # データ構成は
    # https://docs.openvino.ai/2022.1/omz_models_model_ssdlite_mobilenet_v2.html
    # 等の「outputs」の 「Converted Model」を参照

    # バウンディングボックス毎の結果を取得
    res_array = res[0][0]

    for obj in res_array:
        conf = obj[2]                       # confidence for the predicted class(スコア)
        if conf > prob_threshold:           # 閾値より大きいものだけ処理
            class_id = int(obj[1])                         # クラスID
            left     = int(obj[3] * disp_frame.img_width)  # バウンディングボックスの左上のX座標
            top      = int(obj[4] * disp_frame.img_height) # バウンディングボックスの左上のY座標
            right    = int(obj[5] * disp_frame.img_width)  # バウンディングボックスの右下のX座標
            bottom   = int(obj[6] * disp_frame.img_height) # バウンディングボックスの右下のY座標
            
            # 検出結果
            # ラベルが定義されていればラベルを読み出し、なければclass ID
            if labels_map :
                if len(labels_map) > class_id :
                    class_name = labels_map[class_id]
                else :
                    class_name = str(class_id)
            else :
                class_name = str(class_id)
            # 結果をログファイルorコンソールに出力
            console_print(log_f, f'{frame_number:3}:Class={class_name:15}({class_id:3}) Confidence={conf:4f} Location=({int(left)},{int(top)})-({int(right)},{int(bottom)})', False)
            
            # 検出枠の描画
            box_str = f"{class_name} {round(conf * 100, 1)}%"
            disp_frame.draw_box(request_id, box_str, class_id, left, top, right, bottom)

    return
# ================================================================================

# 表示&入力フレームの作成 =======================================================
def prepare_disp_and_input(cap, disp_frame, request_id, input_shape) :
    ret, img_frame = cap.read()    # フレームのキャプチャ
    if not ret :
        # キャプチャ失敗
        return ret, None
    
    # 表示用フレームの作成
    disp_frame.init_image(request_id, img_frame)
    
    # 入力用フレームの作成
    input_n, input_height, input_width, input_colors = input_shape
    in_frame = cv2.resize(img_frame, (input_width, input_height))       # リサイズ
    in_frame = in_frame.reshape(input_shape)                            # HWC → BHWC   ※ 旧バージョンはBCHWだった
    
    return ret, in_frame
# ================================================================================

def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    
    # コマンドラインオプションの解析
    args = build_argparser().parse_args()
    
    model_xml = args.model                                      # モデルファイル名(xml)
    # model_bin = os.path.splitext(model_xml)[0] + ".bin"         # モデルファイル名(bin)
    
    no_disp = args.no_disp
    
    model_label = None
    if args.labels:
        model_label = args.labels
    else:
        model_label = os.path.splitext(model_xml)[0] + ".labels"
    if not os.path.isfile(model_label)  :
        log.warning("label file is not specified")
        model_label = None
    
    labels_map = None
    if model_label:
        # ラベルファイルの読み込み
        with open(model_label, 'r') as f:
            labels_map = [x.strip() for x in f]
    
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
    
    # 推論エンジンの初期化
    log.info("Creating Inference Engine...")
    core = ov_Core()
    
    # 拡張ライブラリのロード(CPU使用時のみ)
    if args.cpu_extension and 'CPU' in args.device:
        log.info("Loading Extension Library...")
        core.add_extension(args.cpu_extension)
    
    # IR(Intermediate Representation ;中間表現)ファイル(.xml & .bin) の読み込み
    log.info(f"Loading model files:\n\t{model_xml}\n\t{model_label}")
    model = core.read_model(model_xml)      # xmlとbinが同名ならbinは省略可能
    
    # 出力レイヤ数のチェックと名前の取得
    log.info("Check outputs")
    outputs = model.outputs
    assert len(outputs) == 1, "Demo supports only single output topologies"
    output_blob = model.outputs[0].get_any_name()       # 出力レイヤ名

    # 入力レイヤ数のチェックと名前の取得
    log.info("Check inputs")
    inputs = model.inputs
    
    img_input_blob_name = None
    img_info_input_blob_name = None
    
    for blob in inputs:
        blob_name = blob.get_any_name()
        blob_shape = blob.shape
        print(f'{blob_name}   {blob_shape}')
        if len(blob_shape) == 4:
            img_input_blob_name = blob_name
            img_input_blob_shape = blob_shape
            input_n, img_input_height, img_input_width, input_colors = blob_shape
        elif len(blob_shape) == 2:
           img_info_input_blob_name = blob_name
        else:
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
    
    # 表示用フレームの作成   (2面(current,next)×高さ×幅×色)
    disp_frame = DisplayFrame(img_height, img_width, 2 if is_async_mode else 1)
    
    # 画像保存オプション
    # writer = None
    jpeg_file = None
    if args.save :
        if all_frames == 1 :
            jpeg_file = args.save
        else :
            disp_frame.create_writer(args.save, org_frame_rate)
    
    # 初期状態のsync/asyncモードを表示
    log.info(f"Starting inference in {'async' if  is_async_mode else 'sync'} mode...")
    
    wait_key_code = 1
    
    # 動画か静止画かをチェック
    if all_frames == 1:
        # 1フレーム -> 静止画
        is_async_mode = False       # 同期モードにする
        wait_key_code = 0           # 永久待ち
    
    cur_request_id = 0          # 同期モードで初期化(ID=0のみ使用)
    next_request_id = 0
    if is_async_mode:
        # 非同期モード
        cur_request_id = 0
        next_request_id = 1
    
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
    frame_number = 1
    
    if is_async_mode:
        # 非同期モード時は最初のフレームの推論を予約しておく
        feed_dict = {}
        if img_info_input_blob_name:
            feed_dict[img_info_input_blob_name] = [img_input_height, img_input_width, 1]
        ret, feed_dict[img_input_blob_name] = prepare_disp_and_input(cap, disp_frame, cur_request_id, img_input_blob_shape)
        if not ret :
            print("failed to capture first frame.")
            sys.exit(1)
        # 推論予約 =============================================================================
        async_queue[cur_request_id].start_async(feed_dict)
    
    # フレーム測定用タイマ
    prev_time = time.time()
    
    while cap.isOpened():           # キャプチャストリームがオープンされてる間ループ
        # 画像の前処理 =============================================================================
        preprocess_start = time.time()                          # 前処理開始時刻            --------------------------------

        # 現在のフレーム番号表示
        print(f'frame_number: {frame_number:5d} / {all_frames}', end='\r')
        # 画像キャプチャと表示/入力用画像を作成
        # 非同期モード時は次のフレームとして
        feed_dict = {}
        if img_info_input_blob_name:
            feed_dict[img_info_input_blob_name] = [img_input_height, img_input_width, 1]
        ret, feed_dict[img_input_blob_name] = prepare_disp_and_input(cap, disp_frame, next_request_id, img_input_blob_shape)
        if not ret:
            # キャプチャ失敗
            break
        
        # 推論予約 =============================================================================
        async_queue[next_request_id].start_async(feed_dict)
        
        preprocess_end = time.time()                            # 前処理終了時刻            --------------------------------
        preprocess_time = preprocess_end - preprocess_start     # 前処理時間
        
        inf_start = time.time()                                 # 推論処理開始時刻          --------------------------------
        
        # 推論結果待ち =============================================================================
        if async_queue[cur_request_id].wait_for(-1) :           # -1: 永久待ち
            inf_end = time.time()                               # 推論処理終了時刻          --------------------------------
            inf_time = inf_end - inf_start                      # 推論処理時間
            
            # 検出結果の解析 =============================================================================
            parse_start = time.time()                           # 解析処理開始時刻          --------------------------------
            res = async_queue[cur_request_id].get_tensor(output_blob).data[:]
            
            parse_result(res, disp_frame, cur_request_id, labels_map, args.prob_threshold, frame_number, log_f)
            
            parse_end = time.time()                             # 解析処理終了時刻          --------------------------------
            parse_time = parse_end - parse_start                # 解析処理開始時間
            
            # 結果の表示 =============================================================================
            render_start = time.time()                          # 表示処理開始時刻          --------------------------------
            # 測定データの表示
            frame_number_message    = f'frame_number   : {frame_number:5d} / {all_frames}'
            if frame_time == 0 :
                frame_time_message  =  'Frame time     : ---'
            else :
                frame_time_message  = f'Frame time     : {(frame_time * 1000):.3f} ms    {(1/frame_time):.2f} fps'  # ここは前のフレームの結果
            render_time_message     = f'Rendering time : {(render_time * 1000):.3f} ms'                             # ここは前のフレームの結果
            inf_time_message        = f'Inference time : {(inf_time * 1000):.3f} ms'
            parsing_time_message    = f'parse time     : {(parse_time * 1000):.3f} ms'
            async_mode_message      = f"Async mode is {' on' if is_async_mode else 'off'}. Processing request {cur_request_id}"
            
            # 結果の書き込み
            disp_frame.status_puts(cur_request_id, frame_number_message, 0)
            disp_frame.status_puts(cur_request_id, inf_time_message,     1)
            disp_frame.status_puts(cur_request_id, parsing_time_message, 2)
            disp_frame.status_puts(cur_request_id, render_time_message,  3)
            disp_frame.status_puts(cur_request_id, frame_time_message,   4)
            disp_frame.status_puts(cur_request_id, async_mode_message,   5)
            # 表示
            if not no_disp :
                disp_frame.disp_image(cur_request_id)        # 表示
            
            # 画像の保存
            if jpeg_file :
                disp_frame.save_jpeg(jpeg_file, cur_request_id)
            # 保存が設定されていか否かはメソッド内でチェック
            disp_frame.write_image(cur_request_id)
            render_end = time.time()                            # 表示処理終了時刻          --------------------------------
            render_time = render_end - render_start             # 表示処理時間
        
        # 非同期モードではフレームバッファ入れ替え
        if is_async_mode:
            cur_request_id, next_request_id = next_request_id, cur_request_id
        
        # タイミング調整 =============================================================================
        wait_start = time.time()                            # タイミング待ち開始時刻    --------------------------------
        key = cv2.waitKey(wait_key_code)
        if key == 27:
            # ESCキー
            break
        wait_end = time.time()                              # タイミング待ち終了時刻    --------------------------------
        wait_time = wait_end - wait_start                   # タイミング待ち時間
        
        # フレーム処理終了 =============================================================================
        cur_time = time.time()                              # 現在のフレーム処理完了時刻
        frame_time = cur_time - prev_time                   # 1フレームの処理時間
        prev_time = cur_time
        if time_f :
            time_f.write(f'{frame_number:5d}, {frame_time * 1000:.3f}, {preprocess_time * 1000:.3f}, {inf_time * 1000:.3f}, {parse_time * 1000:.3f}, {render_time * 1000:.3f}, {wait_key_code}, {wait_time * 1000:.3f}\n')
        frame_number = frame_number + 1
    
    # 後片付け
    if time_f :
        time_f.close()
    
    if log_f :
        log_f.close()
    
    # 保存が設定されていか否かはメソッド内でチェック
    disp_frame.release_writer()
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    sys.exit(main() or 0)
