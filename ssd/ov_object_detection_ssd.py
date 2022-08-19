#!/usr/bin/env python
import sys
import os
import time
import logging as log
from argparse import ArgumentParser, SUPPRESS, RawTextHelpFormatter
import cv2
import numpy as np

# openvino.inference_engine のバージョン取得
# from openvino.inference_engine import get_version as ov_get_version
# ov_vession_str = ov_get_version()
# print(ov_vession_str)               # バージョン2019には '2.1.custom_releases/2019/R～'という文字列が入っている
                                    # バージョン2020には '～-releases/2020/～'という文字列が入っている
                                    # バージョン2021には '～-releases/2021/～'という文字列が入っている
                                    # バージョン2022には '～-releases/2022/～'という文字列が入っている

from openvino.inference_engine import IENetwork, IECore


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
    def __init__(self, img_height, img_width) :
        # インスタンス変数の初期化
        self.STATUS_LINE_HIGHT    = 15                              # ステータス行の1行あたりの高さ
        self.STATUS_AREA_HIGHT    =  self.STATUS_LINE_HIGHT * 6 + 8 # ステータス領域の高さは6行分と余白
        
        self.img_height = img_height
        self.img_width = img_width
        
        self.writer = None
        
        # 表示用フレームの作成   (2面(current,next)×高さ×幅×色)
        self.disp_height = self.img_height + self.STATUS_AREA_HIGHT                    # 情報表示領域分を追加
        self.disp_frame = np.zeros((2, self.disp_height, img_width, 3), np.uint8)
    
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
def parse_result(net, res, disp_frame, request_id, labels_map, prob_threshold, frame_number, log_f=None) :
    out_blob = list(net.outputs.keys())[0]
    # print(res[out_blob].shape)
    #  -> 例：(1, 1, 200, 7)        200:バウンディングボックスの数
    # データ構成は
    # https://docs.openvinotoolkit.org/2019_R1/_pedestrian_and_vehicle_detector_adas_0001_description_pedestrian_and_vehicle_detector_adas_0001.html
    # の「outputs」を参照
    
    res_array = res[out_blob].buffer[0][0]

    for obj in res_array:
        conf = obj[2]                       # confidence for the predicted class(スコア)
        if conf > prob_threshold:           # 閾値より大きいものだけ処理
            class_id = int(obj[1])                          # クラスID
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
    input_n, input_colors, input_height, input_width = input_shape
    in_frame = cv2.resize(img_frame, (input_width, input_height))       # リサイズ
    in_frame = in_frame.transpose((2, 0, 1))                            # HWC →  CHW
    in_frame = in_frame.reshape(input_shape)                            # CHW → BCHW
    
    return ret, in_frame
# ================================================================================

def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    
    # コマンドラインオプションの解析
    args = build_argparser().parse_args()
    
    model_xml = args.model                                      # モデルファイル名(xml)
    model_bin = os.path.splitext(model_xml)[0] + ".bin"         # モデルファイル名(bin)
    
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
    
    # 指定されたデバイスの plugin の初期化
    log.info("Creating Inference Engine...")
    ie = IECore()
    # 拡張ライブラリのロード(CPU使用時のみ)
    if args.cpu_extension and 'CPU' in args.device:
        log.info("Loading Extension Library...")
        ie.add_extension(args.cpu_extension, "CPU")
    
    # IR(Intermediate Representation ;中間表現)ファイル(.xml & .bin) の読み込み
    log.info(f"Loading model files:\n\t{model_xml}\n\t{model_bin}\n\t{model_label}")
    net = ie.read_network(model=model_xml, weights=model_bin)
    
    """
    # 未サポートレイヤの確認
    if "CPU" in args.device:
        # サポートしているレイヤの一覧
        supported_layers = ie.query_network(net, "CPU")
        ### # netで使用されているレイヤでサポートしているレイヤの一覧にないもの
        ### not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        # netで使用されているレイヤ一覧
        if "ngraph" in sys.modules :            # ngraphがインポート済みかで判定
            # バージョン 2021.x以降
            used_layers = [l.friendly_name for l in ngraph.function_from_cnn(net).get_ordered_ops()]
        else :
            # バージョン 2020.x以前
            used_layers = list(net.layers.keys())
        # netで使用されているレイヤでサポートしているレイヤの一覧にないもの
        not_supported_layers = [l for l in used_layers if l not in supported_layers]
        # サポートされていないレイヤがある？
        if len(not_supported_layers) != 0:
            # エラー終了
            log.error(f"Following layers are not supported by the plugin for specified device {args.device}:\n {', '.join(not_supported_layers)}")
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)
    """
    
    # このプログラムは1出力のモデルのみサポートしているので、チェック
    # print(net.outputs)
    assert len(net.outputs) == 1, "Demo supports only single output topologies"

    # 入力の準備
    log.info("Preparing inputs")
    # print(net.inputs)
    # SSDのinputsは1とは限らないのでスキャンする
    img_info_input_blob = None
    inputs = net.input_info

    for blob_name in inputs:

        input_shape = inputs[blob_name].input_data.shape
        # print(f'{blob_name}   {input_shape}')
        if len(input_shape) == 4:
            input_blob = blob_name
        elif len(input_shape) == 2:
           # こういう入力レイヤがあるものがある？
           img_info_input_blob = blob_name
        else:
            raise RuntimeError(f"Unsupported {len(input_shape)} input layer '{ blob_name}'. Only 2D and 4D input layers are supported")
    
    # 入力画像情報の取得
    input_n, input_colors, input_height, input_width = inputs[input_blob].input_data.shape

    feed_dict = {}
    # こういう入力レイヤがあるものがある？
    if img_info_input_blob:
        feed_dict[img_info_input_blob] = [input_height, input_width, 1]
    
    # キャプチャ
    cap = cv2.VideoCapture(input_stream)
    
    # 幅と高さを取得
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # フレームレート(1フレームの時間単位はミリ秒)の取得
    org_frame_rate = int(cap.get(cv2.CAP_PROP_FPS))                 # オリジナルのフレームレート
    org_frame_time = 1.0 / cap.get(cv2.CAP_PROP_FPS)                # オリジナルのフレーム時間
    # フレーム数
    all_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_frames = 1 if all_frames != -1 and all_frames < 0 else all_frames
    
    # 表示用フレームの作成   (2面(current,next)×高さ×幅×色)
    disp_frame = DisplayFrame(img_height, img_width)
    
    # 画像保存オプション
    # writer = None
    jpeg_file = None
    if args.save :
        if all_frames == 1 :
            jpeg_file = args.save
        else :
            disp_frame.create_writer(args.save, org_frame_rate)
    
    cur_request_id = 0
    next_request_id = 1
    
    # 初期状態のsync/asyncモードを表示
    log.info(f"Starting inference in {'async' if  is_async_mode else 'sync'} mode...")

    
    wait_key_code = 1
    
    # 動画か静止画かをチェック
    if all_frames == 1:
        # 1フレーム -> 静止画
        is_async_mode = False       # 同期モードにする
        wait_key_code = 0           # 永久待ち
    
    if is_async_mode:
        cur_request_id = 0
        next_request_id = 1
    else :
        cur_request_id = 0          # 同期モードではID=0のみ使用
        next_request_id = 0
    
    # プラグインへモデルをロード
    log.info("Loading model to the plugin...")
    exec_net = ie.load_network(network=net, num_requests=2, device_name=args.device)
    
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
        ret, feed_dict[input_blob] = prepare_disp_and_input(cap, disp_frame, cur_request_id, (input_n, input_colors, input_height, input_width))
        if not ret :
            print("failed to capture first frame.")
            sys.exit(1)
        # 推論予約 =============================================================================
        exec_net.start_async(request_id=cur_request_id, inputs=feed_dict)
    
    # フレーム測定用タイマ
    prev_time = time.time()
    
    while cap.isOpened():           # キャプチャストリームがオープンされてる間ループ
        # 画像の前処理 =============================================================================
        preprocess_start = time.time()                          # 前処理開始時刻            --------------------------------

        # 現在のフレーム番号表示
        print(f'frame_number: {frame_number:5d} / {all_frames}', end='\r')
        # 画像キャプチャと表示/入力用画像を作成
        # 非同期モード時は次のフレームとして
        ret, feed_dict[input_blob] = prepare_disp_and_input(cap, disp_frame, next_request_id, (input_n, input_colors, input_height, input_width))
        if not ret:
            # キャプチャ失敗
            break
        
        # 推論予約 =============================================================================
        exec_net.start_async(request_id=next_request_id, inputs=feed_dict)
        preprocess_end = time.time()                            # 前処理終了時刻            --------------------------------
        preprocess_time = preprocess_end - preprocess_start     # 前処理時間
        
        inf_start = time.time()                                 # 推論処理開始時刻          --------------------------------
        # 推論結果待ち =============================================================================
        if exec_net.requests[cur_request_id].wait(-1) == 0:
            inf_end = time.time()                               # 推論処理終了時刻          --------------------------------
            inf_time = inf_end - inf_start                      # 推論処理時間
            
            # 検出結果の解析 =============================================================================
            parse_start = time.time()                           # 解析処理開始時刻          --------------------------------
            res = exec_net.requests[cur_request_id].output_blobs
            
            parse_result(net, res, disp_frame, cur_request_id, labels_map, args.prob_threshold, frame_number, log_f)
            
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
