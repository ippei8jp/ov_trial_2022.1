#!/usr/bin/env python3
import sys
import os
import time
import logging as log
from argparse import ArgumentParser, SUPPRESS, RawTextHelpFormatter
import cv2
# import numpy as np

# openVINOモジュール
from openvino.runtime import get_version        as ov_get_version
from openvino.runtime import Core               as ov_Core
# from openvino.runtime import AsyncInferQueue    as ov_AsyncInferQueue

# 自作モジュール
from model.model_ssd_detect import model_ssd_detect
from DispFrame import DispFrame, ImageSave, console_print

# コマンドラインパーサの構築 =====================================================
def build_argparser():
    parser = ArgumentParser(add_help=False, formatter_class=RawTextHelpFormatter)
    parser.add_argument('-h', '--help', action='help', default=SUPPRESS, 
                        help='Show this help message and exit.')
    parser.add_argument("-i", "--input", required=True, type=str, 
                        help="Required.\n"
                             "Path to a image/video file. \n"
                             "(Specify 'cam' to work with camera)")
    parser.add_argument("-l", "--cpu_extension", type=str, default=None, 
                        help="Optional.\n"
                             "Required for CPU custom layers. \n"
                             "Absolute path to a shared library\n"
                             "with the kernels implementations.")
    
    ssd_args = parser.add_argument_group('SSD Options')
    ssd_args.add_argument("-m", "--model", required=True, type=str, 
                        help="Required.\n"
                             "Path to an .xml file with a trained model.")
    ssd_args.add_argument("--labels", default=None, type=str, 
                        help="Optional.\n"
                             "Labels mapping file\n"
                             "Default is to change the extension of the modelfile\n"
                             "to '.labels'.")
    ssd_args.add_argument("-d", "--device", default="CPU", type=str, 
                        help="Optional\n"
                             "Specify the target device to infer on; \n"
                             "CPU, GPU, FPGA, HDDL or MYRIAD is acceptable.\n"
                             "The demo will look for a suitable plugin \n"
                             "for device specified.\n"
                             "Default value is CPU")
    ssd_args.add_argument("--queue_num", default=2, type=int, 
                        help="Optional.\n"
                             "Number of async infer queues")
    ssd_args.add_argument("-t_detect", "--threshold_detect", default=0.5, type=float, 
                        help="Optional.\n"
                             "Probability threshold for detections filtering")
    
    output_args = parser.add_argument_group('Output Options')
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
    ov_version_str = ov_get_version()
    log.info(f"openVINO vertion : {ov_version_str}")

    # コマンドラインオプションの解析 =================================================
    args = build_argparser().parse_args()
    
    # モデルファイル
    model_xml = args.model      # モデルファイル名(xml)
    
    # ラベルファイル
    model_label = None
    if args.labels:
        model_label = args.labels
    else:
        model_label = os.path.splitext(model_xml)[0] + ".labels"
    if not os.path.isfile(model_label)  :
        log.warning("label file is not specified")
        model_label = None
    
    # queue数のチェック
    queue_num = args.queue_num
    if queue_num < 1 :
        log.warning("queue_num option must be greater than or equal to 1. use default value(2)")
        queue_num = 2
    
    # 非表示設定
    no_disp = args.no_disp
    
    # 入力ファイル
    if args.input == 'cam':
        # カメラ入力の場合
        input_file = 0
    else:
        input_file = os.path.abspath(args.input)
        assert os.path.isfile(input_file), "Specified input file doesn't exist"
    
    # ログファイル類の初期化 ====================================================================
    time_f = None
    if args.time :
        time_f = open(args.time, mode='w')
        print(f'command :          {" ".join(sys.argv)}', file=time_f)
        print(f'openVINO version : {ov_version_str}', file=time_f)
        print(f' frame_number, frame_time, preprocess_time, infer_time, postprocess_time', file=time_f)

    log_f = None
    if args.log :
        log_f = open(args.log, mode='w')
        console_print(log_f, f'command :          {" ".join(sys.argv)}')
        console_print(log_f, f'openVINO vertion : {ov_version_str}')
    
    # 推論エンジンの初期化 =========================================================
    log.info("Creating Inference Engine...")
    core = ov_Core()
    
    # 拡張ライブラリのロード(CPU使用時のみ)
    if args.cpu_extension and 'CPU' in args.device:
        log.info("Loading Extension Library...")
        core.add_extension(args.cpu_extension)
    
    # キャプチャデバイスの初期化 =========================================================
    cap = cv2.VideoCapture(input_file)
    
    # 幅と高さを取得
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # フレームレート(1フレームの時間単位はミリ秒)の取得
    org_frame_rate = int(cap.get(cv2.CAP_PROP_FPS))                 # オリジナルのフレームレート
    org_frame_time = 1.0 / cap.get(cv2.CAP_PROP_FPS)                # オリジナルのフレーム時間
    # フレーム数
    all_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_frames = 1 if all_frames != -1 and all_frames < 0 else all_frames   # -1なら静止画
    
    # 画像保存インスタンスの作成 =========================================================
    img_save = ImageSave(img_height, img_width)
    
    # 画像保存オプション
    if args.save :
        if all_frames == 1 :
            img_save.set_jpeg(args.save)
        else :
            img_save.create_writer(args.save, org_frame_rate)
    
    # 1フレーム表示後の待ち時間 ========================================================================
    wait_key_time = 1
    # 動画か静止画かをチェック
    if all_frames == 1:
        # 1フレーム -> 静止画
        wait_key_time = 0           # 永久待ち
    
    # モデルの作成 =====================================================================================
    model_ssd = model_ssd_detect(core, model_xml, model_label, args.device, args.threshold_detect, queue_num, log_f)
    
    # 推論開始 =========================================================================================
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
        if capture_flag and model_ssd.is_ready() :
            # 画像の前処理 =============================================================================
            # 現在のフレーム番号表示
            capture_time = time.perf_counter()
            if infer_frame_number == 1 :
                first_capture_time = capture_time
            capture_time = (capture_time - first_capture_time) * 1000       # 最初のキャプチャからの経過時間
            print(f'frame_number: {infer_frame_number:5d} / {all_frames}', end='\r', flush=True)
            if log_f :
                console_print(log_f, f'frame_number: {infer_frame_number:5d} / {all_frames}     @{capture_time:10.3f}')
                
            # 画像キャプチャ
            preprocess_start_time = time.perf_counter()                         # 前処理開始時刻        --------------------------------
            ret, image = cap.read()    # フレームのキャプチャ
            if not ret:
                # キャプチャ失敗
                capture_flag = False        # 次からキャプチャしない
                # ASYNCモードではキューに残った結果を処理するまでループ継続
                continue
            
            # 表示用フレームの作成
            disp_frame = DispFrame(image, infer_frame_number, all_frames)
            
            # 画像キャプチャと表示/入力用画像を作成
            feed_dict = model_ssd.pre_process(image)
            disp_frame.start_preprocess(preprocess_start_time)
            disp_frame.end_preprocess()                                 # 前処理終了時刻        --------------------------------
        
            disp_frame.start_infer()                                    # 推論処理開始時刻      --------------------------------
            model_ssd.start_infer(feed_dict, (disp_frame, ))
            
            # フレーム番号更新
            infer_frame_number += 1
        
        # 推論結果待ち =============================================================================
        infer_rst = model_ssd.get_infer_result(disp_frame_number)       # まだ結果が出てなければNoneが返る
        if infer_rst :
            cur_frame = infer_rst["disp_frame"]
            results   = infer_rst["result"]
            cur_frame.end_infer()                                       # 推論処理終了時刻      --------------------------------
            
            # 検出結果の解析 =============================================================================
            cur_frame.start_postprocess()                               # 後処理開始時刻            --------------------------------
            for rst in results :
                model_ssd.post_process(cur_frame, rst)
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
        
            # 最後のフレームチェック
            if not capture_flag and (disp_frame_number >= infer_frame_number) :
                # 最後のフレームを表示した
                break;
        
        # キー入力取得
        key = cv2.waitKey(wait_key_time)
        if key == 27:
            # ESCキー
            break
        
    # キュー内の残りのデータが処理されるのを待つ(これをやらないと中断時にプログラムが終了しない)
    model_ssd.wait_all()
    
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
