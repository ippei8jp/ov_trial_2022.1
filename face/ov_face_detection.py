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
from model.model_face_detect import model_face_detect
from model.model_face_landmark5 import model_face_landmark5
from model.model_face_landmark35 import model_face_landmark35
from model.model_face_headpose import model_face_headpose
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
    
    fase_args = parser.add_argument_group('face detect Options')
    fase_args.add_argument("-m", "--model", required=True, type=str, 
                        help="Required.\n"
                             "Path to an .xml file with a trained model.")
    fase_args.add_argument("-d", "--device", default="CPU", type=str, 
                        help="Optional\n"
                             "Specify the target device to infer on; \n"
                             "CPU, GPU, FPGA, HDDL or MYRIAD is acceptable.\n"
                             "The demo will look for a suitable plugin \n"
                             "for device specified.\n"
                             "Default value is CPU")
    fase_args.add_argument("-t_detect", "--threshold_detect", default=0.5, type=float, 
                        help="Optional.\n"
                             "Probability threshold for detections filtering")
    
    lm5_args = parser.add_argument_group('landmark detect (5points) Options')
    lm5_args.add_argument("-m_lm5", "--model_lm5", default=None, type=str, 
                        help="Optional.\n"
                             "Path to an .xml file for landmark detection (5points) model.")
    lm5_args.add_argument("-d_lm5", "--device_lm5", default="CPU", type=str, 
                        help="Optional\n"
                             "Specify the target device to infer for landmark detection (5points)\n"
                             "Default value is CPU")
    
    lm35_args = parser.add_argument_group('landmark detect (35points) Options')
    lm35_args.add_argument("-m_lm35", "--model_lm35", default=None, type=str, 
                        help="Optional.\n"
                             "Path to an .xml file for landmark detection (35point) model.")
    lm35_args.add_argument("-d_lm35", "--device_lm35", default="CPU", type=str, 
                        help="Optional\n"
                             "Specify the target device to infer for landmark detection (35point)\n"
                             "Default value is CPU")
    
    hp_args = parser.add_argument_group('head pose estimation Options')
    hp_args.add_argument("-m_hp", "--model_hp", default=None, type=str, 
                        help="Optional.\n"
                             "Path to an .xml file for head pose estimation model.")
    hp_args.add_argument("-d_hp", "--device_hp", default="CPU", type=str, 
                        help="Optional\n"
                             "Specify the target device to infer for head pose estimation\n"
                             "Default value is CPU")
    
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
    ov_vession_str = ov_get_version()
    log.info(f"openVINO vertion : {ov_vession_str}")

    # コマンドラインオプションの解析
    args = build_argparser().parse_args()
    
    # 非表示設定
    no_disp = args.no_disp
    
    # 入力ファイル
    if args.input == 'cam':
        # カメラ入力の場合
        input_file = 0
    else:
        input_file = os.path.abspath(args.input)
        assert os.path.isfile(input_file), "Specified input file doesn't exist"
    
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
    
    # 推論エンジンの初期化
    log.info("Creating Inference Engine...")
    core = ov_Core()
    
    # 拡張ライブラリのロード(CPU使用時のみ)
    if args.cpu_extension and 'CPU' in args.device:
        log.info("Loading Extension Library...")
        core.add_extension(args.cpu_extension)
    
    # キャプチャデバイス
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
        wait_key_time = 0           # 永久待ち
    
    # モデルの作成
    model_fd = model_face_detect(core, args.device, args.model, args.threshold_detect, 1.2, log_f)
    
    model_lm5  = None
    model_lm35 = None
    model_hp   = None
    
    if args.model_lm5 :
        model_lm5 = model_face_landmark5(core, args.device_lm5, args.model_lm5, log_f)
        
    if args.model_lm35 :
        model_lm35 = model_face_landmark35(core, args.device_lm35, args.model_lm35, log_f)
    
    if args.model_hp :
        model_hp = model_face_headpose(core, args.device_hp, args.model_hp, log_f)
    
    # 推論開始
    log.info("Starting inference...")
    print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")
    
    # 現在のフレーム番号
    frame_number = 1
    
    # キャプチャフラグ
    capture_flag = True
    
    # フレーム測定用タイマ
    prev_time = time.perf_counter()
    
    while True:
        # 画像の前処理 =============================================================================
        # 現在のフレーム番号表示
        capture_time = time.perf_counter()
        if frame_number == 1 :
            first_capture_time = capture_time
        capture_time = (capture_time - first_capture_time) * 1000
        print(f'frame_number: {frame_number:5d} / {all_frames}', end='\r', flush=True)
        if log_f :
            console_print(log_f, f'frame_number: {frame_number:5d} / {all_frames}     @{capture_time:10.3f}')
            
        # 画像キャプチャ
        preprocess_start_time = time.perf_counter()                         # 前処理開始時刻        --------------------------------
        ret, image = cap.read()    # フレームのキャプチャ
        if not ret:
            # キャプチャ失敗
            break;
            
        # 表示用フレームの作成
        disp_frame = DispFrame(image, frame_number, all_frames)
        
        # 画像キャプチャと表示/入力用画像を作成
        feed_dict = model_fd.pre_process(image)
        disp_frame.start_preprocess(preprocess_start_time)
        disp_frame.end_preprocess()                                 # 前処理終了時刻        --------------------------------
        
        disp_frame.start_infer()                                    # 推論処理開始時刻      --------------------------------
        results = model_fd.infer_sync(feed_dict, (image, ))
        
        for result in results :
            if model_lm5 :
                feed_dict = model_lm5.pre_process(result["image"])
                lm5_result = model_lm5.infer_sync(feed_dict, None)
                result["landmark5"] = lm5_result
            
            if model_lm35 :
                feed_dict = model_lm35.pre_process(result["image"])
                lm35_result = model_lm35.infer_sync(feed_dict, None)
                result["landmark35"] = lm35_result
        
            if model_hp :
                feed_dict = model_hp.pre_process(result["image"])
                hp_result = model_hp.infer_sync(feed_dict, None)
                result["head_pose"] = hp_result

        disp_frame.end_infer()                                      # 推論処理開始時刻      --------------------------------

        # 検出結果の解析 =============================================================================
        disp_frame.start_postprocess()                              # 後処理開始時刻            --------------------------------
        
        for result in results :
            model_fd.post_process(disp_frame, result)
            
            if model_lm5 :
                model_lm5.post_process(disp_frame, result["landmark5"], result["pt1_ex"], result["pt2_ex"])
            
            if model_lm35 :
                model_lm35.post_process(disp_frame, result["landmark35"], result["pt1_ex"], result["pt2_ex"])
            
            if model_hp :
                model_hp.post_process(disp_frame,  result["head_pose"], result["pt1_ex"], result["pt2_ex"])
        
        disp_frame.end_postprocess()                                # 後処理終了時刻            --------------------------------
        
        # フレーム処理時間を保存
        cur_time = time.perf_counter()                              # 現在のフレーム処理完了時刻
        frame_time = cur_time - prev_time                           # 1フレームの処理時間
        disp_frame.set_frame_time(frame_time)
        prev_time = cur_time
        
        # 結果の表示 =============================================================================
        # 測定データの表示
        disp_frame.disp_status()
        
        # 処理時間記録
        disp_frame.write_time_data(time_f)
        
        # 画面表示
        if not no_disp :
            disp_frame.disp_image()        # 表示
        
        # 画像の保存
        # 保存が設定されているか否か、MPEGかJPEGかはメソッド内でチェック
        img_save.write_image(disp_frame)
        
        # キー入力取得
        key = cv2.waitKey(wait_key_time)
        if key == 27:
            # ESCキー
            break
        
        # フレーム番号更新
        frame_number += 1
        
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
