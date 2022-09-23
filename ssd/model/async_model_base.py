#!/usr/bin/env python3
import sys
import os
import time
import logging as log
import cv2
import numpy as np

# openVINOモジュール
# from openvino.runtime import get_version        as ov_get_version
# from openvino.runtime import Core               as ov_Core
from openvino.runtime import AsyncInferQueue    as ov_AsyncInferQueue

class async_model_base() :
    def __init__(self, core, model_xml, device="CPU", prob_threshold=0.5, queue_num=2, log_f=None) :
        self.core = core
        self.model_xml = model_xml
        self.device = device
        self.prob_threshold = prob_threshold
        self.queue_num      = queue_num
        self.log_f          = log_f
        
        # 結果格納辞書
        self.infer_results = {}
        
        # IR(Intermediate Representation ;中間表現)ファイル(.xml & .bin) の読み込み
        self.load_model()
        
        # 入力レイヤ数のチェックと名前の取得
        self.check_input_blob()
        
        # 出力レイヤ数のチェックと名前の取得
        self.check_output_blob()
        
        # モデルのコンパイル&推論キュー作成
        self.make_infer_queue()
        
    def load_model(self) :
        # IR(Intermediate Representation ;中間表現)ファイル(.xml & .bin) の読み込み
        log.info(f"Loading model file: {self.model_xml}")
        self.model = self.core.read_model(self.model_xml)      # xmlとbinが同名ならbinは省略可能
    
    def check_input_blob(self) :
        # 入力レイヤ数のチェックと名前の取得
        log.info("Check inputs")
        inputs = self.model.inputs
        
        self.img_input_blob_name  = None
        self.img_info_blob_name_3 = None
        self.img_info_blob_name_6 = None
        for blob in inputs:
            blob_name = blob.get_any_name()         # 名前
            blob_shape = blob.shape                 # サイズ
            print(f'{blob_name}   {blob_shape}')
            if len(blob_shape) == 4:                # 4次元なら入力画像レイヤ
                self.img_input_blob_name  = blob_name
                self.img_input_blob_shape = blob_shape
                if blob_shape[1] in range(1, 5) :
                    # CHW
                    self.img_input_blob_format_NCHW = True
                    self.img_input_n, self.img_input_colors, self.img_input_height, self.img_input_width = blob_shape
                else :
                    # HWC
                    self.img_input_blob_format_NCHW = False
                    self.img_input_n, self.img_input_height, self.img_input_width, self.img_input_colors = blob_shape
            elif len(blob_shape) == 2:              # 2次元なら入力画像情報レイヤ
                if blob_shape[1] == 3 :             # 1x3のタイプ
                   self.img_info_blob_name_3 = blob_name
                elif blob_shape[1] == 6 :           # 1x6のタイプ
                   self.img_info_blob_name_6 = blob_name
            else:                                   # それ以外のレイヤが含まれていたらエラー
                raise RuntimeError(f"Unsupported {len(blob_shape)} input layer '{blob_name}'. Only 2D and 4D input layers are supported.")
    
        if self.img_input_blob_name is None :
            raise RuntimeError("Image input blob not found.")
    
    def make_infer_queue(self) :
        # モデルのコンパイル
        log.info("Loading model to the plugin...")
        self.compiled_model = self.core.compile_model(self.model, self.device)
        
        # 推論キューの作成
        self.async_queue = ov_AsyncInferQueue(self.compiled_model, self.queue_num)
        
        # callbackの設定
        self.async_queue.set_callback(self.callback)
    
    def callback(self, res, params) :
        disp_frame, results = self.analyze_result(res, params)
        self.infer_results[disp_frame.frame_number] = {"disp_frame": disp_frame, "result":results}
    
    def is_ready(self) :
        return self.async_queue.is_ready()
    
    def wait_all(self) :
        self.async_queue.wait_all()
    
    def start_infer(self, feed_dict, params):
        self.async_queue.start_async(feed_dict, params)
    
    def get_infer_result(self, disp_frame_number) :
        infer_rst = self.infer_results.pop(disp_frame_number, None)     # 辞書から要素を取り出して削除、要素がなければNone
        return infer_rst
    
    # 前処理 =======================================================
    def pre_process(self, image) :
        # 入力用フレームの作成
        in_frame = cv2.resize(image, (self.img_input_width, self.img_input_height))     # リサイズ
        if self.img_input_blob_format_NCHW :
             in_frame = in_frame.transpose((2, 0, 1))                                       # HWC → CHW
        in_frame = in_frame.reshape(self.img_input_blob_shape)                              # HWC → BHWC or CHW → BCHW
        
        feed_dict = {self.img_input_blob_name: in_frame}
        if not self.img_info_blob_name_3 is None :      # 1x3のタイプ
            feed_dict[self.img_info_blob_name_3] = np.array([[self.img_input_height, self.img_input_width, 1]])
        if not self.img_info_blob_name_6 is None :      # 1x6のタイプ
            feed_dict[self.img_info_blob_name_6] = np.array([[  self.img_input_height, 
                                                                self.img_input_width, 
                                                                self.img_input_width  / image.shape[1], 
                                                                self.img_input_height / image.shape[0], 
                                                                self.img_input_width  / image.shape[1], 
                                                                self.img_input_height / image.shape[0]
                                                            ]])
        
        return feed_dict
    # ================================================================================
    


