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

class sync_model_base() :
    def load_model(self, core, model_xml) :
        # IR(Intermediate Representation ;中間表現)ファイル(.xml & .bin) の読み込み
        log.info(f"Loading model file: {model_xml}")
        self.model = core.read_model(model_xml)      # xmlとbinが同名ならbinは省略可能
    
    def make_infer_queue(self, core, device) :
        # モデルのコンパイル
        log.info("Loading model to the plugin...")
        self.compiled_model = core.compile_model(self.model, device)
        
        # 推論キューの作成
        # 同期モードなら1面
        self.async_queue = ov_AsyncInferQueue(self.compiled_model, 1)
    
    def check_input_blob(self) :
        # 入力レイヤ数のチェックと名前の取得
        log.info("Check inputs")
        inputs = self.model.inputs
        assert len(inputs) == 1, "Demo supports only single input topologies" # 入力レイヤ数は1のみ対応
        self.input_blob_name = inputs[0].get_any_name()       # 入力レイヤ名
        self.input_blob_shape = inputs[0].shape
        assert len(self.input_blob_shape) == 4, "input shape error" # 入力レイヤ数は1のみ対応
        
        if self.input_blob_shape[1] in range(1, 5) :
            # CHW
            self.input_blob_format_NCHW = True
            self.img_input_n, self.img_input_colors, self.img_input_height, self.img_input_width = self.input_blob_shape
        else :
            # HWC
            self.input_blob_format_NCHW = False
            self.img_input_n, self.img_input_height, self.img_input_width, self.img_input_colors = self.input_blob_shape
    
    def is_ready(self) :
        return self.async_queue.is_ready()
    
    def wait_all(self) :
        self.async_queue.wait_all()
    
    def infer_sync(self, feed_dict, params):
        infer_request = self.async_queue[self.async_queue.get_idle_request_id()]
        infer_request.infer(feed_dict)
        return self.analyze_result(infer_request, params)
    
    # 前処理 =======================================================
    def pre_process(self, image) :
        # 入力用フレームの作成
        in_frame = cv2.resize(image, (self.img_input_width, self.img_input_height))     # リサイズ
        if self.input_blob_format_NCHW :
             in_frame = in_frame.transpose((2, 0, 1))                                       # HWC → CHW
        in_frame = in_frame.reshape(self.input_blob_shape)                                  # HWC → BHWC or CHW → BCHW
        
        feed_dict = {self.input_blob_name: in_frame}
        
        return feed_dict
    # ================================================================================
    
