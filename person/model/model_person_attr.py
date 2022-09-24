#!/usr/bin/env python3
import sys
import os
import time
import logging as log
# import cv2
import numpy as np

# openVINOモジュール
# from openvino.runtime import get_version        as ov_get_version
# from openvino.runtime import Core               as ov_Core
# from openvino.runtime import AsyncInferQueue    as ov_AsyncInferQueue

from .sync_model_base import sync_model_base
from DispFrame import console_print

# Person reidentification (人物同定) model
class model_person_attr(sync_model_base) :
    # 属性名
    AttrNames = {
            "ATTR_8" : [ 
                    "is_male", 
                    "has_bag", 
                    "has_backpack", 
                    "has_hat", 
                    "has_longsleeves", 
                    "has_longpants", 
                    "has_longhair", 
                    "has_coat_jacket", 
                ], 
            "ATTR_7" : [
                    "is_male", 
                    "has_bag", 
                    "has_hat", 
                    "has_longsleeves", 
                    "has_longpants", 
                    "has_longhair", 
                    "has_coat_jacket", 
                ]
        }

    def __init__(self, core, model_xml, device="CPU", threshold=0.5, log_f=None) :
        # 親クラスの初期化をcall
        super().__init__(core, model_xml, device, threshold, log_f)
        
    # output blobの確認 ===============================================
    def check_output_blob(self) :
        # 出力レイヤ数のチェックと名前の取得
        log.info("Check outputs")
        outputs = self.model.outputs
        if len(outputs) == 3 :
            for x in outputs :
                output_blob_name  = x.get_any_name()
                output_blob_shape = x.shape
                if output_blob_name == '453' :
                    self.output_blob_name  = output_blob_name               # 出力レイヤ名
                    self.output_blob_shape = output_blob_shape
                    self.output_blob_type = "ATTR_8"
                    assert tuple(self.output_blob_shape) == (1, 8, 1, 1), f"output shape must be (1, 1, 8, 1), but it is {self.output_blob_shape}" # 出力レイヤのshape確認
                    break
            else :
                # 453ノードが見つからなかった
                assert '453'  in output_blob_names, "'453' not in outputs"
        elif len(outputs) == 1 :
            self.output_blob_name = outputs[0].get_any_name()       # 出力レイヤ名
            self.output_blob_shape = outputs[0].shape
            self.output_blob_type = "ATTR_7"
            assert tuple(self.output_blob_shape) == (1, 7), f"output shape must be (1, 7), but it is {self.output_blob_shape}" # 出力レイヤのshape確認
        else :
            raise RuntimeError(f"Unsupported {len(outputs)} output layers '.")
    
    # 結果の解析 ===============================================
    def analyze_result(self, res, params) :
        # params未使用
        
        # output tensorの取り出し
        indicators = res.get_tensor(self.output_blob_name).data.flatten()
        
        result = {"result": {}, "raw_result":{}}
        for i, indicator in enumerate(indicators) :
            result["raw_result"][self.AttrNames[self.output_blob_type][i]] = indicator
            if indicator > self.threshold :               # 閾値以上
                result["result"][self.AttrNames[self.output_blob_type][i]] = True
            else :
                result["result"][self.AttrNames[self.output_blob_type][i]] = False
        
        return result
    # ================================================================================
    
    # 後処理 =======================================================
    def post_process(self, cur_frame, result, pt1_ex, pt2_ex) :
        indicators      = result["result"]
        raw_indicators  = result["raw_result"]

        # 結果をログファイルorコンソールに出力
        console_print(self.log_f, f'     ATTR=', end='')
        for key, value in indicators.items() :
            if value :
                console_print(self.log_f, f'{key}, ', end='')
        
        # 改行
        console_print(self.log_f, '')
        
        """
        console_print(self.log_f, f'     ATTR_RAW=', end='')
        for key, value in raw_indicators.items() :
            console_print(self.log_f, f'{key}={value}, ', end='')
        
        # 改行
        console_print(self.log_f, '')
        """
    
    # ================================================================================
    
    
