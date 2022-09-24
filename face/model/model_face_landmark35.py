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

class model_face_landmark35(sync_model_base) :
    def __init__(self, core, model_xml, device="CPU", log_f=None) :
        # 親クラスの初期化をcall
        super().__init__(core, model_xml, device=device, log_f=log_f)
        
    # output blobの確認 ===============================================
    def check_output_blob(self) :
        # 出力レイヤ数のチェックと名前の取得
        log.info("Check outputs")
        outputs = self.model.outputs
        assert len(outputs) == 1, "Demo supports only single output topologies" # 出力レイヤ数は1のみ対応
        self.output_blob_name = outputs[0].get_any_name()       # 出力レイヤ名
        self.output_blob_shape = outputs[0].shape
        assert tuple(self.output_blob_shape) == (1, 35 * 2), f"output shape must be (1, 35 * 2), but it is {self.output_blob_shape}" # 出力レイヤのshape確認
    
    # 結果の解析 ===============================================
    def analyze_result(self, res, params) :
        # パラメータをバラす
        # pt1_ex = params[0]
        # pt2_ex = params[1]

        # output tensorの取り出し
        res = res.get_tensor(self.output_blob_name).data[:]
        
        # 結果の取り出し
        res_array = res[0].reshape((-1, 2))
        
        # size_ex = pt2_ex - pt1_ex
        # result = (res_array * size_ex + pt1_ex).astype(int)
        
        return res_array
    # ================================================================================
    
    # 後処理 =======================================================
    def post_process(self, cur_frame, result, pt1_ex, pt2_ex) :
        size_ex = pt2_ex - pt1_ex
        
        console_print(self.log_f, '     ', end='')
        for i, point in enumerate(result) :
            # 特徴点の描画
            point_img = (point * size_ex + pt1_ex).astype(int)  # 画像の座標に変換
            console_print(self.log_f, f'P{i}=({point_img[0]:3d}, {point_img[1]:3d}) ', end='')
            cur_frame.draw_point(point_img, color=(0, 0, 255), text=str(i))
        console_print(self.log_f, '', end='\n')

    # ================================================================================
    
