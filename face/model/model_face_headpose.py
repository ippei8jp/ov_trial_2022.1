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

class model_face_headpose(sync_model_base) :
    def __init__(self, core, model_xml, device="CPU", log_f=None) :
        # 親クラスの初期化をcall
        super().__init__(core, model_xml, device=device, log_f=log_f)
        
    # output blobの確認 ===============================================
    def check_output_blob(self) :
        # 出力レイヤ数のチェックと名前の取得
        log.info("Check outputs")
        outputs = self.model.outputs
        assert len(outputs) == 3, "Demo supports only 3 output topologies" # 出力レイヤ数は3のみ対応

        output_blob_names = [x.get_any_name() for x in outputs]
        assert 'angle_r_fc'  in output_blob_names, "'angle_r_fc' not in outputs"
        assert 'angle_p_fc'  in output_blob_names, "'angle_p_fc' not in outputs"
        assert 'angle_y_fc'  in output_blob_names, "'angle_y_fc' not in outputs"
        
        for x in outputs :
            assert tuple(x.shape) == (1, 1), "each output shape must (1, 1)"
    
    # 結果の解析 ===============================================
    def analyze_result(self, res, params) :
        # params未使用
        
        # output tensorの取り出し
        roll  = res.get_tensor('angle_r_fc').data[0][0]
        pitch = res.get_tensor('angle_p_fc').data[0][0]
        yaw   = res.get_tensor('angle_y_fc').data[0][0]
        
        result = {"roll": roll, "pitch":pitch, "yaw":yaw}
        
        return result
    # ================================================================================
    
    # 後処理 =======================================================
    def post_process(self, cur_frame, result, pt1_ex, pt2_ex) :
        # 顔画像の中心位置
        pt_center = (pt1_ex + pt2_ex) / 2                  
        
        yaw   = result["yaw"]
        pitch = result["pitch"]
        roll  = result["roll"]

        # 結果をログファイルorコンソールに出力
        console_print(self.log_f, f'     yaw={yaw:7.2f} pitch={pitch:7.2f} roll={roll:7.2f}', False)
        
        # XYZ軸の描画
        cur_frame.draw_xyz_axis(result["yaw"], result["pitch"], result["roll"], pt_center)
    
    # ================================================================================
    
    
