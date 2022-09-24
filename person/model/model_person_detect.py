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
# from openvino.runtime import AsyncInferQueue    as ov_AsyncInferQueue

from .sync_model_base import sync_model_base
from DispFrame import console_print

class model_person_detect(sync_model_base) :
    def __init__(self, core, model_xml, device="CPU", threshold=0.5, clip_ratio=1.2, log_f=None) :
        # 親クラスの初期化をcall
        super().__init__(core, model_xml, device, threshold, log_f)

        self.clip_ratio     = clip_ratio

    # output blobの確認 ===============================================
    def check_output_blob(self) :
        # 出力レイヤ数のチェックと名前の取得
        log.info("Check outputs")
        outputs = self.model.outputs
        self.output_type = 0
        
        output_names = [output.get_any_name() for output in outputs]
        target_blob_name = 'boxes'
        if target_blob_name in output_names :
            # outputが複数で'boxes'が含まれる
            self.output_type = 2
            self.output_blob_name = target_blob_name                # 出力レイヤ名
        elif len(outputs) == 1 :
            output_blob = outputs[0]
            if tuple(output_blob.shape)[-1] == 7 :
                # outputが1個でshapeが(X,X,X,7)
                self.output_type = 1
                self.output_blob_name = output_blob.get_any_name()      # 出力レイヤ名
            else :
                raise RuntimeError(f'output type unknown : outputs[0].shape={outputs[0].shape}')
        else :
            raise RuntimeError(f'output type unknown : output names={output_names}')
        
    
    # 結果の解析 ===============================================
    def analyze_result(self, res, params) :
        # paramsをバラす
        image = params[0]
        img_width  = image.shape[1]
        img_height = image.shape[0]
        
        # output tensorの取り出し
        res = res.get_tensor(self.output_blob_name).data[:]
        
        
        # バウンディングボックス毎の結果を取得
        if self.output_type == 1 :
            res_array = res.reshape(-1,7)
        elif self.output_type == 2 :
            res_array = res.reshape(-1,5)
        else :
            raise RuntimeError('output type unknown')
        
        results = []
        for obj in res_array:
            img_size = np.array((img_width,  img_height))
            if self.output_type == 1 :
                conf = obj[2]                       # confidence for the predicted class(スコア)
                pt1 = np.array((obj[3], obj[4])) * img_size    # (left,  top   )
                pt2 = np.array((obj[5], obj[6])) * img_size    # (right, bottom)
            elif self.output_type == 2 :
                input_size = np.array((self.img_input_width, self.img_input_height))
                conf = obj[4]                       # confidence for the predicted class(スコア)
                pt1 = np.array((obj[0], obj[1])) / input_size * img_size    # (left,  top   )
                pt2 = np.array((obj[2], obj[3])) / input_size * img_size    # (right, bottom)
            else :
                raise RuntimeError('output type unknown')
                
            if conf > self.threshold :              # 閾値より大きいものだけ処理
                pt1 = pt1.astype(int)               # 整数化
                pt2 = pt2.astype(int)               
                size = pt2 - pt1                    # 範囲のサイズ
                
                # 検出範囲の拡張比率で切り取り範囲を決定
                pt1_ex = (pt1 - (size / 2) * (self.clip_ratio - 1)).astype(int)        # (x1, y1)
                pt2_ex = (pt2 + (size / 2) * (self.clip_ratio - 1)).astype(int)        # (x2, y2)
                
                # 元画像範囲に収まるように調整
                pt1_ex = np.clip(pt1_ex, (0, 0), (img_width, img_height))
                pt2_ex = np.clip(pt2_ex, (0, 0), (img_width, img_height))
                
                # 顔画像を切り取り
                cliped_image  = np.array(image[pt1_ex[1]:pt2_ex[1], pt1_ex[0]:pt2_ex[0], :])
                
                """
                cv2.imshow("hoge", image)
                cv2.waitKey(0)
                """
                results.append({"conf":conf, "pt1":pt1, "pt2":pt2, "pt1_ex":pt1_ex, "pt2_ex":pt2_ex, "image":cliped_image})
        return results
    # ================================================================================
    
    # 後処理 =======================================================
    def post_process(self, disp_frame, result) :
        # 結果を個別の変数にバラす
        conf     = result["conf"]
        pt1      = result["pt1"]
        pt2      = result["pt2"]
        
        reid     = None
        color    = None
        text     = None
        if 'reid' in result.keys() :
            reid  = result["reid"]["result"]
            color = disp_frame.get_IndexedColor(reid)
            text  = str(reid)
        
        mark     = None
        if 'attr' in result.keys() :
            is_male = result["attr"]["result"]["is_male"]
            mark = "spade" if is_male else "heart"
        
        # 結果をログファイルorコンソールに出力
        console_print(self.log_f, f'{disp_frame.frame_number:3}: Confidence={conf:4f} Location=({pt1[0]},{pt1[1]})-({pt2[0]},{pt2[1]})', False)
        
        # 検出枠の描画
        disp_frame.draw_box(pt1, pt2, color=color, text=text, mark=mark)
        
        """
        # 拡張した検出枠の描画
        pt1_ex   = result["pt1_ex"]
        pt2_ex   = result["pt2_ex"]
        
        disp_frame.draw_box(pt1_ex, pt2_ex, (128, 255, 255))
        """
    # ================================================================================
    
    
