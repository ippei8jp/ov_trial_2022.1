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

from .async_model_base import async_model_base
from DispFrame import console_print

class model_ssd_detect(async_model_base) :
    def __init__(self, core, model_xml, model_label=None, device="CPU", prob_threshold=0.5, queue_num=2, log_f=None) :
        # 親クラスの初期化をcall
        super().__init__(core, model_xml, device, prob_threshold, queue_num, log_f)
        
        # ラベルファイル読み込み
        self.labels_map = None
        if model_label:
            log.info(f"Loading label file: {model_label}")
            # ラベルファイルの読み込み
            with open(model_label, 'r') as f:
                self.labels_map = [x.strip() for x in f]
    
    # output blobの確認 ===============================================
    def check_output_blob(self) :
        # 出力レイヤ数のチェックと名前の取得
        log.info("Check outputs")
        outputs = self.model.outputs
        
        self.output_type = 0
        if len(outputs) == 1 :
            output_blob = self.model.outputs[0]
            if tuple(output_blob.shape)[-1] == 7 :
                # outputが1個でshapeが(X,X,X,7)
                self.output_type = 1
                self.output_blob_name = output_blob.get_any_name()      # 出力レイヤ名
            else :
                raise ValueError(f'output type unknown : outputs[0].shape={outputs[0].shape}')
        else :
            output_names = [output.get_any_name() for output in outputs]
            box_blob_name = 'boxes'
            label_blob_name = 'labels'
            if box_blob_name in output_names and  label_blob_name in output_names :
                # outputが複数で'boxes'と'labels'が含まれる
                self.output_type = 2
                self.output_blob_name = box_blob_name                # 出力レイヤ名
                self.label_blob_name = label_blob_name                # 出力レイヤ名
            else :
                raise ValueError(f'output type unknown : output names={output_names}')
        
    # 結果の解析 ===============================================
    def analyze_result(self, res, params) :
        # paramsをバラす
        disp_frame = params[0]
        img_width  = disp_frame.img_width
        img_height = disp_frame.img_height
        
        results = []
        img_size = np.array((img_width,  img_height))
        
        if self.output_type == 1 :
            # output tensorの取り出し
            res_array = res.get_tensor(self.output_blob_name).data[:]
            # バウンディングボックス毎の結果を取得
            res_array = res_array.reshape(-1,7)
            for obj in res_array:
                conf     = obj[2]           # confidence for the predicted class(スコア)
                if conf > self.prob_threshold:          # 閾値より大きいものだけ処理
                    class_id = int(obj[1])      # クラスID0
                    pt1 = (np.array((obj[3], obj[4])) * img_size).astype(int)  # (left,  top   )
                    pt2 = (np.array((obj[5], obj[6])) * img_size).astype(int)  # (right, bottom)
                
                    results.append({"conf":conf, "class_id": class_id, "pt1":pt1, "pt2":pt2})
        elif self.output_type == 2 :
            # 入力サイズ
            input_size = np.array((self.img_input_width, self.img_input_height))
            
            # output tensorの取り出し
            res_box   = res.get_tensor(self.output_blob_name).data[:]
            # バウンディングボックス毎の結果を取得
            res_box   = res_box.reshape(-1,5)
            # label tenosrの取り出し(reshape不要)
            res_label = res.get_tensor(self.label_blob_name).data[:]
            for obj, class_id in zip(res_box, res_label) :
                conf = obj[4]                       # confidence for the predicted class(スコア)
                if conf > self.prob_threshold:          # 閾値より大きいものだけ処理
                    pt1 = (np.array((obj[0], obj[1])) / input_size * img_size).astype(int)  # (left,  top   )
                    pt2 = (np.array((obj[2], obj[3])) / input_size * img_size).astype(int)  # (right, bottom)
                    results.append({"conf":conf, "class_id": class_id, "pt1":pt1, "pt2":pt2})
        else :
            raise ValueError('output type unknown')
        
        return disp_frame, results
    # ================================================================================
    
    # 後処理 =======================================================
    def post_process(self, disp_frame, result) :
        # 結果を個別の変数にバラす
        conf     = result["conf"]
        class_id = result["class_id"]
        pt1      = result["pt1"]
        pt2      = result["pt2"]
        
        
        # 検出結果の文字列化
        # ラベルが定義されていればラベルを読み出し、なければclass ID
        if self.labels_map :
            if len(self.labels_map) > class_id :
                class_name = self.labels_map[class_id]
            else :
                class_name = str(class_id)
        else :
            class_name = str(class_id)
        
        # 結果をログファイルorコンソールに出力
        console_print(self.log_f, f'{disp_frame.frame_number:3}:Class={class_name:15}({class_id:3}) Confidence={conf:4f} Location=({pt1[0]},{pt1[1]})-({pt2[0]},{pt2[1]})', False)
        
        # 検出枠の描画
        box_str = f'{class_name} {round(conf * 100, 1)}%'
        disp_frame.draw_box(pt1, pt2, color=disp_frame.get_IndexedColor(class_id), text=box_str)
    
    # ================================================================================
    
    
