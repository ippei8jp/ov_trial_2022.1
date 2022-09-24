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
class model_person_reid(sync_model_base) :
    def __init__(self, core, model_xml, device="CPU", threshold=0.8, log_f=None) :
        # 親クラスの初期化をcall
        super().__init__(core, model_xml, device, threshold, log_f)
        
        # reidベクトルの配列
        self.ReIdVectors = []
    
    # output blobの確認 ===============================================
    def check_output_blob(self) :
        # 出力レイヤ数のチェックと名前の取得
        log.info("Check outputs")
        outputs = self.model.outputs
        assert len(outputs) == 1, "Demo supports only single output topologies" # 出力レイヤ数は1のみ対応
        self.output_blob_name = outputs[0].get_any_name()       # 出力レイヤ名
        self.output_blob_shape = outputs[0].shape
        assert tuple(self.output_blob_shape) == (1, 256), f"output shape must be (1, 256), but it is {self.output_blob_shape}" # 出力レイヤのshape確認
    
    # 結果の解析 ===============================================
    def analyze_result(self, res, params) :
        # params未使用
        
        # output tensorの取り出し
        newReIdVec = res.get_tensor(self.output_blob_name).data[0].copy()
        
        # 見つからなかったらappendしたときのindex(=現在のサイズ)を返すので初期値とする
        reid = len(self.ReIdVectors)
        
        for i, vec in enumerate(self.ReIdVectors) :
            cossim = self.cosineSimilarity(newReIdVec, vec);
            # print(f'{i} : cossim = {cossim}')
            if cossim > self.threshold :                # 閾値以上
                # 以前のreidベクトルと類似していると判定
                reid = i
                self.ReIdVectors[i] = newReIdVec        # 現在のベクトルで置き換え
                break
        else :  # forループをbreakせずに抜けてきた
            # 新しいベクトルとして登録
            self.ReIdVectors.append(newReIdVec)
        
        result = {"result": reid}

        return result
    # ================================================================================
    
    # 後処理 =======================================================
    def post_process(self, cur_frame, result, pt1_ex, pt2_ex) :
        reid   = result["result"]
        
        # 結果をログファイルorコンソールに出力
        console_print(self.log_f, f'     REID={reid:3d}', False)
    
    # ================================================================================
    
    # cos類似度の計算 =====================================================================
    # https://atmarkit.itmedia.co.jp/ait/articles/2112/08/news020.html
    # https://qiita.com/Qiitaman/items/fa393d93ce8e61a857b1
    @staticmethod
    def cosineSimilarity(v1, v2) :
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    # ================================================================================
    
