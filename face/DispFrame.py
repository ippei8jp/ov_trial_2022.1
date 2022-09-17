#!/usr/bin/env python3
import sys
import os
import time
import logging as log
import cv2
import numpy as np
import math

class COLORS() :
                    #   B    G    R 
    black         = (   0,   0,   0)  # 黒
    blue          = ( 255,   0,   0)  # 青
    red           = (   0,   0, 255)  # 赤
    magenta       = ( 255,   0, 255)  # マゼンタ
    green         = (   0, 255,   0)  # 緑
    cyan          = ( 255, 255,   0)  # 水色
    yellow        = (   0, 255, 255)  # 黄
    white         = ( 255, 255, 255)  # 白

    gray          = ( 128, 128, 128)  # 灰
    light_blue    = ( 255, 128, 128)  # 青
    light_red     = ( 128, 128, 255)  # 赤
    light_magenta = ( 255, 128, 255)  # マゼンタ
    light_green   = ( 128, 255, 128)  # 緑
    light_cyan    = ( 255, 255, 128)  # 水色
    light_yellow  = ( 128, 255, 255)  # 黄

# 表示フレームクラス ==================================================================
class DispFrame() :
    # カラーパレット(8bitマシン風。ちょっと薄目)
    COLOR_PALETTE = [   #   B    G    R 
                    COLORS.gray,            # 0 (灰)
                    COLORS.light_blue,      # 1 (青)
                    COLORS.light_red,       # 2 (赤)
                    COLORS.light_magenta,   # 3 (マゼンタ)
                    COLORS.light_green,     # 4 (緑)
                    COLORS.light_cyan,      # 5 (水色)
                    COLORS.light_yellow,    # 6 (黄)
                    COLORS.white            # 7 (白)
                ]
    
    # ステータス領域サイズ
    STATUS_LINE_HIGHT   = 15                            # ステータス行の1行あたりの高さ
    STATUS_LINES        =  6                            # ステータス行数
    STATUS_PADDING      =  8                            # ステータス領域の余白
    STATUS_AREA_HIGHT   = STATUS_LINE_HIGHT * STATUS_LINES + STATUS_PADDING # ステータス領域の高さ
    
    def __init__(self, image, frame_number, all_frames) :
        # 画像にステータス表示領域を追加
        # self.image = cv2.copyMakeBorder(image, 0, self.STATUS_AREA_HIGHT, 0, 0, cv2.BORDER_CONSTANT, (0,0,0))
        self.image = image.copy()       # コピーして保持
        self.status_frame = None
        self.image_v = None
        
        # イメージサイズ
        self.img_height = image.shape[0]
        self.img_width  = image.shape[1]
        
        # フレーム番号
        self.frame_number = frame_number
        self.all_frames   = all_frames
        
        # 処理時間計測用変数
        self.preprocess_start     = 0
        self.infer_start          = 0
        self.postprocess_start    = 0
        
        self.frame_time          = 0
        self.preprocess_time     = 0
        self.infer_time          = 0
        self.postprocess_time    = 0
        
    # 画像フレーム表示
    def make_disp_image(self, force_update=False) :
        if force_update or self.image_v is None :
            if self.status_frame is None :
                # statusフレームがなければイメージだけ
                self.image_v = self.image
            else :
                # statusフレームがあったら連結
                self.image_v = cv2.vconcat([self.image, self.status_frame])
        return self.image_v

    def disp_image(self) :
        window_name = "Detection Results"
        
        # 表示イメージの作成
        image_v = self.make_disp_image()
        
        # 表示
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, image_v)                  # 表示
        
        # 画像サイズに合わせてウィンドウサイズ変更
        cv2.resizeWindow(window_name, image_v.shape[1], image_v.shape[0])
    
    # 検出枠の描画
    def draw_box(self, str, pt1, pt2, color=COLORS.cyan) :
        left,  top    = pt1
        right, bottom = pt2
        
        # 対象物の枠とラベルの描画
        cv2.rectangle(self.image,    (left, top     ), (right,      bottom), color,  2)
        if str :
            cv2.rectangle(self.image,    (left, top + 20), (left + 160, top   ), color, -1)
            cv2.putText(self.image, str, (left, top + 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        
        return
    
    # 特徴点の描画
    def draw_point(self, pt1,  color=COLORS.yellow, str=None) :
        cv2.circle(self.image, pt1, 2, color, 2)
        if str :
            cv2.putText(self.image, str, pt1, cv2.FONT_HERSHEY_COMPLEX, 0.5, color)
        
        return
    
    # XYZ軸の描画
    def draw_xyz_axis(self, yaw, pitch, roll, center_pt) :
        # カメラ位置
        camera_distance  = 950
        
        # 度→ラジアン変換
        yaw   = math.radians(yaw)
        pitch = math.radians(pitch)
        roll  = math.radians(roll)
        
        # 各軸に対する回転マトリックス
        yawMatrix   = np.matrix([   [ math.cos(yaw ),                0, -math.sin(yaw  )], 
                                    [              0,                1,                0], 
                                    [ math.sin(yaw ),                0,  math.cos(yaw  )]   ])      # Y軸回転
        
        pitchMatrix = np.matrix([   [              1,                0,                0],
                                    [              0,  math.cos(pitch), -math.sin(pitch)], 
                                    [              0,  math.sin(pitch),  math.cos(pitch)]   ])      # X軸回転
        
        rollMatrix  = np.matrix([   [ math.cos(roll), -math.sin(roll ),                0],
                                    [ math.sin(roll),  math.cos(roll ),                0], 
                                    [              0,                0,                1]   ])      # Z軸回転
        
        # 合成回転マトリックス
        rotationMatrix = yawMatrix * pitchMatrix * rollMatrix
        
        
        # 描画する座標軸
        xAxis  = np.matrix([ [               50, ],
                             [                0, ],
                             [                0, ]  ], dtype=np.float32)         # X軸        左耳方向がプラス
        
        yAxis  = np.matrix([ [                0, ],
                             [              -50, ],
                             [                0, ]  ], dtype=np.float32)         # Y軸        頭上方向がマイナス
        
        zAxis  = np.matrix([ [                0, ],
                             [                0, ],
                             [              -50, ]  ], dtype=np.float32)         # Z軸        顔前方向がマイナス
        
        offset = np.matrix([ [                0, ],
                             [                0, ],
                             [  camera_distance, ]  ], dtype=np.float32)
        
        
        # 各座標軸を回転し、1次元配列化(計算結果はMatrixなので)
        xAxis  = np.asarray(rotationMatrix * xAxis  + offset).flatten()
        yAxis  = np.asarray(rotationMatrix * yAxis  + offset).flatten()
        zAxis  = np.asarray(rotationMatrix * zAxis  + offset).flatten()
        
        # 画面上に投影した座標軸の座標
        x_pt = np.array([xAxis[0] / xAxis[2], xAxis[1] / xAxis[2]]) * camera_distance + center_pt
        y_pt = np.array([yAxis[0] / yAxis[2], yAxis[1] / yAxis[2]]) * camera_distance + center_pt
        z_pt = np.array([zAxis[0] / zAxis[2], zAxis[1] / zAxis[2]]) * camera_distance + center_pt
        
        # 整数化
        center_pt = center_pt.astype(int)
        x_pt      = x_pt.astype(int)
        y_pt      = y_pt.astype(int)
        z_pt      = z_pt.astype(int)
        
        # 座標軸の描画
        cv2.line(       self.image, center_pt, x_pt, COLORS.red,   2)
        cv2.line(       self.image, center_pt, y_pt, COLORS.green, 2)
        cv2.arrowedLine(self.image, center_pt, z_pt, COLORS.blue,  2, tipLength=0.3)
    
    # ステータス表示行座標
    def STATUS_LINE_Y(self, line) : 
        return self.STATUS_LINE_HIGHT * (line + 1)
    
    # ステータス文字列出力
    def status_puts(self, line, message, color=(255, 128, 128)) :
        cv2.putText(self.status_frame, message, (10, self.STATUS_LINE_Y(line)), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
    
    # ステータス表示
    def disp_status(self) :
        # ステータス領域用のイメージ作成
        self.status_frame = np.zeros((self.STATUS_AREA_HIGHT, self.img_width, 3), np.uint8)
        
        # ステータス文字列生成
        frame_number_message    = f'frame_number     : {self.frame_number:5d} / {self.all_frames}'
        if self.frame_time == 0 :
            frame_time_message  =  'Frame time       : ---'
        else :
            frame_time_message      = f'Frame time       : {      self.frame_time:.3f} ms'
        preprocess_time_message     = f'preprocess time  : { self.preprocess_time:.3f} ms'
        infer_time_message          = f'Inference time   : {      self.infer_time:.3f} ms'
        postprocess_time_message    = f'postprocess time : {self.postprocess_time:.3f} ms'
        
        # 文字列の書き込み
        self.status_puts(0, frame_number_message)
        self.status_puts(1, frame_time_message)
        self.status_puts(2, preprocess_time_message)
        self.status_puts(3, infer_time_message)
        self.status_puts(4, postprocess_time_message)

    # 処理時間関連処理
    def set_frame_time(self, frame_time) :
        self.frame_time = frame_time * 1000     # msec単位に変換
    
    def start_preprocess(self, cur_time=None) :
        if cur_time is None :
            cur_time = time.perf_counter()
        self.preprocess_start   = cur_time
    def end_preprocess(self, cur_time=None) :
        if cur_time is None :
            cur_time = time.perf_counter()
        self.preprocess_time     = (cur_time - self.preprocess_start) * 1000     # msec単位に変換
    
    def start_infer(self, cur_time=None) :
        if cur_time is None :
            cur_time = time.perf_counter()
        self.infer_start        = cur_time
    def end_infer(self, cur_time=None) :
        if cur_time is None :
            cur_time = time.perf_counter()
        self.infer_time         = (cur_time - self.infer_start) * 1000           # msec単位に変換
    
    def start_postprocess(self, cur_time=None) :
        if cur_time is None :
            cur_time = time.perf_counter()
        self.postprocess_start  = cur_time
    def end_postprocess(self, cur_time=None) :
        if cur_time is None :
            cur_time = time.perf_counter()
        self.postprocess_time    = (cur_time - self.postprocess_start) * 1000    # msec単位に変換
    
    # 処理時間記録
    def write_time_data(self, time_f) :
        if time_f :
            time_f.write(f'{self.frame_number:5d}, {self.frame_time:.3f}, {self.preprocess_time:.3f}, {self.infer_time:.3f}, {self.postprocess_time:.3f}\n')

# ================================================================================

# 画像保存クラス =================================================================
class ImageSave() :
    # 初期化
    def __init__(self, img_height, img_width) :
        # イメージ領域サイズ
        self.disp_width  = img_width
        self.disp_height = img_height + DispFrame.STATUS_AREA_HIGHT # ステータス領域分を加算しておく
        
        # JPEGファイル名
        self.jpeg_file = None
        # 保存用ライタ
        self.writer    = None
    
    # JPEGファイル名の設定
    def set_jpeg(self, filename) :
        self.jpeg_file = filename
    
    # 動画ファイルのライタ生成
    def create_writer(self, filename, frame_rate) :
        # フォーマット
        # fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        fmt = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(filename, fmt, frame_rate, (self.disp_width, self.disp_height))
    
    # 動画ファイル書き込み
    def write_image(self, frame) :
        if self.jpeg_file :
            cv2.imwrite(self.jpeg_file, frame.image)
        if self.writer:
            # 表示イメージの作成
            image_v = frame.make_disp_image()
            self.writer.write(image_v)
    
    # 動画ファイルのライタ解放
    def release_writer(self) :
        if self.writer:
            self.writer.release()
# ================================================================================

# コンソールとログファイルへの出力 ===============================================
def console_print(log_f, message, both=False, end=None) :
    if (not log_f) or both :
        print(message,end=end)
    if log_f :
        print(message,end=end, file=log_f)
# ================================================================================

