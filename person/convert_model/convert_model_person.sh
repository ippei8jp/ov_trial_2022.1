#!/bin/bash

# モデルを出力するディレクトリ
IR_BASE_DIR=$PWD/_IR

mkdir -p ${IR_BASE_DIR}

#### 各モデル名称と #####################################
# 使用できるモデル一覧は以下を実行して取得
# omz_downloader  --print_all
declare -a model_names=()       # モデル名

model_names+=('person-detection-0200')          # intel : 1, 3, 256, 256 : 1, 1, 200, 7 MobileNetV2 
model_names+=('person-detection-0201')          # intel : 1, 3, 384, 384 : 1, 1, 200, 7 MobileNetV2 
model_names+=('person-detection-0202')          # intel : 1, 3, 512, 512 : 1, 1, 200, 7 MobileNetV2 
model_names+=('person-detection-0203')          # intel : 1, 3, 480, 864 : 100, 5       MobileNetV2 
model_names+=('person-detection-0301')          # intel : 1, 3, 800,1344 : 100, 5       Resnet50 
model_names+=('person-detection-0302')          # intel : 1, 3, 720,1280 : 100, 5       Resnet50 
model_names+=('person-detection-0303')          # intel : 1, 3, 720,1280 : 100, 5       MobileNetV2
model_names+=('person-detection-0106')          # intel : 1, 3, 800,1344 : 100, 5       R-CNN
model_names+=('person-detection-asl-0001')      # intel : 1, 3, 320, 320 : 100, 5       MobileNetV2
model_names+=('person-detection-retail-0002')   # intel : 1, 3, 544, 992 : 1, 1, 200, 7 R-FCN 
model_names+=('person-detection-retail-0013')   # intel : 1, 3, 320, 544 : 1, 1, 200, 7 MobileNetV2-like

# reidentification モデル
model_names+=('person-reidentification-retail-0277')
model_names+=('person-reidentification-retail-0286')
model_names+=('person-reidentification-retail-0287')
model_names+=('person-reidentification-retail-0288')


# attributes モデル
model_names+=('person-attributes-recognition-crossroad-0230')
model_names+=('person-attributes-recognition-crossroad-0234')
model_names+=('person-attributes-recognition-crossroad-0238')

<< _COMMENT_
==== メモ ===================
listで 'person' を含むモデル

# 以下は人物検出モデル
# person-detection-0106
# person-detection-0200
# person-detection-0201
# person-detection-0202
# person-detection-0203
# person-detection-0301
# person-detection-0302
# person-detection-0303
# person-detection-asl-0001
# person-detection-retail-0002
# person-detection-retail-0013

# 以下は属性認識(男/バッグ持ってる/帽子かぶってる/など)と上半身/下半身の色の座標？
# 検出モデルで切り取った画像を入力する
person-attributes-recognition-crossroad-0230
person-attributes-recognition-crossroad-0234
person-attributes-recognition-crossroad-0238

# 以下はセグメンテーション
# 今回は対象外
instance-segmentation-person-0007

# 以下はアクション検出(起立/着席/挙手)
# SSDに付随しているらしい
# 今回は対象外
person-detection-action-recognition-0005
person-detection-action-recognition-0006
person-detection-action-recognition-teacher-0002
person-detection-raisinghand-recognition-0001

# 同一人物の判定(256の判定データを出力)
# 検出モデルで切り取った画像を入力する
person-reidentification-retail-0277
person-reidentification-retail-0286
person-reidentification-retail-0287
person-reidentification-retail-0288

# 以下は人/車/バイク
# 今回は対象外
person-vehicle-bike-detection-2000
person-vehicle-bike-detection-2001
person-vehicle-bike-detection-2002
person-vehicle-bike-detection-2003
person-vehicle-bike-detection-2004
person-vehicle-bike-detection-crossroad-0078
person-vehicle-bike-detection-crossroad-1016
person-vehicle-bike-detection-crossroad-yolov3-1020

# listで 'human' を含むモデル
# 今回は対象外
higher-hrnet-w32-human-pose-estimation
human-pose-estimation-0001
human-pose-estimation-0005
human-pose-estimation-0006
human-pose-estimation-0007
human-pose-estimation-3d-0001
single-human-pose-estimation-0001

_COMMENT_


for ix in ${!model_names[@]}; do
	modelname=${model_names[ix]}

	if [ -d ${IR_BASE_DIR}/*/${modelname} ] ; then 
		echo "${modelname} already downloaded."
	else
		omz_downloader \
		    --name ${modelname}
		
		omz_converter \
		    --precisions FP16 \
		    --name ${modelname} \
		    --output_dir ${IR_BASE_DIR}
	fi
done

# intelディレクトリはconvertしないのでシンボリックリンクで位置合わせ
if [ -d intel ] ; then
	ln -sfr ./intel ${IR_BASE_DIR}
fi


