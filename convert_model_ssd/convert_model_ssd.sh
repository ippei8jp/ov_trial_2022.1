#!/bin/bash

# モデルを出力するディレクトリ
IR_BASE_DIR=$PWD/_IR

#### 各モデル名称と #####################################
# 使用できるモデル一覧は以下を実行して取得
# omz_downloader  --print_all
declare -a model_names=()       # モデル名
model_names+=('mobilenet-ssd')              # caffe
model_names+=('ssd300')                     # caffe
model_names+=('ssd512')                     # caffe
model_names+=('ssd_mobilenet_v1_coco')      # tensorflow
model_names+=('ssd_mobilenet_v1_fpn_coco')  # tensorflow
model_names+=('ssdlite_mobilenet_v2')       # tensorflow

for ix in ${!model_names[@]}; do
	modelname=${model_names[ix]}
	omz_downloader \
	    --name ${modelname}
	
	omz_converter \
	    --precisions FP16 \
	    --name ${modelname} \
	    --output_dir ${IR_BASE_DIR}
done

# ラベルデータファイルの作成
# 本来ならprotobuffで読み込んでごちょごちょやるべきだが、
# (protoファイル作ってprotocでpyに変換したモジュールをimportして、、、という方法をとるらしい...)
# とりあえずIDに抜けがなければこんな姑息な手段で変換できる。
# mscoco
wget -O - https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_complete_label_map.pbtxt \
	| grep display_name | sed -e "s/^.*\"\(.*\)\".*$/\1/g" >  ${IR_BASE_DIR}/mscoco_complete_label_map.labels
# voc
wget -O - https://raw.githubusercontent.com/weiliu89/caffe/ssd/data/VOC0712/labelmap_voc.prototxt \
	| grep display_name | sed -e "s/^.*\"\(.*\)\".*$/\1/g" >  ${IR_BASE_DIR}/voc_label_map.labels


