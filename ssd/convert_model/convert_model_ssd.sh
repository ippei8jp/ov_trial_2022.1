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

model_names+=('person-vehicle-bike-detection-2000')                     #
model_names+=('person-vehicle-bike-detection-2001')                     #
model_names+=('person-vehicle-bike-detection-2002')                     #
model_names+=('person-vehicle-bike-detection-2003')                     #
model_names+=('person-vehicle-bike-detection-2004')                     #
model_names+=('person-vehicle-bike-detection-crossroad-0078')           #
model_names+=('person-vehicle-bike-detection-crossroad-1016')           #

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
	$(cd ${IR_BASE_DIR}; ln -sf ../intel .)
fi


# ラベルデータファイルの作成
# 本来ならprotobuffで読み込んでごちょごちょやるべきだが、
# (protoファイル作ってprotocでpyに変換したモジュールをimportして、、、という方法をとるらしい...)
# とりあえずIDに抜けがなければこんな姑息な手段で変換できる。
# mscoco
if [ ! -f ${IR_BASE_DIR}/mscoco_complete_label_map.labels ] ; then
	wget -O - https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_complete_label_map.pbtxt \
		| grep display_name | sed -e "s/^.*\"\(.*\)\".*$/\1/g" >  ${IR_BASE_DIR}/mscoco_complete_label_map.labels
fi
# voc
if [ ! -f ${IR_BASE_DIR}/voc_label_map.labels ] ; then
	wget -O - https://raw.githubusercontent.com/weiliu89/caffe/ssd/data/VOC0712/labelmap_voc.prototxt \
		| grep display_name | sed -e "s/^.*\"\(.*\)\".*$/\1/g" >  ${IR_BASE_DIR}/voc_label_map.labels
fi

# vehicle-person-bike その1
if [ ! -f ${IR_BASE_DIR}/vehicle-person-bike.labels ] ; then
	echo "vehicle" >  ${IR_BASE_DIR}/vehicle-person-bike.labels
	echo "person"  >> ${IR_BASE_DIR}/vehicle-person-bike.labels
	echo "bike"    >> ${IR_BASE_DIR}/vehicle-person-bike.labels
fi

# vehicle-person-bike その2
if [ ! -f ${IR_BASE_DIR}/vehicle-person-bike-2.labels ] ; then
	echo "other"    >  ${IR_BASE_DIR}/vehicle-person-bike-2.labels
	echo "person"   >> ${IR_BASE_DIR}/vehicle-person-bike-2.labels
	echo "vehicle"  >> ${IR_BASE_DIR}/vehicle-person-bike-2.labels
	echo "bike"     >> ${IR_BASE_DIR}/vehicle-person-bike-2.labels
fi
