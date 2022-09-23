#!/bin/bash

# モデルを出力するディレクトリ
IR_BASE_DIR=$PWD/_IR

mkdir -p ${IR_BASE_DIR}

#### 各モデル名称と #####################################
# 使用できるモデル一覧は以下を実行して取得
# omz_downloader  --print_all
declare -a model_names=()       # モデル名
model_names+=('face-detection-retail-0004')     # output shape : 1, 1, N, 7
model_names+=('face-detection-retail-0005')     # output shape : 1, 1, N, 7
model_names+=('face-detection-retail-0044')     # output shape : 1, 1, N, 7
model_names+=('face-detection-adas-0001')       # output shape : 1, 1, N, 7
model_names+=('face-detection-0200')            # output shape : 1, 1, N, 7
model_names+=('face-detection-0202')            # output shape : 1, 1, N, 7
model_names+=('face-detection-0204')            # output shape : 1, 1, N, 7

model_names+=('face-detection-0205')            # output shape : 200, 5
model_names+=('face-detection-0206')            # output shape : 750, 5


model_names+=('landmarks-regression-retail-0009')
model_names+=('facial-landmarks-35-adas-0002')
model_names+=('head-pose-estimation-adas-0001')


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


