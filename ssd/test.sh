#!/bin/bash

# =====================【注意】========================
# あらかじめ 
#     pip install pandas 
# を実行しておくこと
# =====================================================

# コマンド名
COMMAND_NAME=$0

# 処理本体のファイル名
MAIN_SCRIPT=ov_object_detection_ssd.py

# プロジェクトベースディレクトリ
BASE_DIR=$(realpath $(pwd)/..)				# 念のため絶対パスにしておく
											# `～`だとネストできないので、$(～)で

# モデルデータのベースディレクトリ
IR_BASE=${BASE_DIR}/convert_model_ssd/_IR

# 入出力ファイル/ディレクトリ
COCO_LABEL_FILE=${IR_BASE}/mscoco_complete_label_map.labels 
VOC_LABEL_FILE=${IR_BASE}/voc_label_map.labels 
INPUT_DEFAULT_FILE=${BASE_DIR}/images/testvideo3.mp4 
RESULT_DIR=./_result

# 結果格納ディレクトリを作っておく
mkdir -p ${RESULT_DIR}

# モデル名リスト
    # ==== メモ ==== 
    # inline comment は 「`#～`」 で囲むと可能。
    # 行頭の場合はさらに 「;」 を付け加えると安心。

MODEL_NAMES=()   # 初期化
LABEL_FILES=()   # 初期化

`#  0`;MODEL_NAMES+=("mobilenet-ssd")
       LABEL_FILES+=(${VOC_LABEL_FILE})
`#  1`;MODEL_NAMES+=("ssd300")
       LABEL_FILES+=(${VOC_LABEL_FILE})
`#  2`;MODEL_NAMES+=("ssd512")
       LABEL_FILES+=(${VOC_LABEL_FILE})
`#  3`;MODEL_NAMES+=("ssd_mobilenet_v1_coco")
       LABEL_FILES+=(${COCO_LABEL_FILE})
`#  4`;MODEL_NAMES+=("ssd_mobilenet_v1_fpn_coco")
       LABEL_FILES+=(${COCO_LABEL_FILE})
`#  5`;MODEL_NAMES+=("ssdlite_mobilenet_v2")
       LABEL_FILES+=(${COCO_LABEL_FILE})

# ======== USAGE 表示 =================================================
usage(){
	echo '==== USAGE ===='
	echo "  ${COMMAND_NAME} [option_model] [option_log] [model_number | list | all | allall ] [input_file]"
	echo '    ---- option_model ----'
 	echo '      -c | --cpu : CPUを使用'
	echo '      -n | --ncs : NCS2を使用'
	echo '    ---- option_log ----'
	echo '      -l | --log : 実行ログを保存(model_number指定時のみ有効'
	echo '                       all/allall指定時は指定の有無に関わらずログを保存'
	echo '      --no_disp  : 表示を省略'
	echo '                       --log指定時は指定の有無に関わらず表示を省略'
	echo '                       all/allall指定時は指定の有無に関わらず表示を省略'
	echo '      --sync     : 同期モードで実行'
	echo '                       省略時は非同期モードで実行'
	echo '    input_file 省略時はデフォルトの入力ファイルを使用'
	echo ' '
}

# ======== USAGE 表示 =================================================
disp_list() {
	# usage 表示
	usage
	echo '==== MODEL LIST ===='
	count=0
	for MODEL_NAME in ${MODEL_NAMES[@]}
	do 
		echo "${count} : ${MODEL_NAME}"
		count=`expr ${count} + 1`
	done
	echo ' '
	exit
}

# python ov_object_detection_ssd.py \
# --model ssd_mobilenet_v2_coco/FP16/ssd_mobilenet_v2_coco.xml  \
# --labels mscoco_complete_label_map.labels 
# --device MYRIAD 
# --input ../images/testvideo3.mp4 
# --save c.mp4 
# --time c.time 
# --log c.log 
# --sync
# ======== コマンド本体実行 =================================================
execute() {
	# モデルファイル
	local MODEL_FILE=${IR_BASE}/public/${MODEL_NAME}/FP16/${MODEL_NAME}.xml
	if [ ! -f ${MODEL_FILE} ]; then
		# 指定されたモデルファイルが存在しない
		echo "==== モデルファイル ${MODEL_FILE} は存在しません ===="
		return 1
	fi
	echo "MODEL_FILE : ${MODEL_FILE}"
	echo "LABEL_FILE : ${LABEL_FILE}"
	#追加オプション
	model_name_ext=""
	EXT_OPTION=""
	if [[ "${device}" == "CPU" ]] ; then
		model_name_ext+="_cpu"
	elif [[ "${device}" == "MYRIAD" ]] ; then
			model_name_ext+="_ncs2"
	fi
	if [[ "${sync_flag}" == "yes" ]] ; then
		EXT_OPTION+=" --sync"
		model_name_ext+="_sync"
	else
		model_name_ext+="_async"
	fi
	if [[ "${disp_flag}" == "no" ]] ; then
		EXT_OPTION+=" --no_disp"
	else
		model_name_ext+="_disp"
	fi
	if [[ "${log_flag}" == "yes" ]] ; then
		local SAVE_EXT=${INPUT_FILE##*.}    # 入力ファイルの拡張子
		local LOG_NAME_BASE=${RESULT_DIR}/${MODEL_NAME}${model_name_ext}
		local SAVE_NAME=${LOG_NAME_BASE}.${SAVE_EXT}
		local TIME_FILE=${LOG_NAME_BASE}.time
		local LOG_FILE=${LOG_NAME_BASE}.log
		EXT_OPTION+=" --save ${SAVE_NAME} --time ${TIME_FILE} --log ${LOG_FILE}"
	fi
	command="python3 ${MAIN_SCRIPT} --device=${device} --input ${INPUT_FILE} --label ${LABEL_FILE} --model ${MODEL_FILE} ${EXT_OPTION}"
	echo "COMMAND : ${command}"
	eval ${command}
	
	MAIN_RET=$?
	if [[ "${log_flag}" == "yes" ]] ; then
		# echo "RET : ${MAIN_RET}"
		if [ ${MAIN_RET} -eq 0 ]; then
			# 実行時間の平均値を計算してファイルに出力
			python3 -c "import sys; import pandas as pd; data = pd.read_csv(sys.argv[1], index_col=0); ave=data.mean(); print(ave)" ${TIME_FILE} > ${TIME_FILE}.average
		fi
	fi
	
	return $MAIN_RET
}

# ======== 自動実行処理 =================================================
all_execute() {
	time_name_ext=""
	if [[ "${device}" == "CPU" ]] ; then
		time_name_ext+="_cpu"
	elif [[ "${device}" == "MYRIAD" ]] ; then
		time_name_ext+="_ncs2"
	fi
	if [[ "${sync_flag}" == "yes" ]] ; then
		time_name_ext+="_sync"
	else 
		time_name_ext+="_async"
	fi
	if [[ "${disp_flag}" == "yes" ]] ; then
		time_name_ext+="_disp"
	fi

	# 実行時間記録ファイル
	local TIME_LOG=${RESULT_DIR}/time${time_name_ext}.txt
	echo "各モデルの実行時間" > ${TIME_LOG}
	
	# 各モデルに対する処理ループ
	# for MODEL_NAME in ${MODEL_NAMES[@]}
	# do 
	for ix in ${!MODEL_NAMES[@]}; do
		MODEL_NAME=${MODEL_NAMES[ix]}
		LABEL_FILE=${LABEL_FILES[ix]}

		# 前処理
		echo "######## ${MODEL_NAME} ########" | tee -a ${TIME_LOG}
		
		# 実行開始時刻(秒で取得して日付で表示)
		local start_time=`date +%s`
		echo "***START*** : `date -d @${start_time} +'%Y/%m/%d %H:%M:%S'`" | tee -a ${TIME_LOG}
		
		# if [[ ${MODEL_NAME} == "ssd_mobilenet_v1_fpn_coco" && ${devic} == "MYRIAD" ]];
		# then
		# 	echo "***SKIP *** : ${MODEL_NAME} の ${devic} は大きすぎてエラーになるのでスキップします" | tee -a ${TIME_LOG}
		# 	continue
		# fi
		
		# 処理本体
		execute
		local EXEC_RET=$?		# 処理本体の戻り値
		
		# 後処理
		if [ ${EXEC_RET} -eq 0 ]; then
			# 実行終了時刻(秒で取得して日付で表示)
			local end_time=`date +%s`
			echo "*** END *** : `date -d @${end_time} +'%Y/%m/%d %H:%M:%S'`" | tee -a ${TIME_LOG}
			# 実行時間(ちょっと姑息な方法で "秒数" を "時:分:秒" に変換)
			local execution_time=$(expr ${end_time} - ${start_time})
			echo "=== Execution time : `TZ=0 date -d@${execution_time} +%H:%M:%S`" | tee -a ${TIME_LOG}
		else
			echo "***ERROR*** : ${MODEL_NAME} でエラーが発生しました " | tee -a ${TIME_LOG}
		fi
	done
}
# ======== 全自動実行処理 =================================================
allall_execute() {
	for syn in "no" "yes"; do
		sync_flag=${syn}
		for dev in "CPU" "MYRIAD"; do
			device=${dev}
			all_execute
		done
	done
}

# #################################################################################
# オプション連動変数
device="CPU";				# デフォルトはCPU
log_flag="no"
disp_flag="yes"
sync_flag="no"

# オプション解析
OPT=`getopt -o cnl -l cpu,ncs,log,no_disp,sync -- "$@"`
if [ $? != 0 ] ; then
	echo オプション解析エラー
	exit 1
fi
eval set -- "$OPT"

while true; do
	case $1 in
		-c | --cpu)
			device="CPU";
			shift ;;
		-n | --ncs)
			device="MYRIAD"
			shift ;;
		-l | --log)
			log_flag="yes"
			shift ;;
		--no_disp)
			disp_flag="no"
			log_flag="yes"					# no_disp時はログ有効
			shift ;;
		--sync)
			sync_flag="yes"
			shift ;;
		--)
			# getoptのオプションと入力データを分けるための--が最後に検出されるので、ここで処理終了
			shift; break ;;
		*)
			# 未定義オプション
			echo "Internal error!" 1>&2
			exit 1 ;;
	esac
done
# #################################################################################

# 引数の個数
num_args=$#
if [ ${num_args} -eq 0 ] ;then
	disp_list			# リスト表示
	exit
fi

if [ ${num_args} -ge 2 ] ;then
	# 第2パラメータがあったら入力ファイルを変更
	INPUT_FILE=$2
else
	INPUT_FILE=${INPUT_DEFAULT_FILE}
fi

if [[ "$1" == "list" ]] ;then
	disp_list			# リスト表示
	exit
elif [[ "$1" == "all" ]] ;then
    log_flag="yes"		# ログ保存有効
	disp_flag="no"		# 表示しない
	all_execute			# 自動実行
	exit
elif [[ "$1" == "allall" ]] ;then
    log_flag="yes"		# ログ保存有効
	disp_flag="no"		# 表示しない
	allall_execute		# 自動実行
	exit
else
	MODEL_NO=$1			# モデル番号
fi

num_models=${#MODEL_NAMES[@]}

# パラメータが数値か確認
expr "${MODEL_NO}" + 1 >/dev/null 2>&1
if [ $? -ge 2 ]
then
  echo "${MODEL_NO}"
  echo " 0 以上 $num_models 未満の数値を指定してください(1)"
  usage
  exit
fi
# パラメータが正数か確認
if [ ${MODEL_NO} -lt 0 ] ; then
  echo " 0 以上 $num_models 未満の数値を指定してください(3)"
  usage
  exit
fi
# パラメータが配列数未満か確認
if [ ${MODEL_NO} -ge ${num_models} ] ; then
  echo " 0 以上 $num_models 未満の数値を指定してください(2)"
  usage
  exit
fi

# モデル名
MODEL_NAME=${MODEL_NAMES[${MODEL_NO}]}
LABEL_FILE=${LABEL_FILES[${MODEL_NO}]}

# 実行
execute

exit
