#!/bin/bash

# =====================【注意】========================
# あらかじめ 
#     pip install pandas 
# を実行しておくこと
# =====================================================

# コマンド名
COMMAND_NAME=$0

# 処理本体のファイル名
MAIN_SCRIPT=ov_person_detection.py

# モデルデータのベースディレクトリ
IR_BASE="${PWD}/convert_model/_IR"

# 画像ファイル格納ディレクトリ
IMAGE_DIR=$(realpath ${PWD}/../images)		# 念のため絶対パスにしておく

# 入力ファイル
INPUT_FILE="${IMAGE_DIR}/face-demographics-walking-and-pause.mp4"

# 結果出力ディレクトリ
RESULT_DIR=./_result

# オプション連動変数
device="CPU";				# デフォルトはCPU
log_flag="no"
disp_flag="yes"
allow_long_proc_flag="no"
MODEL_REID_NO="OFF"
MODEL_ATTR_NO="OFF"
cmd_check="no"

# 結果格納ディレクトリを作っておく
mkdir -p ${RESULT_DIR}

# モデル名リスト
    # ==== メモ ==== 
    # inline comment は 「`#～`」 で囲むと可能。
    # 行頭の場合はさらに 「;」 を付け加えると安心。

MODEL_NAMES=()   # 初期化
MODEL_DIRS=()

`#  0`;MODEL_NAMES+=('person-detection-0200')          # intel : 1, 3, 256, 256 : 1, 1, 200, 7 MobileNetV2 
       MODEL_DIRS+=('intel')
`#  1`;MODEL_NAMES+=('person-detection-0201')          # intel : 1, 3, 384, 384 : 1, 1, 200, 7 MobileNetV2 
       MODEL_DIRS+=('intel')
`#  2`;MODEL_NAMES+=('person-detection-0202')          # intel : 1, 3, 512, 512 : 1, 1, 200, 7 MobileNetV2 
       MODEL_DIRS+=('intel')
`#  3`;MODEL_NAMES+=('person-detection-0203')          # intel : 1, 3, 480, 864 : 100, 5       MobileNetV2 
       MODEL_DIRS+=('intel')
`#  4`;MODEL_NAMES+=('person-detection-0301')          # intel : 1, 3, 800,1344 : 100, 5       Resnet50 
       MODEL_DIRS+=('intel')
`#  5`;MODEL_NAMES+=('person-detection-0302')          # intel : 1, 3, 720,1280 : 100, 5       Resnet50 
       MODEL_DIRS+=('intel')
`#  6`;MODEL_NAMES+=('person-detection-0303')          # intel : 1, 3, 720,1280 : 100, 5       MobileNetV2
       MODEL_DIRS+=('intel')
`#  7`;MODEL_NAMES+=('person-detection-0106')          # intel : 1, 3, 800,1344 : 100, 5       R-CNN
       MODEL_DIRS+=('intel')
`#  8`;MODEL_NAMES+=('person-detection-asl-0001')      # intel : 1, 3, 320, 320 : 100, 5       MobileNetV2
       MODEL_DIRS+=('intel')
`#  9`;MODEL_NAMES+=('person-detection-retail-0002')   # intel : 1, 3, 544, 992 : 1, 1, 200, 7 R-FCN 
       MODEL_DIRS+=('intel')
`# 10`;MODEL_NAMES+=('person-detection-retail-0013')   # intel : 1, 3, 320, 544 : 1, 1, 200, 7 MobileNetV2-like
       MODEL_DIRS+=('intel')

# 特徴点推定モデル
# reidentification モデル
`#  0`;MODEL_REID_NAMES+=('person-reidentification-retail-0277')
       MODEL_REID_DIRS+=('intel')
`#  1`;MODEL_REID_NAMES+=('person-reidentification-retail-0286')
       MODEL_REID_DIRS+=('intel')
`#  2`;MODEL_REID_NAMES+=('person-reidentification-retail-0287')
       MODEL_REID_DIRS+=('intel')
`#  3`;MODEL_REID_NAMES+=('person-reidentification-retail-0288')
       MODEL_REID_DIRS+=('intel')

# attributes モデル
`#  0`;MODEL_ATTR_NAMES+=('person-attributes-recognition-crossroad-0230')
       MODEL_ATTR_DIRS+=('intel')
`#  1`;MODEL_ATTR_NAMES+=('person-attributes-recognition-crossroad-0234')
       MODEL_ATTR_DIRS+=('intel')
`#  2`;MODEL_ATTR_NAMES+=('person-attributes-recognition-crossroad-0238')
       MODEL_ATTR_DIRS+=('intel')

# ======== USAGE 表示 =================================================
usage(){
	echo '==== USAGE ===='
	echo "  ${COMMAND_NAME} [option_model] [option_log] [model_number | list | all | allall ] [input_file]"
	echo '    ---- option_model ----'
 	echo '      -c | --cpu : CPUを使用'
	echo '      -n | --ncs : NCS2を使用'
	echo '    ---- option_append_model ----'
	echo '      --reid <model_no>     : 人物同定処理をCPUで実行'
	echo '      --reid_ncs <model_no> : 人物同定処理をNCS2で実行'
	echo '      --attr <model_no>     : 属性検出をCPUで実行'
	echo '      --attr_ncs <model_no> : 属性検出をNCS2で実行'
	echo '    ---- option_log ----'
	echo '      --no_disp  : 表示を省略'
	echo '                       all/allall指定時は指定の有無に関わらず表示を省略'
	echo '      -l | --log : 実行ログを保存(model_number指定時のみ有効'
	echo '                       --no_disp指定時は指定の有無に関わらずログを保存'
	echo '                       all/allall指定時は指定の有無に関わらずログを保存'
	echo '    ---- option_other ----'
	echo '      --allow_long_proc  : 実行時間の長いモデルの実行を許可'
	echo '      --check            : 実行コマンドのチェックのみ行う'
	echo '    input_file 省略時はデフォルトの入力ファイルを使用'
	echo ' '
	disp_list
}

# ======== LIST 表示 =================================================
disp_list() {
	echo '==== MODEL LIST ===='
	count=0
	for MODEL_NAME in ${MODEL_NAMES[@]}
	do 
		echo "${count} : ${MODEL_NAME}"
		count=$(expr ${count} + 1)
	done
	echo ' '
	
	echo '==== --model_reid MODEL LIST ===='
	count=0
	for MODEL_NAME in ${MODEL_REID_NAMES[@]}
	do 
		echo "${count} : ${MODEL_NAME}"
		count=$(expr ${count} + 1)
	done
	echo ' '
	
	echo '==== --model_attr MODEL LIST ===='
	count=0
	for MODEL_NAME in ${MODEL_ATTR_NAMES[@]}
	do 
		echo "${count} : ${MODEL_NAME}"
		count=$(expr ${count} + 1)
	done
	echo ' '
	
	exit
}

# ======== オプション解析 =================================================
# シェル変数を設定するので、$(analyze_options $*) で子プロセスとして実行してはいけない
# オプション解析後の残りのコマンドライン引数は変数 NEW_ARGV に格納される
analyze_options() {
	# オプション解析
	OPT=$(getopt -o cnl -l cpu,ncs,log,no_disp,allow_long_proc,reid:,attr:,reid_ncs:,attr_ncs:,check -- "$@")
	if [ $? != 0 ] ; then
		echo オプション解析エラー
		exit 1
	fi
	
	# コマンドライン引数の書き換え
	eval set -- "${OPT}"
	
	while true; do
		case $1 in
			-c | --cpu)
				device="CPU"
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
			--allow_long_proc)
				allow_long_proc_flag="yes"
				shift ;;
			--check)
				cmd_check="yes"
				shift ;;
			--reid)
				MODEL_REID_NO=$2
				DEVICE_ATTR="CPU"
				shift 2;;						# パラメータの分もずらす
			--reid_ncs)
				MODEL_REID_NO=$2
				DEVICE_REID="NCS"
				shift 2;;						# パラメータの分もずらす
			--attr)
				MODEL_ATTR_NO=$2
				DEVICE_ATTR="CPU"
				shift 2;;						# パラメータの分もずらす
			--attr_ncs)
				MODEL_ATTR_NO=$2
				DEVICE_ATTR="NCS"
				shift 2;;						# パラメータの分もずらす
			--)
				# getoptのオプションと入力データを分けるための--が最後に検出されるので、ここで処理終了
				shift
				break ;;
			*)
				# 未定義オプション(caseパターン書き忘れをトラップ)
				echo "undefined option! : $1" 1>&2
				exit 1 ;;
		esac
	done
	
	# コマンドライン解析後のコマンドライン引数
	NEW_ARGV=$*
}

# ======== コマンド本体実行 =================================================
execute() {
	# モデル名
	local MODEL_NO=$1
	local MODEL_NAME=${MODEL_NAMES[${MODEL_NO}]}
	local MODEL_DIR=${MODEL_DIRS[${MODEL_NO}]}

	# モデルファイル
	local MODEL_FILE="$(get_model_path ${MODEL_DIR} ${MODEL_NAME})"

	#追加オプション
	local log_name_ext=""
	local EXT_OPTION=""
	if [[ "${device}" == "CPU" ]] ; then
		EXT_OPTION+=" --device=${device}"
		log_name_ext+="-cpu"
	elif [[ "${device}" == "MYRIAD" ]] ; then
		EXT_OPTION+=" --device=${device}"
		log_name_ext+="-ncs"
	fi

	# モデルファイルオプションの設定
	EXT_OPTION+=" --model=${MODEL_FILE}"
	
	# 追加モデルファイルのオプション設定
	if [[ "${MODEL_REID_NO}" != "OFF" ]] ;then
		EXT_OPTION+=" --model_reid=$(get_model_path ${MODEL_REID_DIRS[${MODEL_REID_NO}]} ${MODEL_REID_NAMES[${MODEL_REID_NO}]})"
		if [[ "${DEVICE_REID}" == "NCS" ]] ; then
			EXT_OPTION+=" --device_reid=MYRIAD"
			log_name_ext+="-reid_${MODEL_REID_NO}_ncs"
		else
			EXT_OPTION+=" --device_reid=CPU"
			log_name_ext+="-reid_${MODEL_REID_NO}_cpu"
		fi
	fi
	if [[ "${MODEL_ATTR_NO}" != "OFF" ]] ;then
		EXT_OPTION+=" --model_attr=$(get_model_path ${MODEL_ATTR_DIRS[${MODEL_ATTR_NO}]} ${MODEL_ATTR_NAMES[${MODEL_ATTR_NO}]})"
		if [[ "${DEVICE_ATTR}" == "NCS" ]] ; then
			EXT_OPTION+=" --device_attr=MYRIAD"
			log_name_ext+="-attr_${MODEL_ATTR_NO}_ncs"
		else
			EXT_OPTION+=" --device_attr=CPU"
			log_name_ext+="-attr_${MODEL_ATTR_NO}_cpu"
		fi
	fi

	# 入力ファイルオプションの設定
	EXT_OPTION+=" --input=${INPUT_FILE}"

	# 表示オプション関連
	if [[ "${disp_flag}" == "no" ]] ; then
		EXT_OPTION+=" --no_disp"
	else
		log_name_ext+="-disp"
	fi

	# ログファイル関連設定
	if [[ "${log_flag}" == "yes" ]] ; then
		local SAVE_EXT=${INPUT_FILE##*.}    # 入力ファイルの拡張子
		local LOG_NAME_BASE=${RESULT_DIR}/${MODEL_NO}_${MODEL_NAME}${log_name_ext}
		local SAVE_NAME=${LOG_NAME_BASE}.${SAVE_EXT}
		local TIME_FILE=${LOG_NAME_BASE}.time
		local LOG_FILE=${LOG_NAME_BASE}.log
		EXT_OPTION+=" --save=${SAVE_NAME} --time=${TIME_FILE} --log=${LOG_FILE}"
	else
		local LOG_FILE=/dev/stdout
	fi

	# スキップしたい組み合わせがある場合はここでチェック
	if [[ "${allow_long_proc_flag}" != "yes" ]];then
		if  [[ "${MODEL_NAME}" == "person-detection-0301" ]] || \
			[[ "${MODEL_NAME}" == "person-detection-0302" ]] || \
			[[ "${MODEL_NAME}" == "person-detection-0106" ]] ; then
			echo "Do not run this test because it Requires a long time to execute." > ${LOG_FILE}
			echo "Please specify the --allow_long_proc option to execute." >> ${LOG_FILE}
			local MAIN_RET=39 # 実行スキップエラー
			return ${MAIN_RET}
		fi
	fi

	# コマンド実行
	local command="python3 ${MAIN_SCRIPT} ${EXT_OPTION}"
	echo "COMMAND : ${command}"
	if [[ ${cmd_check} == "yes" ]] ; then
		# 実行せずに終了
		local MAIN_RET=0
		return ${MAIN_RET}
	else 
		eval ${command}
		local MAIN_RET=$?
		if [[ "${log_flag}" == "yes" ]] ; then
			# コマンド正常終了
			# echo "RET : ${MAIN_RET}"
			if [ ${MAIN_RET} -eq 0 ]; then
				# 実行時間の平均値を計算してファイルに出力
				head --lines=2 ${TIME_FILE} > ${TIME_FILE}.average
				python3 -c "import sys; import pandas as pd; data = pd.read_csv(sys.argv[1], skiprows=2, skipinitialspace=True, index_col=0); ave=data.mean(); print(ave); print(f'FPS   {1000/ave[\"frame_time\"]}')" ${TIME_FILE} >> ${TIME_FILE}.average
			fi
		fi
	fi
	return ${MAIN_RET}
}

# ======== 自動実行処理 =================================================
all_execute() {
	local time_name_ext=""
	if [[ "${device}" == "CPU" ]] ; then
		time_name_ext+="-cpu"
	elif [[ "${device}" == "MYRIAD" ]] ; then
		time_name_ext+="-ncs2"
	fi

	# 追加モデルファイルのオプション設定
	if [[ "${MODEL_REID_NO}" != "OFF" ]] ;then
		if [[ "${DEVICE_REID}" == "NCS" ]] ; then
			time_name_ext+="-reid_${MODEL_REID_NO}_ncs"
		else
			time_name_ext+="-reid_${MODEL_REID_NO}_cpu"
		fi
	fi
	if [[ "${MODEL_ATTR_NO}" != "OFF" ]] ;then
		if [[ "${DEVICE_ATTR}" == "NCS" ]] ; then
			time_name_ext+="-attr_${MODEL_ATTR_NO}_ncs"
		else
			time_name_ext+="-attr_${MODEL_ATTR_NO}_cpu"
		fi
	fi
	
	if [[ "${disp_flag}" == "yes" ]] ; then
		time_name_ext+="-disp"
	fi

	# 実行時間記録ファイル
	local TIME_LOG=${RESULT_DIR}/time${time_name_ext}.txt
	echo "各モデルの実行時間" > ${TIME_LOG}
	
	# 各モデルに対する処理ループ
	for ix in ${!MODEL_NAMES[@]}; do
		local MODEL_NO=${ix}
		local MODEL_NAME=${MODEL_NAMES[${MODEL_NO}]}

		# 前処理
		echo "######## ${MODEL_NO} ${MODEL_NAME} ########" | tee -a ${TIME_LOG}
		
		# 実行開始時刻(秒で取得して日付で表示)
		local start_time=$(date +%s)
		echo "***START*** : $(date -d @${start_time} +'%Y/%m/%d %H:%M:%S')" | tee -a ${TIME_LOG}
		
		# 処理本体
		execute ${MODEL_NO}
		local EXEC_RET=$?		# 処理本体の戻り値
		
		# 後処理
		if [ ${EXEC_RET} -eq 0 ]; then
			# 実行終了時刻(秒で取得して日付で表示)
			local end_time=$(date +%s)
			echo "*** END *** : $(date -d @${end_time} +'%Y/%m/%d %H:%M:%S')" | tee -a ${TIME_LOG}
			# 実行時間(ちょっと姑息な方法で "秒数" を "時:分:秒" に変換)
			local execution_time=$(expr ${end_time} - ${start_time})
			echo "=== Execution time : $(TZ=0 date -d@${execution_time} +%H:%M:%S)" | tee -a ${TIME_LOG}
		elif [ ${EXEC_RET} -eq 39 ]; then
			echo "***SKIP*** : ${MODEL_NAME} はスキップしました " | tee -a ${TIME_LOG}
		else
			echo "***ERROR*** : ${MODEL_NAME} でエラーが発生しました " | tee -a ${TIME_LOG}
		fi
	done
}
# ======== 全自動実行処理 =================================================
allall_execute() {
	for dev in "CPU" "MYRIAD"; do
		device=${dev}
		all_execute
	done
}

# ======== 数値パラメータのチェック =========================================
# $(check_numerical_parameter $MODEL_NO $num_models) で子プロセスとして実行し、
# 戻り値が0ならOK、それ以外ならNG
check_numerical_parameter() {
	local MODEL_NO=$1
	local num_models=$2

	# echo "${MODEL_NO}  ${num_models}"
	
	# パラメータが数値か確認
	expr "${MODEL_NO}" + 1 >/dev/null 2>&1
	if [ $? -ge 2 ] ; then
		# echo "${MODEL_NO}"
		echo " 0 以上 ${num_models} 未満の数値を指定してください(1)"
		return 1
	fi
	# パラメータが 正数 かつ 配列数未満 か確認
	if [ ${MODEL_NO} -ge 0 ] && [ ${MODEL_NO} -lt ${num_models} ] ; then
		return 0
	fi
	echo " 0 以上 ${num_models} 未満の数値を指定してください(2)"
	return 1
}

# ======== モデルのパスを取得 =========================================
# $(get_model_path $DIR $NAME) で子プロセスとして実行し、戻り値で結果を取得
get_model_path() {
	local M_DIR=$1
	local M_NAME=$2
	echo "${IR_BASE}/${M_DIR}/${M_NAME}/FP16/${M_NAME}.xml"
}


# ======== メイン処理 =========================================
# オプション解析
analyze_options $*

# コマンドライン引数の書き換え
eval set -- "${NEW_ARGV}"

# ここに来た時点でオプションのコマンドライン引数は削除されている

# 引数の個数
num_args=$#
if [ ${num_args} -eq 0 ] ;then
	usage			# USAGE表示
	exit
fi

if [ ${num_args} -ge 2 ] ;then
	# 第2パラメータがあったら入力ファイルを変更
	INPUT_FILE=$2
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

# パラメータが数値か確認
check_numerical_parameter ${MODEL_NO} ${#MODEL_NAMES[@]}
if [ $? -ne 0 ] ; then
	echo "1st param check error"
	usage
	exit
fi

if [[ "${MODEL_REID_NO}" != "OFF" ]] ;then
	# echo "REID param check   ${MODEL_REID_NO}"
	check_numerical_parameter ${MODEL_REID_NO} ${#MODEL_REID_NAMES[@]}
	if [ $? -ne 0 ] ; then
		echo "REID param check error"
		usage
		exit
	fi
fi

if [[ "${MODEL_ATTR_NO}" != "OFF" ]] ;then
	# echo "ATTR param check ${MODEL_ATTR_NO}"
	check_numerical_parameter ${MODEL_ATTR_NO} ${#MODEL_ATTR_NAMES[@]}
	if [ $? -ne 0 ] ; then
		echo "ATTR param check error"
		usage
		exit
	fi
fi

# 実行
execute ${MODEL_NO}

exit
