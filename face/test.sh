#!/bin/bash

# =====================【注意】========================
# あらかじめ 
#     pip install pandas 
# を実行しておくこと
# =====================================================

# コマンド名
COMMAND_NAME=$0

# 処理本体のファイル名
MAIN_SCRIPT=ov_face_detection.py

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
lm5_flag="no"
lm35_flag="no"
hp_flag="no"
cmd_check="no"

# 結果格納ディレクトリを作っておく
mkdir -p ${RESULT_DIR}

# モデル名リスト
    # ==== メモ ==== 
    # inline comment は 「`#～`」 で囲むと可能。
    # 行頭の場合はさらに 「;」 を付け加えると安心。

MODEL_NAMES=()   # 初期化
MODEL_DIRS=()

`#  0`;MODEL_NAMES+=('face-detection-retail-0004')     # output shape : 1, 1, N, 7
       MODEL_DIRS+=('intel')
`#  1`;MODEL_NAMES+=('face-detection-retail-0005')     # output shape : 1, 1, N, 7
       MODEL_DIRS+=('intel')
`#  2`;MODEL_NAMES+=('face-detection-retail-0044')     # output shape : 1, 1, N, 7
       MODEL_DIRS+=('public')
`#  3`;MODEL_NAMES+=('face-detection-adas-0001')       # output shape : 1, 1, N, 7
       MODEL_DIRS+=('intel')
`#  4`;MODEL_NAMES+=('face-detection-0200')            # output shape : 1, 1, N, 7
       MODEL_DIRS+=('intel')
`#  5`;MODEL_NAMES+=('face-detection-0202')            # output shape : 1, 1, N, 7
       MODEL_DIRS+=('intel')
`#  6`;MODEL_NAMES+=('face-detection-0204')            # output shape : 1, 1, N, 7
       MODEL_DIRS+=('intel')
`#  7`;MODEL_NAMES+=('face-detection-0205')            # output shape : 200, 5
       MODEL_DIRS+=('intel')
`#  8`;MODEL_NAMES+=('face-detection-0206')            # output shape : 750, 5      NCSではエラーになります
       MODEL_DIRS+=('intel')

# 特徴点推定モデル
MODEL_LM5_NAME='landmarks-regression-retail-0009'
MODEL_LM5_DIR='intel'
MODEL_LM35_NAME='facial-landmarks-35-adas-0002'
MODEL_LM35_DIR='intel'
MODEL_HP_NAME='head-pose-estimation-adas-0001'
MODEL_HP_DIR='intel'

# ======== USAGE 表示 =================================================
usage(){
	echo '==== USAGE ===='
	echo "  ${COMMAND_NAME} [option_model] [option_log] [model_number | list | all | allall ] [input_file]"
	echo '    ---- option_model ----'
 	echo '      -c | --cpu : CPUを使用'
	echo '      -n | --ncs : NCS2を使用'
	echo '    ---- option_append_model ----'
	echo '      --lm5      : 特徴点検出(5点)をCPUで実行'
	echo '      --lm5_ncs  : 特徴点検出(5点)をNCS2で実行'
	echo '      --lm35     : 特徴点検出(35点)をCPUで実行'
	echo '      --lm35_ncs : 特徴点検出(35点)をNCS2で実行'
	echo '      --hp       : 顔の向き推定をCPUで実行'
	echo '      --hp_ncs   : 顔の向き推定をNCS2で実行'
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
	
	exit
}

# ======== オプション解析 =================================================
# シェル変数を設定するので、$(analyze_options $*) で子プロセスとして実行してはいけない
# オプション解析後の残りのコマンドライン引数は変数 NEW_ARGV に格納される
analyze_options() {
	# オプション解析
	OPT=$(getopt -o cnl -l cpu,ncs,log,no_disp,allow_long_proc,lm5,lm5_ncs,lm35,lm35_ncs,hp,hp_ncs,check -- "$@")
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
			--lm5)
				lm5_flag="CPU"
				shift ;;
			--lm5_ncs)
				lm5_flag="NCS"
				shift ;;
			--lm35)
				lm35_flag="CPU"
				shift ;;
			--lm35_ncs)
				lm35_flag="NCS"
				shift ;;
			--hp)
				hp_flag="CPU"
				shift ;;
			--hp_ncs)
				hp_flag="NCS"
				shift ;;
			--)
				# getoptのオプションと入力データを分けるための--が最後に検出されるので、ここで処理終了
				shift; break ;;
			*)
				# 未定義オプション
				echo "undefined option!" 1>&2
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
	if [[ "${lm5_flag}" == "CPU" ]] ; then
		EXT_OPTION+=" --model_lm5=$(get_model_path ${MODEL_LM5_DIR} ${MODEL_LM5_NAME}) --device_lm5=CPU"
		log_name_ext+="-lm5_cpu"
	elif [[ "${lm5_flag}" == "NCS" ]] ; then
		EXT_OPTION+=" --model_lm5=$(get_model_path ${MODEL_LM5_DIR} ${MODEL_LM5_NAME}) --device_lm5=MYRIAD"
		log_name_ext+="-lm5_ncs"
	else
		log_name_ext+="-lm5_off"
	fi
	if [[ "${lm35_flag}" == "CPU" ]] ; then
		EXT_OPTION+=" --model_lm35=$(get_model_path ${MODEL_LM35_DIR} ${MODEL_LM35_NAME}) --device_lm35=CPU"
		log_name_ext+="-lm35_cpu"
	elif [[ "${lm35_flag}" == "NCS" ]] ; then
		EXT_OPTION+=" --model_lm35=$(get_model_path ${MODEL_LM35_DIR} ${MODEL_LM35_NAME}) --device_lm35=MYRIAD"
		log_name_ext+="-lm35_ncs"
	else
		log_name_ext+="-lm35_off"
	fi
	if [[ "${hp_flag}" == "CPU" ]] ; then
		EXT_OPTION+=" --model_hp=$(get_model_path ${MODEL_HP_DIR} ${MODEL_HP_NAME}) --device_hp=CPU"
		log_name_ext+="-hp_cpu"
	elif [[ "${hp_flag}" == "NCS" ]] ; then
		EXT_OPTION+=" --model_hp=$(get_model_path ${MODEL_HP_DIR} ${MODEL_HP_NAME}) --device_hp=MYRIAD"
		log_name_ext+="-hp_ncs"
	else
		log_name_ext+="-hp_off"
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
	if [[ "${device}" == "MYRIAD" ]] ; then
		if  [[ "${MODEL_NAME}" == "face-detection-0205" ]] || \
			[[ "${MODEL_NAME}" == "face-detection-0206" ]] ; then
			echo "Do not run this test because it contains layers not supported by NCS" > ${LOG_FILE}
			local MAIN_RET=39 # 実行スキップエラー
			return ${MAIN_RET}
		fi
	fi
	if [[ "${allow_long_proc_flag}" != "yes" ]];then
		if  [[ "${MODEL_NAME}" == "face-detection-0206" ]] ; then
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
	if [[ "${lm5_flag}" == "CPU" ]] ; then
		time_name_ext+="-lm5_cpu"
	elif [[ "${lm5_flag}" == "NCS" ]] ; then
		time_name_ext+="-lm5_ncs"
	else
		time_name_ext+="-lm5_off"
	fi
	if [[ "${lm35_flag}" == "CPU" ]] ; then
		time_name_ext+="-lm35_cpu"
	elif [[ "${lm35_flag}" == "NCS" ]] ; then
		time_name_ext+="-lm35_ncs"
	else
		time_name_ext+="-lm35_off"
	fi
	if [[ "${hp_flag}" == "CPU" ]] ; then
		time_name_ext+="-hp_cpu"
	elif [[ "${hp_flag}" == "NCS" ]] ; then
		time_name_ext+="-hp_ncs"
	else
		time_name_ext+="-hp_off"
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

# 実行
execute ${MODEL_NO}

exit
