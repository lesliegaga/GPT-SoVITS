#!/bin/bash

# ==================== 配置参数 ====================
# 基础路径配置
INPUT_AUDIO="/home/tongyan.zjy/workspace/git/GPT-SoVITS/input_audio/test"
WORK_DIR="/home/tongyan.zjy/workspace/git/GPT-SoVITS/test"
EXP_NAME="my_speaker"
EXP_DIR="$WORK_DIR/experiments/$EXP_NAME"

# 数据处理路径
SLICED_DIR="$WORK_DIR/sliced"
DENOISED_DIR="$WORK_DIR/denoised"
ASR_OUTPUT="$WORK_DIR/transcripts"

# 预训练模型路径 - 从config.py中读取，而不是硬编码
BERT_DIR="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
CNHUBERT_DIR="GPT_SoVITS/pretrained_models/chinese-hubert-base"
PRETRAINED_SV="GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"

# 训练参数
BATCH_SIZE=16
EPOCHS_S2=50
EPOCHS_S1=15
GPU_ID="0"
LANGUAGE="zh"

# ==================== 函数定义 ====================
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1"
    exit 1
}

check_file_exists() {
    if [[ ! -f "$1" ]]; then
        log_error "文件不存在: $1"
    fi
}

check_dir_exists() {
    if [[ ! -d "$1" ]]; then
        log_error "目录不存在: $1"
    fi
}

# ==================== 环境检查 ====================
log_info "检查环境和依赖文件..."
check_dir_exists "$INPUT_AUDIO"

# ==================== 创建工作目录 ====================
log_info "创建工作目录..."
mkdir -p "$WORK_DIR" "$SLICED_DIR" "$DENOISED_DIR" "$EXP_DIR"

# ==================== 数据预处理阶段 ====================
log_info "==================== 开始数据预处理 ===================="

# 步骤0: 音频格式转换(如果需要)
log_info "步骤0/9: 检查音频格式并转换..."
if find "$INPUT_AUDIO" -name "*.m4a" -o -name "*.mp3" -o -name "*.aac" -o -name "*.flac" | grep -q .; then
    log_info "发现非WAV格式音频文件，开始转换..."
    CONVERTED_DIR="$WORK_DIR/converted_wav"
    mkdir -p "$CONVERTED_DIR"

    # 转换所有非WAV格式到WAV
    for file in "$INPUT_AUDIO"/*.{m4a,mp3,aac,flac}; do
        if [[ -f "$file" ]]; then
            filename=$(basename "$file")
            name_without_ext="${filename%.*}"
            log_info "转换文件: $filename -> ${name_without_ext}.wav"
            ffmpeg -i "$file" -ar 32000 -ac 1 "$CONVERTED_DIR/${name_without_ext}.wav" -y
            if [[ $? -ne 0 ]]; then
                log_error "转换失败: $file"
            fi
        fi
    done

    # 同时复制已存在的WAV文件
    if ls "$INPUT_AUDIO"/*.wav 1> /dev/null 2>&1; then
        cp "$INPUT_AUDIO"/*.wav "$CONVERTED_DIR/"
    fi

    # 更新输入目录为转换后的目录
    INPUT_AUDIO="$CONVERTED_DIR"
    log_info "音频格式转换完成，使用目录: $INPUT_AUDIO"
else
    log_info "音频文件格式检查完成，无需转换"
fi

# 步骤1: 音频切片
log_info "步骤1/9: 音频切片..."
python tools/slice_audio.py \
    "$INPUT_AUDIO" \
    "$SLICED_DIR" \
    -34 4000 300 10 500 0.9 0.25 0 1
if [[ $? -ne 0 ]]; then
    log_error "音频切片失败"
fi

# 步骤2: 降噪(可选)
log_info "步骤2/9: 音频降噪..."
python tools/cmd-denoise.py \
    -i "$SLICED_DIR" \
    -o "$DENOISED_DIR" \
    -p "float16"
if [[ $? -ne 0 ]]; then
    log_error "音频降噪失败"
fi

# 步骤3: ASR
log_info "步骤3/9: 语音识别..."
if [[ "$LANGUAGE" == "zh" ]]; then
    # 中文使用FunASR
    python tools/asr/funasr_asr.py \
        -i "$DENOISED_DIR" \
        -o "$ASR_OUTPUT"
else
    # 其他语言使用Faster-Whisper
    python tools/asr/fasterwhisper_asr.py \
        -i "$DENOISED_DIR" \
        -o "$ASR_OUTPUT" \
        -l "$LANGUAGE" \
        -p "float16"
fi

if [[ $? -ne 0 ]]; then
    log_error "语音识别失败"
fi

# 检查ASR输出文件的实际位置
if [[ -d "$ASR_OUTPUT" ]]; then
    # ASR工具创建了目录，查找实际的.list文件
    ACTUAL_LIST_FILE=$(find "$ASR_OUTPUT" -name "*.list" | head -1)
    if [[ -f "$ACTUAL_LIST_FILE" ]]; then
        log_info "发现ASR输出文件: $ACTUAL_LIST_FILE"
        ASR_OUTPUT="$ACTUAL_LIST_FILE"
    else
        log_error "未找到ASR输出的.list文件"
    fi
elif [[ ! -f "$ASR_OUTPUT" ]]; then
    # 检查是否在当前目录或其他位置生成了文件
    POSSIBLE_OUTPUT="${ASR_OUTPUT%.*}.list"
    if [[ -f "$POSSIBLE_OUTPUT" ]]; then
        ASR_OUTPUT="$POSSIBLE_OUTPUT"
        log_info "发现ASR输出文件: $ASR_OUTPUT"
    else
        log_error "未找到ASR输出文件: $ASR_OUTPUT"
    fi
fi

log_info "数据预处理完成，请检查并校对转录文件: $ASR_OUTPUT"
echo "按Enter继续，或Ctrl+C退出以手动校对转录..."
if [[ -t 0 ]]; then
    read
else
    log_info "非交互模式，跳过手动校对确认"
fi

# ==================== 特征提取阶段 ====================
log_info "==================== 开始特征提取 ===================="

# 设置环境变量
export inp_text="$ASR_OUTPUT"
export inp_wav_dir="$DENOISED_DIR"
export exp_name="$EXP_NAME"
export opt_dir="$EXP_DIR"
export bert_pretrained_dir="$BERT_DIR"
export cnhubert_base_dir="$CNHUBERT_DIR"
export sv_path="$PRETRAINED_SV"
export s2config_path="GPT_SoVITS/configs/s2v2ProPlus.json"
export is_half="True"
export BATCH_SIZE="$BATCH_SIZE"
export EPOCHS_S2="$EPOCHS_S2"
# 提前设置S2_VERSION，确保3-get-semantic.py能够正确读取版本信息
export S2_VERSION="v2ProPlus"

### 并行分片设置（按 GPU_ID 形如 "0" 或 "0-1-2" 解析）
gpu_names=(${GPU_ID//-/ })
all_parts=${#gpu_names[@]}
if [[ $all_parts -eq 0 ]]; then
    all_parts=1
    gpu_names=("0")
fi

# 步骤4: 文本分词与BERT特征提取（并行）
log_info "步骤4/9: 文本分词与BERT特征提取（并行 $all_parts 份）..."
pids=()
for idx in "${!gpu_names[@]}"; do
    i_part="$idx"
    _CUDA_VISIBLE_DEVICES_PART="${gpu_names[$idx]}"
    log_info "启动1-get-text分片 i_part=$i_part 总份数=$all_parts 绑定GPU=$_CUDA_VISIBLE_DEVICES_PART"
    i_part="$i_part" all_parts="$all_parts" _CUDA_VISIBLE_DEVICES="$_CUDA_VISIBLE_DEVICES_PART" \
    python GPT_SoVITS/prepare_datasets/1-get-text.py &
    pids+=("$!")
done

for pid in "${pids[@]}"; do
    wait "$pid" || log_error "文本特征提取分片任务失败 (pid=$pid)"
done

# 合并 2-name2text-*.txt -> 2-name2text.txt
MERGED_TXT="$EXP_DIR/2-name2text.txt"
> "$MERGED_TXT"
for idx in "${!gpu_names[@]}"; do
    PART_TXT="$EXP_DIR/2-name2text-$idx.txt"
    if [[ -f "$PART_TXT" ]]; then
        cat "$PART_TXT" >> "$MERGED_TXT"
        rm -f "$PART_TXT"
    else
        log_error "缺少分片文本文件: $PART_TXT"
    fi
done

# 步骤5: 音频特征提取（并行）
log_info "步骤5/9: 音频特征提取（并行 $all_parts 份）..."
pids=()
for idx in "${!gpu_names[@]}"; do
    i_part="$idx"
    _CUDA_VISIBLE_DEVICES_PART="${gpu_names[$idx]}"
    log_info "启动2-get-hubert-wav32k分片 i_part=$i_part 总份数=$all_parts 绑定GPU=$_CUDA_VISIBLE_DEVICES_PART"
    i_part="$i_part" all_parts="$all_parts" _CUDA_VISIBLE_DEVICES="$_CUDA_VISIBLE_DEVICES_PART" \
    python GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py &
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    wait "$pid" || log_error "音频特征提取分片任务失败 (pid=$pid)"
done

# 步骤5.5: 说话人向量提取（并行）
log_info "步骤5.5/9: 说话人向量提取（并行 $all_parts 份）..."
pids=()
for idx in "${!gpu_names[@]}"; do
    i_part="$idx"
    _CUDA_VISIBLE_DEVICES_PART="${gpu_names[$idx]}"
    log_info "启动2-get-sv分片 i_part=$i_part 总份数=$all_parts 绑定GPU=$_CUDA_VISIBLE_DEVICES_PART"
    i_part="$i_part" all_parts="$all_parts" _CUDA_VISIBLE_DEVICES="$_CUDA_VISIBLE_DEVICES_PART" \
    python GPT_SoVITS/prepare_datasets/2-get-sv.py &
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    wait "$pid" || log_error "说话人向量提取分片任务失败 (pid=$pid)"
done

# 步骤6: 语义特征提取（并行）
log_info "步骤6/9: 语义特征提取（并行 $all_parts 份）..."
pids=()
for idx in "${!gpu_names[@]}"; do
    i_part="$idx"
    _CUDA_VISIBLE_DEVICES_PART="${gpu_names[$idx]}"
    log_info "启动3-get-semantic分片 i_part=$i_part 总份数=$all_parts 绑定GPU=$_CUDA_VISIBLE_DEVICES_PART"
    i_part="$i_part" all_parts="$all_parts" _CUDA_VISIBLE_DEVICES="$_CUDA_VISIBLE_DEVICES_PART" \
    python GPT_SoVITS/prepare_datasets/3-get-semantic.py &
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    wait "$pid" || log_error "语义特征提取分片任务失败 (pid=$pid)"
done

# 合并 6-name2semantic-*.tsv -> 6-name2semantic.tsv（含表头）
MERGED_SEM="$EXP_DIR/6-name2semantic.tsv"
echo -e "item_name\tsemantic_audio" > "$MERGED_SEM"
for idx in "${!gpu_names[@]}"; do
    PART_SEM="$EXP_DIR/6-name2semantic-$idx.tsv"
    if [[ -f "$PART_SEM" ]]; then
        cat "$PART_SEM" >> "$MERGED_SEM"
        rm -f "$PART_SEM"
    else
        log_error "缺少分片语义文件: $PART_SEM"
    fi
done

# ==================== 模型训练阶段 ====================
log_info "==================== 开始模型训练 ===================="

# 步骤7: SoVITS模型训练
log_info "步骤7/9: SoVITS模型训练..."

# 创建SoVITS训练配置
python generate_s2_config.py "$WORK_DIR/config_s2.json"

# 执行SoVITS训练
python GPT_SoVITS/s2_train.py --config "$WORK_DIR/config_s2.json"
if [[ $? -ne 0 ]]; then
    log_error "SoVITS训练失败"
fi

# 步骤8: GPT模型训练
log_info "步骤8/9: GPT模型训练..."

# 设置GPT训练所需的环境变量
export hz="25hz"

# 创建GPT训练配置
python generate_s1_config.py "$WORK_DIR/config_s1.yaml"

# 执行GPT训练
python GPT_SoVITS/s1_train.py --config_file "$WORK_DIR/config_s1.yaml"
if [[ $? -ne 0 ]]; then
    log_error "GPT训练失败"
fi

# ==================== 推理测试 ====================
log_info "==================== 步骤9/9: 推理测试准备 ===================="

# 查找训练好的模型
# 注意：根据webui.py的配置，GPT模型保存在logs_s1_版本号目录下，SoVITS模型保存在logs_s2_版本号目录下
# GPT模型文件保存在ckpt子目录中，SoVITS模型文件直接在logs_s2_版本号目录下
GPT_MODEL=$(ls $EXP_DIR/logs_s1_${S2_VERSION}/ckpt/*-e*.ckpt | tail -1)
SOVITS_MODEL=$(ls $EXP_DIR/logs_s2_${S2_VERSION}/G_*.pth | tail -1)

if [[ -z "$GPT_MODEL" ]] || [[ -z "$SOVITS_MODEL" ]]; then
    log_error "找不到训练好的模型文件"
fi

log_info "找到模型文件:"
log_info "GPT模型: $GPT_MODEL"
log_info "SoVITS模型: $SOVITS_MODEL"

# 自动选择参考音频和文本
log_info "自动选择参考音频和文本..."

# 创建输出目录
mkdir -p "$WORK_DIR/output"

# 从训练集中选择一个合适的参考音频（选择第一个音频文件）
REF_AUDIO=$(find "$DENOISED_DIR" -name "*.wav" | head -1)
if [[ -z "$REF_AUDIO" ]]; then
    log_error "在训练集中找不到音频文件"
fi

# 获取音频文件名（不含路径）
AUDIO_FILENAME=$(basename "$REF_AUDIO")
log_info "选择参考音频: $AUDIO_FILENAME"

# 从 .list 文件中提取对应的文本内容
if [[ -f "$ASR_OUTPUT" ]]; then
    # 查找包含该音频文件名的行
    REF_TEXT_LINE=$(grep "$AUDIO_FILENAME" "$ASR_OUTPUT" | head -1)
    if [[ -n "$REF_TEXT_LINE" ]]; then
        # 解析 .list 文件格式：file_path|output_file_name|language|text
        # 文本内容在第4个字段（用 | 分隔）
        REF_TEXT_CONTENT=$(echo "$REF_TEXT_LINE" | cut -d'|' -f4)
        if [[ -z "$REF_TEXT_CONTENT" ]]; then
            # 如果第4个字段为空，尝试其他分隔符
            REF_TEXT_CONTENT=$(echo "$REF_TEXT_LINE" | cut -d'\t' -f2-)
        fi
        if [[ -z "$REF_TEXT_CONTENT" ]]; then
            # 如果还是没有，使用整行作为文本
            REF_TEXT_CONTENT="$REF_TEXT_LINE"
        fi
        log_info "从转录文件提取文本: $REF_TEXT_CONTENT"
    else
        log_error "在转录文件中找不到音频 $AUDIO_FILENAME 对应的文本"
    fi
else
    log_error "转录文件不存在: $ASR_OUTPUT"
fi

# 创建参考文本文件
REF_TEXT_FILE="$WORK_DIR/ref_text.txt"
echo "$REF_TEXT_CONTENT" > "$REF_TEXT_FILE"
log_info "创建参考文本文件: $REF_TEXT_FILE"
log_info "参考文本内容: $REF_TEXT_CONTENT"

# 创建目标文本文件（示例文本）
TARGET_TEXT_FILE="$WORK_DIR/target_text.txt"
cat > "$TARGET_TEXT_FILE" << 'TARGET_EOF'
这是一个测试文本，用于验证训练好的模型效果。
请根据实际需要修改此文件中的内容。
TARGET_EOF
log_info "创建目标文本文件: $TARGET_TEXT_FILE"

# 创建推理测试脚本
cat > "$WORK_DIR/test_inference.sh" << EOF
#!/bin/bash

# 使用训练好的模型进行推理测试
# 参考音频和文本已自动选择

echo "使用以下文件进行推理测试："
echo "参考音频: $REF_AUDIO"
echo "参考文本: $REF_TEXT_FILE"
echo "目标文本: $TARGET_TEXT_FILE"
echo ""

# 执行推理
python GPT_SoVITS/inference_cli.py \\
    --gpt_model "$GPT_MODEL" \\
    --sovits_model "$SOVITS_MODEL" \\
    --ref_audio "$REF_AUDIO" \\
    --ref_text "$REF_TEXT_FILE" \\
    --ref_language "中文" \\
    --target_text "$TARGET_TEXT_FILE" \\
    --target_language "中文" \\
    --output_path "$WORK_DIR/output" \\
    --bert_path "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large" \\
    --cnhubert_base_path "GPT_SoVITS/pretrained_models/chinese-hubert-base" \\
    --gpu_number "0" \\
    --is_half

echo ""
echo "推理完成！输出文件保存在: $WORK_DIR/output/output.wav"
EOF

chmod +x "$WORK_DIR/test_inference.sh"

# ==================== 完成 ====================
log_info "==================== 训练完成 ===================="
log_info "训练结果保存在: $EXP_DIR"
log_info "推理测试脚本: $WORK_DIR/test_inference.sh"
log_info "可以使用以下命令进行推理:"
echo ""
echo "python GPT_SoVITS/inference_cli.py \\"
echo "    --gpt_model \"$GPT_MODEL\" \\"
echo "    --sovits_model \"$SOVITS_MODEL\" \\"
echo "    --ref_audio \"$REF_AUDIO\" \\"
echo "    --ref_text \"$REF_TEXT_FILE\" \\"
echo "    --ref_language \"中文\" \\"
echo "    --target_text \"$TARGET_TEXT_FILE\" \\"
echo "    --target_language \"中文\" \\"
echo "    --output_path \"$WORK_DIR/output\" \\"
echo "    --bert_path \"GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large\" \\"
echo "    --cnhubert_base_path \"GPT_SoVITS/pretrained_models/chinese-hubert-base\" \\"
echo "    --gpu_number \"0\" \\"
echo "    --is_half"
echo ""
echo "或者直接运行自动生成的测试脚本:"
echo "bash \"$WORK_DIR/test_inference.sh\""
echo ""
log_info "训练完成！"