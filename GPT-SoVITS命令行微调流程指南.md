# GPT-SoVITS 命令行微调流程指南

本文档详细说明如何使用命令行工具复现WebUI的finetune流程，实现完全的命令行化训练。

## 概述

WebUI的finetune流程包括以下步骤：
1. 填入音频路径
2. 音频切片
3. 降噪(可选)
4. ASR(自动语音识别)
5. 校对ASR转录
6. 模型训练

## 环境准备

确保已经按照官方文档安装好GPT-SoVITS环境，并激活了conda环境：

```bash
conda activate GPTSoVits
```

## 数据集格式要求

训练前需要准备符合以下格式的数据集文件(.list):
```
vocal_path|speaker_name|language|text
```

语言代码：
- 'zh': 中文
- 'ja': 日语  
- 'en': 英语
- 'ko': 韩语
- 'yue': 粤语

示例：
```
/path/to/audio1.wav|speaker1|zh|这是一段中文语音
/path/to/audio2.wav|speaker1|en|This is English speech
```

## 详细步骤

### 步骤1: 准备音频文件

#### 1.1 音频格式支持

GPT-SoVITS使用ffmpeg处理音频文件，理论上支持大多数音频格式，包括：
- ✅ **WAV** (推荐格式)
- ✅ **M4A** (需要ffmpeg支持)
- ✅ **MP3** (需要ffmpeg支持)
- ✅ **FLAC** (需要ffmpeg支持)
- ✅ **AAC** (需要ffmpeg支持)

#### 1.2 处理M4A格式音频

**方法一：直接使用M4A文件**
```bash
mkdir -p input_audio
# 直接将M4A文件放在这个目录中
cp your_audio_files/*.m4a input_audio/
```

**方法二：预先转换为WAV格式（推荐）**
```bash
mkdir -p input_audio wav_converted

# 批量转换M4A到WAV
for file in your_audio_files/*.m4a; do
    filename=$(basename "$file" .m4a)
    ffmpeg -i "$file" -ar 32000 -ac 1 "wav_converted/${filename}.wav"
done

# 使用转换后的WAV文件
cp wav_converted/*.wav input_audio/
```

**方法三：使用单文件转换**
```bash
# 转换单个M4A文件
ffmpeg -i input.m4a -ar 32000 -ac 1 output.wav

# 参数说明：
# -ar 32000: 设置采样率为32kHz (GPT-SoVITS的目标采样率)
# -ac 1: 转换为单声道
```

#### 1.3 音频质量建议

- **采样率**: 建议16kHz或以上，最终会重采样到32kHz
- **声道**: 单声道或立体声均可，处理时会转为单声道
- **音频质量**: 建议使用无损或高质量压缩格式
- **文件大小**: 单个文件建议不超过几分钟，便于后续切片处理

### 步骤2: 音频切片

使用音频切片工具将长音频切分为短片段：

```bash
python tools/slice_audio.py \
    "/path/to/input_audio" \
    "/path/to/output_sliced" \
    -34 \
    4000 \
    300 \
    10 \
    500 \
    0.9 \
    0.25 \
    0 \
    1
```

参数说明：
- `输入路径`: 音频文件或文件夹路径
- `输出路径`: 切分后音频的输出目录
- `threshold (-34)`: 音量阈值，小于此值视作静音的备选切割点
- `min_length (4000)`: 每段最小长度(ms)
- `min_interval (300)`: 最短切割间隔(ms)
- `hop_size (10)`: 音量曲线计算步长
- `max_sil_kept (500)`: 切分后静音最大保留长度(ms)
- `_max (0.9)`: 音频最大音量
- `alpha (0.25)`: 音频归一化参数
- `i_part (0)`: 分片处理的当前部分
- `all_part (1)`: 总分片数

### 步骤3: 降噪(可选)

如果音频质量不佳，可以使用降噪工具：

```bash
python tools/cmd-denoise.py \
    -i "/path/to/input_audio_folder" \
    -o "/path/to/denoised_output" \
    -p "float16"
```

参数说明：
- `-i, --input_folder`: 包含WAV文件的输入文件夹路径
- `-o, --output_folder`: 降噪后音频的输出文件夹路径  
- `-p, --precision`: 精度选择 ("float16" 或 "float32")

### 步骤4: ASR(自动语音识别)

根据语言选择对应的ASR工具：

#### 中文ASR (使用FunASR)

```bash
python tools/asr/funasr_asr.py \
    -i "/path/to/audio_folder" \
    -o "/path/to/output.list"
```

#### 其他语言ASR (使用Faster-Whisper)

```bash
python tools/asr/fasterwhisper_asr.py \
    -i "/path/to/audio_folder" \
    -o "/path/to/output.list" \
    -l "en" \
    -p "float16" \
    -s "large-v3"
```

参数说明：
- `-i, --input`: 输入音频文件夹路径
- `-o, --output`: 输出.list文件路径
- `-l, --language`: 语言代码 (en, ja, ko等，或auto自动检测)
- `-p, --precision`: 精度 ("float16", "float32", "int8")
- `-s, --model_size`: 模型大小 ("large-v3", "medium", "small"等)

支持的语言代码包括：
- en (英语), ja (日语), ko (韩语), zh (中文), yue (粤语)
- 以及其他40+种语言，详见源码中的language_code_list

### 步骤5: 校对转录

手动检查和修正ASR生成的.list文件中的文本转录，确保准确性。

### 步骤6: 训练数据预处理

在模型训练前，需要将ASR结果处理成训练所需的特征数据。这个步骤分为三个子任务：

#### 6.1 文本分词与BERT特征提取

```bash
# 设置环境变量
export inp_text="/path/to/transcripts.list"
export inp_wav_dir="/path/to/sliced_audio_folder"  
export exp_name="my_experiment"
export opt_dir="/path/to/experiments/my_experiment"
export bert_pretrained_dir="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
export i_part="0"
export all_parts="1"
export _CUDA_VISIBLE_DEVICES="0"
export is_half="True"

# 执行文本预处理
python GPT_SoVITS/prepare_datasets/1-get-text.py
```

**输入**: ASR生成的.list文件 (格式: `wav_path|speaker_name|language|text`)
**输出**: 
- `$opt_dir/2-name2text.txt`: 音素数据文件
- `$opt_dir/3-bert/`: BERT特征文件(.pt格式)

#### 6.2 音频特征提取

```bash
# 设置环境变量(继承上面的设置)
export cnhubert_base_path="GPT_SoVITS/pretrained_models/chinese-hubert-base"

# 执行音频特征提取
python GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py
```

**输入**: 切片后的音频文件
**输出**:
- `$opt_dir/4-cnhubert/`: 音频特征文件(.pt格式)
- `$opt_dir/5-wav32k/`: 32kHz重采样音频文件

#### 6.3 语义特征提取

```bash
# 设置环境变量(继承上面的设置)
export pretrained_s2G="GPT_SoVITS/pretrained_models/s2G.pth"
export s2config_path="GPT_SoVITS/configs/s2.json"

# 执行语义特征提取
python GPT_SoVITS/prepare_datasets/3-get-semantic.py
```

**输入**: 音频特征文件和模型配置
**输出**:
- `$opt_dir/6-name2semantic.tsv`: 语义特征文件

### 步骤7: 模型训练

训练分为两个阶段：SoVITS训练和GPT训练。

#### 7.1 训练数据目录结构

完成数据预处理后，训练目录结构如下：
```
experiments/my_experiment/
├── 2-name2text.txt          # 音素数据 (格式: name\tphones\tword2ph\tnorm_text)
├── 3-bert/                  # BERT特征文件夹
│   ├── audio1.pt
│   └── audio2.pt
├── 4-cnhubert/              # 音频特征文件夹
│   ├── audio1.pt
│   └── audio2.pt  
├── 5-wav32k/                # 32k采样率音频文件夹
│   ├── audio1.wav
│   └── audio2.wav
└── 6-name2semantic.tsv      # 语义特征 (格式: name\tsemantic_tokens)
```

#### 7.2 SoVITS模型训练

```bash
# 创建训练配置文件
cat > config_s2.json << EOF
{
  "train": {
    "log_interval": 100,
    "eval_interval": 500,
    "seed": 1234,
    "epochs": 100,
    "learning_rate": 0.0001,
    "betas": [0.8, 0.99],
    "eps": 1e-09,
    "batch_size": 32,
    "fp16_run": true,
    "lr_decay": 0.999875,
    "segment_size": 20480,
    "init_lr_ratio": 1,
    "warmup_epochs": 0,
    "c_mel": 45,
    "c_kl": 1.0,
    "text_low_lr_rate": 0.4,
    "grad_ckpt": false
  },
  "data": {
    "max_wav_value": 32768.0,
    "sampling_rate": 32000,
    "filter_length": 2048,
    "hop_length": 640,
    "win_length": 2048,
    "n_mel_channels": 128,
    "mel_fmin": 0.0,
    "mel_fmax": null,
    "add_blank": true,
    "n_speakers": 300,
    "cleaned_text": true,
    "exp_dir": "/path/to/experiments/my_experiment",
    "training_files": "/path/to/experiments/my_experiment/2-name2text.txt",
    "validation_files": "/path/to/experiments/my_experiment/2-name2text.txt"
  },
  "model": {
    "inter_channels": 192,
    "hidden_channels": 192,
    "filter_channels": 768,
    "n_heads": 2,
    "n_layers": 6,
    "kernel_size": 3,
    "p_dropout": 0.1,
    "resblock": "1",
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "upsample_rates": [10, 8, 2, 2, 2],
    "upsample_initial_channel": 512,
    "upsample_kernel_sizes": [16, 16, 8, 2, 2],
    "n_layers_q": 3,
    "use_spectral_norm": false,
    "gin_channels": 512,
    "semantic_frame_rate": "25hz",
    "freeze_quantizer": true
  },
  "s2_ckpt_dir": "/path/to/experiments/my_experiment/logs_s2",
  "content_module": "cnhubert",
  "pretrained_s2G": "GPT_SoVITS/pretrained_models/s2G.pth",
  "pretrained_s2D": "GPT_SoVITS/pretrained_models/s2D.pth",
  "name": "my_experiment",
  "version": "v2"
}
EOF

# 根据版本选择对应的训练脚本
# 对于v1, v2, v2Pro, v2ProPlus版本：
python GPT_SoVITS/s2_train.py --config config_s2.json

# 对于v3及以上版本：
# python GPT_SoVITS/s2_train_v3_lora.py --config config_s2.json
```

#### 7.3 GPT模型训练

```bash
# 创建GPT训练配置文件
cat > config_s1.yaml << EOF
model:
  t2s_model_path: /path/to/experiments/my_experiment/logs_s1/G_*.pth
  symbols_path: GPT_SoVITS/text/symbols.py

train:
  batch_size: 16
  epochs: 15
  learning_rate: 0.0001
  save_every_n_epoch: 5
  if_save_every_weights: true
  if_save_latest: false
  if_dpo: false
  half_weights_save_dir: /path/to/experiments/my_experiment/SoVITS_weights
  exp_name: my_experiment

data:
  train_semantic_path: /path/to/experiments/my_experiment/6-name2semantic.tsv
  train_phoneme_path: /path/to/experiments/my_experiment/2-name2text.txt

output_dir: /path/to/experiments/my_experiment/logs_s1
pretrained_s1: GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt
EOF

# 执行GPT训练
python GPT_SoVITS/s1_train.py --config_file config_s1.yaml
```

### 步骤8: 模型推理

训练完成后，可以使用命令行接口进行推理：

```bash
# 使用命令行推理接口
python GPT_SoVITS/inference_cli.py \
    --gpt_model "/path/to/experiments/my_experiment/GPT_weights/my_experiment-e15.ckpt" \
    --sovits_model "/path/to/experiments/my_experiment/SoVITS_weights/my_experiment_e30_s12000.pth" \
    --ref_audio "/path/to/reference_audio.wav" \
    --ref_text "/path/to/reference_text.txt" \
    --ref_language "中文" \
    --target_text "/path/to/target_text.txt" \
    --target_language "中文" \
    --output_path "/path/to/output_directory"
```

参数说明：
- `--gpt_model`: GPT模型路径
- `--sovits_model`: SoVITS模型路径  
- `--ref_audio`: 参考音频文件路径
- `--ref_text`: 参考音频对应的文本文件路径
- `--ref_language`: 参考音频的语言
- `--target_text`: 要合成的目标文本文件路径
- `--target_language`: 目标文本的语言
- `--output_path`: 输出音频的保存目录

## WebUI实现原理

通过分析webui.py代码，我们发现WebUI实际上是通过subprocess调用这些命令行工具：

1. **音频切片**: 调用`tools/slice_audio.py`
2. **降噪**: 调用`tools/cmd-denoise.py` 
3. **ASR**: 调用`tools/asr/funasr_asr.py`或`tools/asr/fasterwhisper_asr.py`
4. **SoVITS训练**: 调用`GPT_SoVITS/s2_train.py`或`GPT_SoVITS/s2_train_v3_lora.py`
5. **GPT训练**: 调用`GPT_SoVITS/s1_train.py`

## 注意事项

1. **环境变量**: 确保CUDA环境配置正确
2. **模型下载**: 首次运行需要下载预训练模型
3. **内存需求**: 训练过程需要足够的GPU内存
4. **数据质量**: 音频质量和转录准确性直接影响训练效果
5. **路径处理**: 所有路径建议使用绝对路径避免错误
6. **音频格式**: 
   - 推荐使用WAV格式以确保兼容性
   - M4A、MP3等格式需要确保ffmpeg支持对应编解码器
   - 建议预先转换为WAV格式避免潜在问题
7. **ffmpeg依赖**: 确保系统已安装ffmpeg并支持所需的音频编解码器

## 完整批处理脚本示例

为了简化整个流程，可以创建一个完整的批处理脚本：

```bash
#!/bin/bash

# ==================== 配置参数 ====================
# 基础路径配置
INPUT_AUDIO="/path/to/input_audio"
WORK_DIR="/path/to/work_directory"  
EXP_NAME="my_speaker"
EXP_DIR="$WORK_DIR/experiments/$EXP_NAME"

# 数据处理路径
SLICED_DIR="$WORK_DIR/sliced"
DENOISED_DIR="$WORK_DIR/denoised"  
ASR_OUTPUT="$WORK_DIR/transcripts.list"

# 预训练模型路径
BERT_DIR="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
CNHUBERT_DIR="GPT_SoVITS/pretrained_models/chinese-hubert-base"
PRETRAINED_S2G="GPT_SoVITS/pretrained_models/s2G.pth"
PRETRAINED_S2D="GPT_SoVITS/pretrained_models/s2D.pth"
PRETRAINED_S1="GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"

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
check_file_exists "$PRETRAINED_S2G"
check_file_exists "$PRETRAINED_S1"

# ==================== 创建工作目录 ====================
log_info "创建工作目录..."
mkdir -p "$WORK_DIR" "$SLICED_DIR" "$DENOISED_DIR" "$EXP_DIR"

# ==================== 数据预处理阶段 ====================
log_info "==================== 开始数据预处理 ===================="

# 步骤0: 音频格式转换(如果需要)
log_info "步骤0/8: 检查音频格式并转换..."
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
python tools/asr/fasterwhisper_asr.py \
    -i "$DENOISED_DIR" \
    -o "$ASR_OUTPUT" \
    -l "$LANGUAGE" \
    -p "float16"
if [[ $? -ne 0 ]]; then
    log_error "语音识别失败"
fi

log_info "数据预处理完成，请检查并校对转录文件: $ASR_OUTPUT"
echo "按Enter继续，或Ctrl+C退出以手动校对转录..."
read

# ==================== 特征提取阶段 ====================
log_info "==================== 开始特征提取 ===================="

# 设置环境变量
export inp_text="$ASR_OUTPUT"
export inp_wav_dir="$DENOISED_DIR"
export exp_name="$EXP_NAME"
export opt_dir="$EXP_DIR"
export bert_pretrained_dir="$BERT_DIR"
export cnhubert_base_path="$CNHUBERT_DIR"
export pretrained_s2G="$PRETRAINED_S2G"
export s2config_path="GPT_SoVITS/configs/s2.json"
export i_part="0"
export all_parts="1"
export _CUDA_VISIBLE_DEVICES="$GPU_ID"
export is_half="True"

# 步骤4: 文本分词与BERT特征提取
log_info "步骤4/9: 文本分词与BERT特征提取..."
python GPT_SoVITS/prepare_datasets/1-get-text.py
if [[ $? -ne 0 ]]; then
    log_error "文本特征提取失败"
fi

# 步骤5: 音频特征提取
log_info "步骤5/9: 音频特征提取..."
python GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py
if [[ $? -ne 0 ]]; then
    log_error "音频特征提取失败"
fi

# 步骤6: 语义特征提取
log_info "步骤6/9: 语义特征提取..."
python GPT_SoVITS/prepare_datasets/3-get-semantic.py
if [[ $? -ne 0 ]]; then
    log_error "语义特征提取失败"
fi

# ==================== 模型训练阶段 ====================
log_info "==================== 开始模型训练 ===================="

# 步骤7: SoVITS模型训练
log_info "步骤7/9: SoVITS模型训练..."

# 创建SoVITS训练配置
cat > "$WORK_DIR/config_s2.json" << EOF
{
  "train": {
    "log_interval": 100,
    "eval_interval": 500,
    "seed": 1234,
    "epochs": $EPOCHS_S2,
    "learning_rate": 0.0001,
    "betas": [0.8, 0.99],
    "eps": 1e-09,
    "batch_size": $BATCH_SIZE,
    "fp16_run": true,
    "lr_decay": 0.999875,
    "segment_size": 20480,
    "init_lr_ratio": 1,
    "warmup_epochs": 0,
    "c_mel": 45,
    "c_kl": 1.0,
    "text_low_lr_rate": 0.4,
    "grad_ckpt": false
  },
  "data": {
    "max_wav_value": 32768.0,
    "sampling_rate": 32000,
    "filter_length": 2048,
    "hop_length": 640,
    "win_length": 2048,
    "n_mel_channels": 128,
    "mel_fmin": 0.0,
    "mel_fmax": null,
    "add_blank": true,
    "n_speakers": 300,
    "cleaned_text": true,
    "exp_dir": "$EXP_DIR",
    "training_files": "$EXP_DIR/2-name2text.txt",
    "validation_files": "$EXP_DIR/2-name2text.txt"
  },
  "model": {
    "inter_channels": 192,
    "hidden_channels": 192,
    "filter_channels": 768,
    "n_heads": 2,
    "n_layers": 6,
    "kernel_size": 3,
    "p_dropout": 0.1,
    "resblock": "1",
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "upsample_rates": [10, 8, 2, 2, 2],
    "upsample_initial_channel": 512,
    "upsample_kernel_sizes": [16, 16, 8, 2, 2],
    "n_layers_q": 3,
    "use_spectral_norm": false,
    "gin_channels": 512,
    "semantic_frame_rate": "25hz",
    "freeze_quantizer": true
  },
  "s2_ckpt_dir": "$EXP_DIR/logs_s2",
  "content_module": "cnhubert",
  "pretrained_s2G": "$PRETRAINED_S2G",
  "pretrained_s2D": "$PRETRAINED_S2D",
  "name": "$EXP_NAME",
  "version": "v2"
}
EOF

# 执行SoVITS训练
python GPT_SoVITS/s2_train.py --config "$WORK_DIR/config_s2.json"
if [[ $? -ne 0 ]]; then
    log_error "SoVITS训练失败"
fi

# 步骤8: GPT模型训练
log_info "步骤8/9: GPT模型训练..."

# 创建GPT训练配置
cat > "$WORK_DIR/config_s1.yaml" << EOF
model:
  t2s_model_path: $EXP_DIR/logs_s1/G_*.pth
  symbols_path: GPT_SoVITS/text/symbols.py

train:
  batch_size: $BATCH_SIZE
  epochs: $EPOCHS_S1
  learning_rate: 0.0001
  save_every_n_epoch: 5
  if_save_every_weights: true
  if_save_latest: false
  if_dpo: false
  half_weights_save_dir: $EXP_DIR/SoVITS_weights
  exp_name: $EXP_NAME

data:
  train_semantic_path: $EXP_DIR/6-name2semantic.tsv
  train_phoneme_path: $EXP_DIR/2-name2text.txt

output_dir: $EXP_DIR/logs_s1
pretrained_s1: $PRETRAINED_S1
EOF

# 执行GPT训练
python GPT_SoVITS/s1_train.py --config_file "$WORK_DIR/config_s1.yaml"
if [[ $? -ne 0 ]]; then
    log_error "GPT训练失败"
fi

# ==================== 推理测试 ====================
log_info "==================== 步骤9/9: 推理测试准备 ===================="

# 查找训练好的模型
GPT_MODEL=$(ls $EXP_DIR/logs_s1/*-e*.ckpt | tail -1)
SOVITS_MODEL=$(ls $EXP_DIR/logs_s2/G_*.pth | tail -1)

if [[ -z "$GPT_MODEL" ]] || [[ -z "$SOVITS_MODEL" ]]; then
    log_error "找不到训练好的模型文件"
fi

log_info "找到模型文件:"
log_info "GPT模型: $GPT_MODEL"
log_info "SoVITS模型: $SOVITS_MODEL"

# 创建推理测试脚本
cat > "$WORK_DIR/test_inference.sh" << EOF
#!/bin/bash

# 使用训练好的模型进行推理测试
# 需要准备参考音频和文本

# 示例：
# python GPT_SoVITS/inference_cli.py \\
#     --gpt_model "$GPT_MODEL" \\
#     --sovits_model "$SOVITS_MODEL" \\
#     --ref_audio "/path/to/reference_audio.wav" \\
#     --ref_text "/path/to/reference_text.txt" \\
#     --ref_language "中文" \\
#     --target_text "/path/to/target_text.txt" \\
#     --target_language "中文" \\
#     --output_path "$WORK_DIR/output"

echo "请修改此脚本中的参考音频和文本路径，然后运行推理测试"
echo "GPT模型路径: $GPT_MODEL"
echo "SoVITS模型路径: $SOVITS_MODEL"
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
echo "    --ref_audio \"/path/to/reference_audio.wav\" \\"
echo "    --ref_text \"/path/to/reference_text.txt\" \\"
echo "    --ref_language \"中文\" \\"
echo "    --target_text \"/path/to/target_text.txt\" \\"
echo "    --target_language \"中文\" \\"
echo "    --output_path \"$WORK_DIR/output\""
echo ""
log_info "训练完成！"
```

### 使用批处理脚本

1. **修改配置参数**: 编辑脚本开头的配置部分，设置正确的路径
2. **运行脚本**: `bash complete_training.sh`
3. **检查日志**: 脚本会输出详细的执行日志
4. **校对转录**: 在ASR步骤后，脚本会暂停让你校对转录文件
5. **等待训练完成**: 训练过程可能需要数小时到数天
6. **测试推理**: 使用生成的测试脚本进行推理验证

## 故障排除

1. **CUDA内存不足**: 减少batch_size或使用CPU训练
2. **模型下载失败**: 检查网络连接，或手动下载模型
3. **音频格式问题**: 
   - **M4A文件无法处理**: 检查ffmpeg是否支持AAC编解码器
   - **音频加载失败**: 尝试预先转换为WAV格式
   - **音频质量问题**: 确保采样率16kHz以上，避免过度压缩的音频
   - **编解码器错误**: 安装完整版ffmpeg: `sudo apt install ffmpeg` (Linux) 或从官网下载完整版
4. **路径错误**: 使用绝对路径，确保所有文件存在
5. **依赖缺失**: 检查requirements.txt中的依赖是否已安装

### 音频格式问题详细解决方案

#### 问题：M4A文件处理失败
```bash
# 解决方案1: 检查ffmpeg版本和编解码器支持
ffmpeg -version
ffmpeg -formats | grep m4a
ffmpeg -codecs | grep aac

# 解决方案2: 手动转换M4A到WAV
ffmpeg -i input.m4a -ar 32000 -ac 1 output.wav

# 解决方案3: 批量转换
find . -name "*.m4a" -exec sh -c 'ffmpeg -i "$1" -ar 32000 -ac 1 "${1%.m4a}.wav"' _ {} \;
```

#### 问题：音频加载失败
```bash
# 检查音频文件是否损坏
ffmpeg -v error -i input_audio.m4a -f null -

# 尝试重新编码
ffmpeg -i input_audio.m4a -c:a aac -b:a 128k output_audio.m4a
```

#### 问题：ffmpeg不支持某些编解码器
```bash
# Ubuntu/Debian安装完整版ffmpeg
sudo apt update
sudo apt install ffmpeg

# 验证支持的格式
ffmpeg -formats | grep -i "m4a\|aac\|mp3"
```

通过以上步骤，你就可以完全使用命令行完成GPT-SoVITS的微调训练，无需依赖WebUI界面。
