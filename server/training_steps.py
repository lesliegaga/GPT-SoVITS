#!/usr/bin/env python3
"""
训练步骤处理器
包含每个训练步骤的具体实现逻辑
"""

import os
import json
import subprocess
import asyncio
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StepProcessor:
    """训练步骤处理器"""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        
    async def convert_audio(self, input_dir: str, output_dir: str, target_sr: int = 32000) -> bool:
        """音频格式转换"""
        try:
            input_path = Path(input_dir)
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 查找需要转换的音频文件
            audio_files = []
            for ext in ['*.m4a', '*.mp3', '*.aac', '*.flac']:
                audio_files.extend(input_path.glob(ext))
            
            if not audio_files:
                logger.info("未发现需要转换的音频文件")
                # 复制已存在的WAV文件
                for wav_file in input_path.glob('*.wav'):
                    shutil.copy2(wav_file, output_path / wav_file.name)
                return True
            
            # 转换音频文件
            for audio_file in audio_files:
                output_file = output_path / f"{audio_file.stem}.wav"
                cmd = [
                    'ffmpeg', '-i', str(audio_file),
                    '-ar', str(target_sr), '-ac', '1',
                    str(output_file), '-y'
                ]
                
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                
                if process.returncode != 0:
                    logger.error(f"转换失败 {audio_file}: {stderr.decode()}")
                    return False
                    
                logger.info(f"转换成功: {audio_file.name} -> {output_file.name}")
            
            # 复制已存在的WAV文件
            for wav_file in input_path.glob('*.wav'):
                shutil.copy2(wav_file, output_path / wav_file.name)
            
            return True
            
        except Exception as e:
            logger.error(f"音频转换失败: {str(e)}")
            return False
    
    async def slice_audio(self, input_dir: str, output_dir: str, **kwargs) -> bool:
        """音频切片"""
        try:
            cmd = [
                'python', str(self.base_dir / 'tools/slice_audio.py'),
                input_dir, output_dir,
                str(kwargs.get('min_length', -34)),
                str(kwargs.get('min_interval', 4000)),
                str(kwargs.get('hop_size', 300)),
                str(kwargs.get('max_sil_kept', 10)),
                str(kwargs.get('_max', 500)),
                str(kwargs.get('alpha', 0.9)),
                str(kwargs.get('n_parts', 0.25)),
                str(kwargs.get('is_speech', 0)),
                str(kwargs.get('num_processes', 1))
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self.base_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info("音频切片完成")
                return True
            else:
                logger.error(f"音频切片失败: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"音频切片异常: {str(e)}")
            return False
    
    async def denoise_audio(self, input_dir: str, output_dir: str, precision: str = "float16") -> bool:
        """音频降噪"""
        try:
            cmd = [
                'python', str(self.base_dir / 'tools/cmd-denoise.py'),
                '-i', input_dir,
                '-o', output_dir,
                '-p', precision
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self.base_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info("音频降噪完成")
                return True
            else:
                logger.error(f"音频降噪失败: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"音频降噪异常: {str(e)}")
            return False
    
    async def asr_transcribe(self, input_dir: str, output_dir: str, language: str = "zh", precision: str = "float16") -> bool:
        """语音识别"""
        try:
            if language == "zh":
                # 中文使用FunASR
                cmd = [
                    'python', str(self.base_dir / 'tools/asr/funasr_asr.py'),
                    '-i', input_dir,
                    '-o', output_dir
                ]
            else:
                # 其他语言使用Faster-Whisper
                cmd = [
                    'python', str(self.base_dir / 'tools/asr/fasterwhisper_asr.py'),
                    '-i', input_dir,
                    '-o', output_dir,
                    '-l', language,
                    '-p', precision
                ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self.base_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info("语音识别完成")
                return True
            else:
                logger.error(f"语音识别失败: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"语音识别异常: {str(e)}")
            return False
    
    async def extract_features(self, script_name: str, env_vars: Dict[str, str], 
                             parallel_parts: int = 1, gpu_ids: str = "0") -> bool:
        """特征提取（通用方法）"""
        try:
            gpu_list = gpu_ids.split('-') if '-' in gpu_ids else [gpu_ids]
            if parallel_parts > len(gpu_list):
                parallel_parts = len(gpu_list)
            
            # 并行执行
            tasks = []
            for i in range(parallel_parts):
                part_env = env_vars.copy()
                part_env.update({
                    'i_part': str(i),
                    'all_parts': str(parallel_parts),
                    '_CUDA_VISIBLE_DEVICES': gpu_list[i % len(gpu_list)]
                })
                
                cmd = ['python', str(self.base_dir / f'GPT_SoVITS/prepare_datasets/{script_name}')]
                
                task = asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=str(self.base_dir),
                    env=part_env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                tasks.append(task)
            
            # 等待所有任务完成
            processes = await asyncio.gather(*tasks)
            results = await asyncio.gather(*[p.communicate() for p in processes])
            
            # 检查结果
            for i, (process, (stdout, stderr)) in enumerate(zip(processes, results)):
                if process.returncode != 0:
                    logger.error(f"特征提取分片{i}失败: {stderr.decode()}")
                    return False
            
            logger.info(f"{script_name} 特征提取完成")
            return True
            
        except Exception as e:
            logger.error(f"特征提取异常: {str(e)}")
            return False
    
    async def merge_feature_files(self, base_dir: str, filename_pattern: str, 
                                output_file: str, has_header: bool = False, parallel_parts: int = 1):
        """合并并行处理产生的特征文件"""
        try:
            output_path = Path(base_dir) / output_file
            
            with open(output_path, 'w', encoding='utf-8') as outf:
                if has_header:
                    # 对于TSV文件，写入表头
                    if output_file.endswith('.tsv'):
                        outf.write("item_name\tsemantic_audio\n")
                
                for i in range(parallel_parts):
                    part_file = Path(base_dir) / filename_pattern.format(i)
                    if part_file.exists():
                        with open(part_file, 'r', encoding='utf-8') as inf:
                            content = inf.read()
                            outf.write(content)
                        # 删除分片文件
                        part_file.unlink()
                    else:
                        logger.warning(f"分片文件不存在: {part_file}")
            
            logger.info(f"文件合并完成: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"文件合并失败: {str(e)}")
            return False
    
    async def train_model(self, script_name: str, config_file: str, env_vars: Optional[Dict[str, str]] = None) -> bool:
        """模型训练（通用方法）"""
        try:
            if script_name == "s2_train.py":
                cmd = ['python', str(self.base_dir / f'GPT_SoVITS/{script_name}'), '--config', config_file]
            elif script_name == "s1_train.py":
                cmd = ['python', str(self.base_dir / f'GPT_SoVITS/{script_name}'), '--config_file', config_file]
            else:
                raise ValueError(f"不支持的训练脚本: {script_name}")
            
            # 设置环境变量
            train_env = os.environ.copy()
            if env_vars:
                train_env.update(env_vars)
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self.base_dir),
                env=train_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info(f"{script_name} 训练完成")
                return True
            else:
                logger.error(f"{script_name} 训练失败: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"模型训练异常: {str(e)}")
            return False
    
    async def test_inference(self, **kwargs) -> bool:
        """推理测试"""
        try:
            cmd = [
                'python', str(self.base_dir / 'GPT_SoVITS/inference_cli.py'),
                '--gpt_model', kwargs.get('gpt_model', ''),
                '--sovits_model', kwargs.get('sovits_model', ''),
                '--ref_audio', kwargs.get('ref_audio', ''),
                '--ref_text', kwargs.get('ref_text', ''),
                '--ref_language', kwargs.get('ref_language', '中文'),
                '--target_text', kwargs.get('target_text', ''),
                '--target_language', kwargs.get('target_language', '中文'),
                '--output_path', kwargs.get('output_path', ''),
                '--bert_path', kwargs.get('bert_path', ''),
                '--cnhubert_base_path', kwargs.get('cnhubert_base_path', ''),
                '--gpu_number', kwargs.get('gpu_number', '0')
            ]
            
            if kwargs.get('is_half', False):
                cmd.append('--is_half')
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self.base_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info("推理测试完成")
                return True
            else:
                logger.error(f"推理测试失败: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"推理测试异常: {str(e)}")
            return False

class ConfigGenerator:
    """配置文件生成器"""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
    
    def generate_s2_config(self, task_config: Dict[str, Any], output_path: str) -> bool:
        """生成SoVITS训练配置"""
        try:
            s2_config = {
                "train": {
                    "log_interval": 100,
                    "eval_interval": 500,
                    "seed": 1234,
                    "epochs": task_config["EPOCHS_S2"],
                    "learning_rate": 0.0001,
                    "betas": [0.8, 0.99],
                    "eps": 1e-9,
                    "batch_size": task_config["BATCH_SIZE"],
                    "fp16_run": True,
                    "lr_decay": 0.999875,
                    "segment_size": 20480,
                    "init_lr_ratio": 1,
                    "warmup_epochs": 0,
                    "c_mel": 45,
                    "c_kl": 1.0,
                    "use_sr": True,
                    "max_speclen": 512,
                    "port": "8001",
                    "cache_all_data": True,
                    "cache_device": "cpu",
                    "amp_dtype": "float16"
                },
                "data": {
                    "exp_dir": task_config["EXP_DIR"],
                    "training_files": os.path.join(task_config["EXP_DIR"], "2-name2text.txt"),
                    "max_wav_value": 32768.0,
                    "sampling_rate": 32000,
                    "filter_length": 2048,
                    "hop_length": 320,
                    "win_length": 2048,
                    "n_mel_channels": 128,
                    "mel_fmin": 0.0,
                    "mel_fmax": "null",
                    "add_blank": True,
                    "n_speakers": 300,
                    "cleaned_text": True,
                    "exp_dir": task_config["EXP_DIR"]
                },
                "model": {
                    "ms_istft_vits": True,
                    "mb_istft_vits": False,
                    "istft_vits": False,
                    "subbands": 4,
                    "gen_istft_n_fft": 2048,
                    "gen_istft_hop_size": 320,
                    "gen_istft_win_size": 2048,
                    "spec_bwd_max_iter": 8,
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
                    "upsample_rates": [10, 8, 2, 2],
                    "upsample_initial_channel": 512,
                    "upsample_kernel_sizes": [16, 16, 4, 4],
                    "n_layers_q": 3,
                    "use_spectral_norm": False,
                    "gin_channels": 512,
                    "ssl_dim": 1024,
                    "n_speakers": 300,
                    "speaker_embedding": True
                },
                "s2_ckpt_dir": task_config["EXP_DIR"],
                "content_module": "cnhubert",
                "save_every_epoch": 5,
                "name": task_config["EXP_NAME"],
                "version": "v2Pro"
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(s2_config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"S2配置文件生成成功: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"S2配置文件生成失败: {str(e)}")
            return False
    
    def generate_s1_config(self, task_config: Dict[str, Any], output_path: str) -> bool:
        """生成GPT训练配置"""
        try:
            s1_config = f"""
exp_name: "{task_config['EXP_NAME']}"
exp_dir: "{task_config['EXP_DIR']}"
os_gpu: "{task_config['GPU_ID']}"
if_dpo: false
if_tts: false
model_name: "s1bert25hz-2kh-longer-epoch=68e-step=50232"
save_every_epoch: 5
if_save_latest: true
if_save_every_weights: true
half_weights_save_dir: "{task_config['EXP_DIR']}"
lr: 0.05
decay_step: [4, 8, 12]
decay: 0.5
epoch: {task_config['EPOCHS_S1']}
batch_size: {task_config['BATCH_SIZE']}
dpo_lr: 0.00001
version: "v2"
pretrained_s1: "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
text_path: "{os.path.join(task_config['EXP_DIR'], '2-name2text.txt')}"
semantic_path: "{os.path.join(task_config['EXP_DIR'], '6-name2semantic.tsv')}"
bert_pretrained_dir: "{task_config['BERT_DIR']}"
"""
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(s1_config)
            
            logger.info(f"S1配置文件生成成功: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"S1配置文件生成失败: {str(e)}")
            return False
