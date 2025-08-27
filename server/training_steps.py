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
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

# 获取日志记录器（继承父级配置）
logger = logging.getLogger(__name__)

# 确保日志级别为INFO，以便显示调试信息
logger.setLevel(logging.INFO)

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
            
            logger.info("开始执行推理命令:")
            logger.info(f"命令: {' '.join(cmd)}")
            logger.info(f"工作目录: {self.base_dir}")
            
            # 验证关键文件存在
            for key in ['gpt_model', 'sovits_model', 'ref_audio', 'ref_text', 'target_text']:
                file_path = kwargs.get(key, '')
                if file_path and not os.path.exists(file_path):
                    logger.error(f"关键文件不存在: {key} = {file_path}")
                    return False
            
            # 确保子进程继承父进程的环境变量，特别是CUDA_VISIBLE_DEVICES
            env = os.environ.copy()
            logger.info(f"环境变量 CUDA_VISIBLE_DEVICES: {env.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self.base_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            
            stdout, stderr = await process.communicate()
            
            stdout_text = stdout.decode('utf-8') if stdout else ""
            stderr_text = stderr.decode('utf-8') if stderr else ""
            
            logger.info(f"推理进程返回码: {process.returncode}")
            if stdout_text:
                logger.info(f"推理输出:\n{stdout_text}")
            if stderr_text:
                logger.warning(f"推理错误输出:\n{stderr_text}")
            
            if process.returncode == 0:
                # 验证输出文件是否生成
                output_path = kwargs.get('output_path', '')
                if output_path:
                    output_file = os.path.join(output_path, 'output.wav')
                    if os.path.exists(output_file):
                        file_size = os.path.getsize(output_file)
                        logger.info(f"✅ 推理测试完成，输出文件: {output_file}, 大小: {file_size} bytes")
                        return True
                    else:
                        logger.error(f"❌ 推理命令成功但未生成输出文件: {output_file}")
                        return False
                else:
                    logger.info("✅ 推理测试完成")
                    return True
            else:
                logger.error(f"❌ 推理测试失败，返回码: {process.returncode}")
                logger.error(f"错误输出: {stderr_text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ 推理测试异常: {str(e)}")
            import traceback
            logger.error(f"异常堆栈: {traceback.format_exc()}")
            return False

class ConfigGenerator:
    """配置文件生成器"""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
    
    def generate_s2_config(self, task_config: Dict[str, Any], output_path: str) -> bool:
        """生成SoVITS训练配置，与generate_s2_config.py逻辑完全一致"""
        try:
            import sys
            
            # 添加项目根路径到Python路径，以便导入config模块
            if str(self.base_dir) not in sys.path:
                sys.path.insert(0, str(self.base_dir))
            
            try:
                from config import SoVITS_weight_version2root, exp_root
            except ImportError:
                # 如果无法导入，使用默认值
                logger.warning("无法导入config模块，使用默认配置")
                SoVITS_weight_version2root = {
                    "v1": "SoVITS_weights/",
                    "v2": "SoVITS_weights_v2/",
                    "v2Pro": "SoVITS_weights_v2Pro/",
                    "v2ProPlus": "SoVITS_weights_v2ProPlus/"
                }
            
            # 按照webui.py的逻辑选择基础配置文件
            version = task_config.get("VERSION", "v2ProPlus")
            config_file = (
                self.base_dir / "GPT_SoVITS/configs/s2.json"
                if version not in {"v2Pro", "v2ProPlus"}
                else self.base_dir / f"GPT_SoVITS/configs/s2{version}.json"
            )
            
            # 加载基础配置文件，与webui.py完全一致
            with open(config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 设置实验目录路径
            s2_dir = task_config["EXP_DIR"]
            
            # 按照webui.py的逻辑处理is_half
            batch_size = task_config["BATCH_SIZE"]
            is_half = task_config.get("IS_HALF", True)
            if is_half == False:
                data["train"]["fp16_run"] = False
                batch_size = max(1, batch_size // 2)
            
            # 设置训练参数（与generate_s2_config.py完全一致）
            gpu_numbers = task_config.get("GPU_ID", "0")
            data["train"]["batch_size"] = batch_size
            data["train"]["epochs"] = task_config["EPOCHS_S2"]
            data["train"]["text_low_lr_rate"] = task_config.get("TEXT_LOW_LR_RATE", 0.4)
            # 如果没有指定pretrained_s2G，从config.py中读取（与webui.py逻辑一致）
            if not task_config.get("pretrained_s2G") or task_config.get("pretrained_s2G") == "":
                from config import pretrained_sovits_name
                if version in pretrained_sovits_name and pretrained_sovits_name[version]:
                    pretrained_s2G = pretrained_sovits_name[version]
                    logger.info(f"从config.py读取{version}版本的SoVITS-G预训练模型: {pretrained_s2G}")
                else:
                    raise ValueError(f"警告: 在config.py中找不到{version}版本的SoVITS-D预训练模型")

            else:
                pretrained_s2G = task_config["pretrained_s2G"]
            
            # 如果没有指定pretrained_s2D，从config.py中读取（与webui.py逻辑一致）
            if not task_config.get("pretrained_s2D") or task_config.get("pretrained_s2D") == "":
                from config import pretrained_sovits_name
                if version in pretrained_sovits_name and pretrained_sovits_name[version]:
                    pretrained_s2D = pretrained_sovits_name[version].replace("s2G", "s2D")
                    logger.info(f"从config.py读取{version}版本的SoVITS-D预训练模型: {pretrained_s2D}")
                else:
                    raise ValueError(f"警告: 在config.py中找不到{version}版本的SoVITS-D预训练模型")
            else:
                pretrained_s2D = task_config["pretrained_s2D"]
            
            data["train"]["pretrained_s2G"] = pretrained_s2G
            data["train"]["pretrained_s2D"] = pretrained_s2D
            data["train"]["if_save_latest"] = task_config.get("IF_SAVE_LATEST", True)
            data["train"]["if_save_every_weights"] = task_config.get("IF_SAVE_EVERY_WEIGHTS", True)
            data["train"]["save_every_epoch"] = task_config.get("SAVE_EVERY_EPOCH_S2", 4)
            data["train"]["gpu_numbers"] = gpu_numbers  # 关键：添加缺失的gpu_numbers字段
            data["train"]["grad_ckpt"] = task_config.get("IF_GRAD_CKPT", False)
            data["train"]["lora_rank"] = task_config.get("LORA_RANK", 32)
            data["model"]["version"] = version
            data["data"]["exp_dir"] = data["s2_ckpt_dir"] = s2_dir
            data["save_weight_dir"] = SoVITS_weight_version2root.get(version, "SoVITS_weights_v2ProPlus/")  # 添加权重保存目录
            data["name"] = task_config["EXP_NAME"]
            data["version"] = version
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"S2配置文件生成成功: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"S2配置文件生成失败: {str(e)}")
            return False
    
    def generate_s1_config(self, task_config: Dict[str, Any], output_path: str) -> bool:
        """生成GPT训练配置，与generate_s1_config.py逻辑完全一致"""
        try:
            import yaml
            import sys
            
            # 添加项目根路径到Python路径，以便导入config模块
            if str(self.base_dir) not in sys.path:
                sys.path.insert(0, str(self.base_dir))
            
            try:
                from config import GPT_weight_version2root, exp_root
            except ImportError:
                # 如果无法导入，使用默认值
                logger.warning("无法导入config模块，使用默认配置")
                GPT_weight_version2root = {
                    "v1": "GPT_weights/",
                    "v2": "GPT_weights_v2/",
                    "v2ProPlus": "GPT_weights_v2ProPlus/"
                }
            
            # 按照webui.py的逻辑选择基础配置文件
            version = task_config.get("VERSION", "v2ProPlus")
            config_file = (
                self.base_dir / "GPT_SoVITS/configs/s1longer.yaml" 
                if version == "v1" 
                else self.base_dir / "GPT_SoVITS/configs/s1longer-v2.yaml"
            )
            
            # 加载基础配置文件，与webui.py完全一致
            with open(config_file) as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
            
            # 设置实验目录路径
            s1_dir = task_config["EXP_DIR"]
            
            # 按照webui.py的逻辑处理is_half
            batch_size = task_config["BATCH_SIZE"]
            is_half = task_config.get("IS_HALF", True)
            if is_half == False:
                data["train"]["precision"] = "32"
                batch_size = max(1, batch_size // 2)
            
            # 设置训练参数（与generate_s1_config.py完全一致）
            data["train"]["batch_size"] = batch_size
            data["train"]["epochs"] = task_config["EPOCHS_S1"]
            
            # 如果没有指定pretrained_s1，从config.py中读取（与webui.py逻辑一致）
            if not task_config.get("pretrained_s1") or task_config.get("pretrained_s1") == "":
                from config import pretrained_gpt_name
                if version in pretrained_gpt_name and pretrained_gpt_name[version]:
                    pretrained_s1 = pretrained_gpt_name[version]
                    logger.info(f"从config.py读取{version}版本的GPT预训练模型: {pretrained_s1}")
                else:
                    raise ValueError(f"警告: 在config.py中找不到{version}版本的GPT预训练模型")
            else:
                pretrained_s1 = task_config["pretrained_s1"]
            
            data["pretrained_s1"] = pretrained_s1
            data["train"]["save_every_n_epoch"] = task_config.get("SAVE_EVERY_EPOCH_S1", 5)
            data["train"]["if_save_every_weights"] = task_config.get("IF_SAVE_EVERY_WEIGHTS", True)
            data["train"]["if_save_latest"] = task_config.get("IF_SAVE_LATEST", True)
            data["train"]["if_dpo"] = task_config.get("IF_DPO", False)
            data["train"]["half_weights_save_dir"] = GPT_weight_version2root.get(version, "GPT_weights_v2ProPlus/")  # 关键：添加缺失的字段
            data["train"]["exp_name"] = task_config["EXP_NAME"]
            data["train_semantic_path"] = "%s/6-name2semantic.tsv" % s1_dir
            data["train_phoneme_path"] = "%s/2-name2text.txt" % s1_dir
            data["output_dir"] = "%s/logs_s1_%s" % (s1_dir, version)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"S1配置文件生成成功: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"S1配置文件生成失败: {str(e)}")
            return False
