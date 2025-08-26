#!/usr/bin/env python3
"""
GPT-SoVITS训练服务API
提供完整的语音克隆训练流程API接口
"""

import os
import sys
import json
import uuid
import asyncio
import subprocess
import argparse
import logging
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from enum import Enum

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn

# 导入步骤处理器和配置
import os

# 根据运行环境选择导入方式
if os.getenv('GPT_SOVITS_SERVER_MODE') == 'standalone':
    # 独立运行模式（从server目录直接运行）
    from training_steps import StepProcessor, ConfigGenerator
    from service_config import get_base_path, get_work_dir, get_model_paths
else:
    # 包模式（从根目录导入）
    from .training_steps import StepProcessor, ConfigGenerator
    from .service_config import get_base_path, get_work_dir, get_model_paths

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training_service.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# 处理状态枚举
class ProcessingStatus(str, Enum):
    PENDING = "pending"      # 等待处理
    RUNNING = "running"      # 正在处理
    COMPLETED = "completed"  # 处理完成
    FAILED = "failed"        # 处理失败
    CANCELLED = "cancelled"  # 已取消

# 音频处理步骤枚举
class AudioProcessingStep(str, Enum):
    CONVERT_AUDIO = "convert_audio"
    SLICE_AUDIO = "slice_audio"
    DENOISE_AUDIO = "denoise_audio"
    ASR_TRANSCRIBE = "asr_transcribe"

# 训练步骤枚举
class TrainingStep(str, Enum):
    EXTRACT_TEXT_FEATURES = "extract_text_features"
    EXTRACT_AUDIO_FEATURES = "extract_audio_features"
    EXTRACT_SPEAKER_VECTORS = "extract_speaker_vectors"
    EXTRACT_SEMANTIC_FEATURES = "extract_semantic_features"
    TRAIN_SOVITS = "train_sovits"
    TRAIN_GPT = "train_gpt"

# 推理步骤枚举
class InferenceStep(str, Enum):
    TEST_INFERENCE = "test_inference"

# 角色配置模型
class CharacterConfig(BaseModel):
    character_name: str = Field(description="角色名称")
    language: str = Field(default="zh", description="语言设置")
    batch_size: int = Field(default=16, description="批次大小")
    epochs_s2: int = Field(default=50, description="SoVITS训练轮数")
    epochs_s1: int = Field(default=15, description="GPT训练轮数")
    gpu_id: str = Field(default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"), description="GPU设备ID")
    enable_denoise: bool = Field(default=True, description="是否启用降噪")

# 角色创建请求
class CharacterCreateRequest(BaseModel):
    character_name: str = Field(description="角色名称")
    config: CharacterConfig = Field(description="角色配置")

# 角色重命名请求
class CharacterRenameRequest(BaseModel):
    new_name: str = Field(description="新角色名称")

# 角色信息模型
class CharacterInfo(BaseModel):
    character_name: str
    config: CharacterConfig
    created_at: datetime
    updated_at: datetime
    audio_count: int = 0
    audio_processing_status: ProcessingStatus = ProcessingStatus.PENDING
    training_status: ProcessingStatus = ProcessingStatus.PENDING
    model_exists: bool = False
    is_default: bool = False

# 音频处理信息模型
class AudioProcessingInfo(BaseModel):
    character_name: str
    status: ProcessingStatus
    current_step: Optional[str] = None
    progress: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    processed_audio_count: int = 0

# 训练信息模型
class TrainingInfo(BaseModel):
    character_name: str
    status: ProcessingStatus
    current_step: Optional[str] = None
    progress: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    gpt_model_path: Optional[str] = None
    sovits_model_path: Optional[str] = None

# 推理请求模型
class InferenceRequest(BaseModel):
    character_name: Optional[str] = Field(default=None, description="角色名称，为空则使用默认角色")
    target_text: str = Field(description="目标文本")
    ref_audio: Optional[str] = Field(default=None, description="参考音频路径")
    ref_text: Optional[str] = Field(default=None, description="参考文本")
    ref_language: str = Field(default="中文", description="参考语言")
    target_language: str = Field(default="中文", description="目标语言")
    output_name: Optional[str] = Field(default=None, description="输出文件名")

# 推理信息模型
class InferenceInfo(BaseModel):
    inference_id: str
    character_name: str
    target_text: str
    status: ProcessingStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    output_path: Optional[str] = None
    error_message: Optional[str] = None

# 步骤执行请求模型
class StepExecuteRequest(BaseModel):
    params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="步骤特定参数")

# 全局数据存储
characters_db: Dict[str, CharacterInfo] = {}
audio_processing_db: Dict[str, AudioProcessingInfo] = {}
training_db: Dict[str, TrainingInfo] = {}
inference_db: Dict[str, InferenceInfo] = {}
step_processes: Dict[str, subprocess.Popen] = {}
default_character: Optional[str] = None

# FastAPI应用
app = FastAPI(
    title="GPT-SoVITS训练服务",
    description="提供完整的语音克隆训练流程API接口",
    version="1.0.0"
)

class CharacterBasedTrainingService:
    """基于角色的训练服务核心类"""
    
    def __init__(self):
        # 使用配置文件获取路径
        self.base_dir = get_base_path()
        self.work_dir = get_work_dir()
        
        self.step_processor = StepProcessor(str(self.base_dir))
        self.config_generator = ConfigGenerator(str(self.base_dir))
        
        # 创建角色目录
        self.characters_dir = self.work_dir / "characters"
        self.characters_dir.mkdir(exist_ok=True)
        
        # 创建推理输出目录
        self.inference_output_dir = self.work_dir / "inference_output"
        self.inference_output_dir.mkdir(exist_ok=True)
        
        # 加载现有角色和默认角色设置
        self._load_existing_characters()
        self._load_default_character()
        
        logger.info(f"✅ 初始化完成:")
        logger.info(f"   基础目录: {self.base_dir}")
        logger.info(f"   工作目录: {self.work_dir}")
        logger.info(f"   角色目录: {self.characters_dir}")
        logger.info(f"   推理输出目录: {self.inference_output_dir}")
    
    # ==================== 角色管理 ====================
    
    def create_character(self, character_name: str, config: CharacterConfig) -> CharacterInfo:
        """创建新角色"""
        if character_name in characters_db:
            raise ValueError(f"角色已存在: {character_name}")
        
        # 验证角色名称（避免特殊字符）
        if not character_name.replace('_', '').replace('-', '').isalnum():
            raise ValueError("角色名称只能包含字母、数字、下划线和连字符")
        
        # 创建角色目录结构
        character_dir = self.get_character_dir(character_name)
        character_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        (character_dir / "raw_audio").mkdir(exist_ok=True)
        (character_dir / "converted_audio").mkdir(exist_ok=True)
        (character_dir / "sliced_audio").mkdir(exist_ok=True)
        (character_dir / "denoised_audio").mkdir(exist_ok=True)
        (character_dir / "transcripts").mkdir(exist_ok=True)
        (character_dir / "experiments").mkdir(exist_ok=True)
        (character_dir / "models").mkdir(exist_ok=True)
        
        # 创建角色信息
        character_info = CharacterInfo(
            character_name=character_name,
            config=config,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # 保存到数据库
        characters_db[character_name] = character_info
        
        # 保存角色配置文件
        self._save_character_config(character_name, character_info)
        
        # 如果是第一个角色，设置为默认角色
        global default_character
        if default_character is None:
            default_character = character_name
            character_info.is_default = True
            self._save_default_character()
        
        logger.info(f"✅ 角色创建成功: {character_name}")
        return character_info
    
    def rename_character(self, old_name: str, new_name: str) -> CharacterInfo:
        """重命名角色"""
        if old_name not in characters_db:
            raise ValueError(f"角色不存在: {old_name}")
        
        if new_name in characters_db:
            raise ValueError(f"新角色名称已存在: {new_name}")
        
        # 验证新角色名称
        if not new_name.replace('_', '').replace('-', '').isalnum():
            raise ValueError("角色名称只能包含字母、数字、下划线和连字符")
        
        # 重命名目录
        old_dir = self.get_character_dir(old_name)
        new_dir = self.get_character_dir(new_name)
        
        if old_dir.exists():
            shutil.move(str(old_dir), str(new_dir))
        
        # 更新数据库
        character_info = characters_db[old_name]
        character_info.character_name = new_name
        character_info.config.character_name = new_name
        character_info.updated_at = datetime.now()
        
        characters_db[new_name] = character_info
        del characters_db[old_name]
        
        # 更新音频处理和训练信息
        if old_name in audio_processing_db:
            audio_processing_db[old_name].character_name = new_name
            audio_processing_db[new_name] = audio_processing_db[old_name]
            del audio_processing_db[old_name]
        
        if old_name in training_db:
            training_db[old_name].character_name = new_name
            training_db[new_name] = training_db[old_name]
            del training_db[old_name]
        
        # 更新默认角色
        global default_character
        if default_character == old_name:
            default_character = new_name
            character_info.is_default = True
            self._save_default_character()
        
        # 保存配置
        self._save_character_config(new_name, character_info)
        
        logger.info(f"✅ 角色重命名成功: {old_name} -> {new_name}")
        return character_info
    
    def delete_character(self, character_name: str) -> bool:
        """删除角色"""
        if character_name not in characters_db:
            raise ValueError(f"角色不存在: {character_name}")
        
        # 删除目录
        character_dir = self.get_character_dir(character_name)
        if character_dir.exists():
            shutil.rmtree(character_dir)
        
        # 删除数据库记录
        del characters_db[character_name]
        
        if character_name in audio_processing_db:
            del audio_processing_db[character_name]
        
        if character_name in training_db:
            del training_db[character_name]
        
        # 更新默认角色
        global default_character
        if default_character == character_name:
            # 选择另一个角色作为默认角色
            remaining_characters = list(characters_db.keys())
            if remaining_characters:
                default_character = remaining_characters[0]
                characters_db[default_character].is_default = True
                self._save_default_character()
            else:
                default_character = None
                # 删除默认角色配置文件
                default_file = self.work_dir / "default_character.txt"
                if default_file.exists():
                    default_file.unlink()
        
        logger.info(f"✅ 角色删除成功: {character_name}")
        return True
    
    def set_default_character(self, character_name: str) -> bool:
        """设置默认角色"""
        if character_name not in characters_db:
            raise ValueError(f"角色不存在: {character_name}")
        
        global default_character
        
        # 清除旧的默认标记
        if default_character and default_character in characters_db:
            characters_db[default_character].is_default = False
        
        # 设置新的默认角色
        default_character = character_name
        characters_db[character_name].is_default = True
        
        # 保存配置
        self._save_default_character()
        
        logger.info(f"✅ 默认角色设置为: {character_name}")
        return True
    
    def get_default_character(self) -> Optional[str]:
        """获取默认角色"""
        return default_character
    
    def list_characters(self) -> List[CharacterInfo]:
        """列出所有角色"""
        return list(characters_db.values())
    
    def get_character(self, character_name: str) -> CharacterInfo:
        """获取角色信息"""
        if character_name not in characters_db:
            raise ValueError(f"角色不存在: {character_name}")
        return characters_db[character_name]
    
    # ==================== 目录管理 ====================
    
    def get_character_dir(self, character_name: str) -> Path:
        """获取角色目录"""
        return self.characters_dir / character_name
    
    def get_character_raw_audio_dir(self, character_name: str) -> Path:
        """获取角色原始音频目录"""
        return self.get_character_dir(character_name) / "raw_audio"
    
    def get_character_converted_audio_dir(self, character_name: str) -> Path:
        """获取角色转换后音频目录"""
        return self.get_character_dir(character_name) / "converted_audio"
    
    def get_character_sliced_audio_dir(self, character_name: str) -> Path:
        """获取角色切片音频目录"""
        return self.get_character_dir(character_name) / "sliced_audio"
    
    def get_character_denoised_audio_dir(self, character_name: str) -> Path:
        """获取角色降噪音频目录"""
        return self.get_character_dir(character_name) / "denoised_audio"
    
    def get_character_transcripts_dir(self, character_name: str) -> Path:
        """获取角色转录目录"""
        return self.get_character_dir(character_name) / "transcripts"
    
    def get_character_experiments_dir(self, character_name: str) -> Path:
        """获取角色实验目录"""
        exp_dir = self.get_character_dir(character_name) / "experiments" / character_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir
    
    def get_character_models_dir(self, character_name: str) -> Path:
        """获取角色模型目录"""
        return self.get_character_dir(character_name) / "models"
    
    # ==================== 音频处理管理 ====================
    
    def get_audio_count(self, character_name: str) -> int:
        """获取角色音频数量"""
        raw_audio_dir = self.get_character_raw_audio_dir(character_name)
        if not raw_audio_dir.exists():
            return 0
        
        audio_files = []
        for ext in ['wav', 'mp3', 'm4a', 'flac', 'aac']:
            audio_files.extend(raw_audio_dir.glob(f"*.{ext}"))
        
        return len(audio_files)
    
    def update_character_audio_count(self, character_name: str):
        """更新角色音频数量"""
        if character_name in characters_db:
            characters_db[character_name].audio_count = self.get_audio_count(character_name)
            characters_db[character_name].updated_at = datetime.now()
    
    async def start_audio_processing(self, character_name: str, steps: List[AudioProcessingStep] = None) -> AudioProcessingInfo:
        """开始音频处理"""
        if character_name not in characters_db:
            raise ValueError(f"角色不存在: {character_name}")
        
        if steps is None:
            steps = [
                AudioProcessingStep.CONVERT_AUDIO,
                AudioProcessingStep.SLICE_AUDIO,
                AudioProcessingStep.DENOISE_AUDIO,
                AudioProcessingStep.ASR_TRANSCRIBE
            ]
            
            # 如果禁用降噪，移除降噪步骤
            if not characters_db[character_name].config.enable_denoise:
                steps.remove(AudioProcessingStep.DENOISE_AUDIO)
        
        # 创建音频处理信息
        processing_info = AudioProcessingInfo(
            character_name=character_name,
            status=ProcessingStatus.RUNNING,
            started_at=datetime.now()
        )
        
        audio_processing_db[character_name] = processing_info
        characters_db[character_name].audio_processing_status = ProcessingStatus.RUNNING
        
        # 启动异步处理
        asyncio.create_task(self._execute_audio_processing(character_name, steps))
        
        return processing_info
    
    async def _execute_audio_processing(self, character_name: str, steps: List[AudioProcessingStep]):
        """执行音频处理步骤"""
        processing_info = audio_processing_db[character_name]
        
        try:
            for i, step in enumerate(steps):
                processing_info.current_step = step.value
                processing_info.progress = (i / len(steps)) * 100
                
                logger.info(f"开始执行 {character_name} 的音频处理步骤: {step.value}")
                
                success = await self._execute_audio_processing_step(character_name, step)
                
                if not success:
                    processing_info.status = ProcessingStatus.FAILED
                    processing_info.error_message = f"步骤 {step.value} 失败"
                    characters_db[character_name].audio_processing_status = ProcessingStatus.FAILED
                    return
            
            # 所有步骤完成
            processing_info.status = ProcessingStatus.COMPLETED
            processing_info.progress = 100.0
            processing_info.completed_at = datetime.now()
            processing_info.processed_audio_count = self.get_processed_audio_count(character_name)
            
            characters_db[character_name].audio_processing_status = ProcessingStatus.COMPLETED
            
            logger.info(f"✅ {character_name} 音频处理完成")
            
        except Exception as e:
            processing_info.status = ProcessingStatus.FAILED
            processing_info.error_message = str(e)
            characters_db[character_name].audio_processing_status = ProcessingStatus.FAILED
            logger.error(f"❌ {character_name} 音频处理失败: {e}")
    
    async def _execute_audio_processing_step(self, character_name: str, step: AudioProcessingStep) -> bool:
        """执行单个音频处理步骤"""
        try:
            if step == AudioProcessingStep.CONVERT_AUDIO:
                input_dir = str(self.get_character_raw_audio_dir(character_name))
                output_dir = str(self.get_character_converted_audio_dir(character_name))
                return await self.step_processor.convert_audio(input_dir, output_dir)
                
            elif step == AudioProcessingStep.SLICE_AUDIO:
                input_dir = str(self.get_character_converted_audio_dir(character_name))
                output_dir = str(self.get_character_sliced_audio_dir(character_name))
                return await self.step_processor.slice_audio(input_dir, output_dir)
                
            elif step == AudioProcessingStep.DENOISE_AUDIO:
                input_dir = str(self.get_character_sliced_audio_dir(character_name))
                output_dir = str(self.get_character_denoised_audio_dir(character_name))
                return await self.step_processor.denoise_audio(input_dir, output_dir)
                
            elif step == AudioProcessingStep.ASR_TRANSCRIBE:
                if characters_db[character_name].config.enable_denoise:
                    input_dir = str(self.get_character_denoised_audio_dir(character_name))
                else:
                    input_dir = str(self.get_character_sliced_audio_dir(character_name))
                    
                output_dir = str(self.get_character_transcripts_dir(character_name))
                language = characters_db[character_name].config.language
                return await self.step_processor.asr_transcribe(input_dir, output_dir, language)
            
            return False
            
        except Exception as e:
            logger.error(f"音频处理步骤 {step.value} 失败: {e}")
            return False
    
    def get_processed_audio_count(self, character_name: str) -> int:
        """获取已处理音频数量"""
        if characters_db[character_name].config.enable_denoise:
            audio_dir = self.get_character_denoised_audio_dir(character_name)
        else:
            audio_dir = self.get_character_sliced_audio_dir(character_name)
            
        if not audio_dir.exists():
            return 0
            
        audio_files = list(audio_dir.glob("*.wav"))
        return len(audio_files)
    
    # ==================== 训练管理 ====================
    
    async def start_training(self, character_name: str, steps: List[TrainingStep] = None) -> TrainingInfo:
        """开始训练"""
        if character_name not in characters_db:
            raise ValueError(f"角色不存在: {character_name}")
        
        # 检查音频处理是否完成
        if characters_db[character_name].audio_processing_status != ProcessingStatus.COMPLETED:
            raise ValueError(f"角色 {character_name} 的音频处理尚未完成，无法开始训练")
        
        if steps is None:
            steps = [
                TrainingStep.EXTRACT_TEXT_FEATURES,
                TrainingStep.EXTRACT_AUDIO_FEATURES,
                TrainingStep.EXTRACT_SPEAKER_VECTORS,
                TrainingStep.EXTRACT_SEMANTIC_FEATURES,
                TrainingStep.TRAIN_SOVITS,
                TrainingStep.TRAIN_GPT
            ]
        
        # 创建训练信息
        training_info = TrainingInfo(
            character_name=character_name,
            status=ProcessingStatus.RUNNING,
            started_at=datetime.now()
        )
        
        training_db[character_name] = training_info
        characters_db[character_name].training_status = ProcessingStatus.RUNNING
        
        # 启动异步训练
        asyncio.create_task(self._execute_training(character_name, steps))
        
        return training_info
    
    async def _execute_training(self, character_name: str, steps: List[TrainingStep]):
        """执行训练步骤"""
        training_info = training_db[character_name]
        
        try:
            for i, step in enumerate(steps):
                training_info.current_step = step.value
                training_info.progress = (i / len(steps)) * 100
                
                logger.info(f"开始执行 {character_name} 的训练步骤: {step.value}")
                
                success = await self._execute_training_step(character_name, step)
                
                if not success:
                    training_info.status = ProcessingStatus.FAILED
                    training_info.error_message = f"步骤 {step.value} 失败"
                    characters_db[character_name].training_status = ProcessingStatus.FAILED
                    return
            
            # 所有步骤完成
            training_info.status = ProcessingStatus.COMPLETED
            training_info.progress = 100.0
            training_info.completed_at = datetime.now()
            
            # 查找生成的模型文件
            self._find_trained_models(character_name)
            # 同步模型到角色目录
            self._sync_character_models_dir(character_name)
            
            characters_db[character_name].training_status = ProcessingStatus.COMPLETED
            characters_db[character_name].model_exists = True
            
            logger.info(f"✅ {character_name} 训练完成")
            
        except Exception as e:
            training_info.status = ProcessingStatus.FAILED
            training_info.error_message = str(e)
            characters_db[character_name].training_status = ProcessingStatus.FAILED
            logger.error(f"❌ {character_name} 训练失败: {e}")
    
    async def _execute_training_step(self, character_name: str, step: TrainingStep) -> bool:
        """执行单个训练步骤"""
        try:
            config = self._build_training_config(character_name)
            
            if step == TrainingStep.EXTRACT_TEXT_FEATURES:
                env_vars = self._build_step_env(config)
                gpu_ids = config["GPU_ID"]
                parallel_parts = len(gpu_ids.split('-')) if '-' in gpu_ids else 1
                
                success = await self.step_processor.extract_features(
                    "1-get-text.py", env_vars, parallel_parts, gpu_ids
                )
                
                if success:
                    await self.step_processor.merge_feature_files(
                        config["EXP_DIR"], "2-name2text-{}.txt", "2-name2text.txt", 
                        has_header=False, parallel_parts=parallel_parts
                    )
                
                return success
                
            elif step == TrainingStep.EXTRACT_AUDIO_FEATURES:
                env_vars = self._build_step_env(config)
                gpu_ids = config["GPU_ID"]
                parallel_parts = len(gpu_ids.split('-')) if '-' in gpu_ids else 1
                
                return await self.step_processor.extract_features(
                    "2-get-hubert-wav32k.py", env_vars, parallel_parts, gpu_ids
                )
                
            elif step == TrainingStep.EXTRACT_SPEAKER_VECTORS:
                env_vars = self._build_step_env(config)
                gpu_ids = config["GPU_ID"]
                parallel_parts = len(gpu_ids.split('-')) if '-' in gpu_ids else 1
                
                return await self.step_processor.extract_features(
                    "2-get-sv.py", env_vars, parallel_parts, gpu_ids
                )
                
            elif step == TrainingStep.EXTRACT_SEMANTIC_FEATURES:
                env_vars = self._build_step_env(config)
                gpu_ids = config["GPU_ID"]
                parallel_parts = len(gpu_ids.split('-')) if '-' in gpu_ids else 1
                
                success = await self.step_processor.extract_features(
                    "3-get-semantic.py", env_vars, parallel_parts, gpu_ids
                )
                
                if success:
                    await self.step_processor.merge_feature_files(
                        config["EXP_DIR"], "6-name2semantic-{}.tsv", "6-name2semantic.tsv", 
                        has_header=True, parallel_parts=parallel_parts
                    )
                
                return success
                
            elif step == TrainingStep.TRAIN_SOVITS:
                s2_config_path = str(self.get_character_dir(character_name) / "config_s2.json")
                self.config_generator.generate_s2_config(config, s2_config_path)
                return await self.step_processor.train_model("s2_train.py", s2_config_path)
                
            elif step == TrainingStep.TRAIN_GPT:
                s1_config_path = str(self.get_character_dir(character_name) / "config_s1.yaml")
                self.config_generator.generate_s1_config(config, s1_config_path)
                env_vars = {"hz": "25hz"}
                return await self.step_processor.train_model("s1_train.py", s1_config_path, env_vars)
            
            return False
            
        except Exception as e:
            logger.error(f"训练步骤 {step.value} 失败: {e}")
            return False
    
    # ==================== 推理管理 ====================
    
    async def start_inference(self, request: InferenceRequest) -> InferenceInfo:
        """开始推理"""
        # 确定使用的角色
        character_name = request.character_name or default_character
        
        if not character_name:
            raise ValueError("未指定角色且无默认角色")
        
        if character_name not in characters_db:
            raise ValueError(f"角色不存在: {character_name}")
        
        if not characters_db[character_name].model_exists:
            raise ValueError(f"角色 {character_name} 的模型尚未训练完成")
        
        # 创建推理信息
        inference_id = str(uuid.uuid4())
        inference_info = InferenceInfo(
            inference_id=inference_id,
            character_name=character_name,
            target_text=request.target_text,
            status=ProcessingStatus.RUNNING,
            created_at=datetime.now()
        )
        
        inference_db[inference_id] = inference_info
        
        # 启动异步推理
        asyncio.create_task(self._execute_inference(inference_id, request))
        
        return inference_info
    
    async def _execute_inference(self, inference_id: str, request: InferenceRequest):
        """执行推理"""
        inference_info = inference_db[inference_id]
        character_name = inference_info.character_name
        
        try:
            # 构建推理参数
            inference_params = await self._build_inference_params(character_name, request)
            
            # 执行推理
            success = await self.step_processor.test_inference(**inference_params)
            
            if success:
                inference_info.status = ProcessingStatus.COMPLETED
                inference_info.completed_at = datetime.now()
                # 输出路径应该是包含output.wav的完整文件路径
                output_dir = inference_params.get("output_path")
                output_file = Path(output_dir) / "output.wav"
                inference_info.output_path = str(output_file)
                logger.info(f"✅ 推理完成: {inference_id}")
                logger.info(f"   输出文件: {inference_info.output_path}")
            else:
                inference_info.status = ProcessingStatus.FAILED
                inference_info.error_message = "推理执行失败"
                logger.error(f"❌ 推理失败: {inference_id}")
                
        except Exception as e:
            inference_info.status = ProcessingStatus.FAILED
            inference_info.error_message = str(e)
            logger.error(f"❌ 推理异常: {inference_id} - {e}")
    
    # ==================== 辅助方法 ====================
    
    def _load_existing_characters(self):
        """加载现有角色"""
        if not self.characters_dir.exists():
            return
        
        for character_dir in self.characters_dir.iterdir():
            if character_dir.is_dir():
                character_name = character_dir.name
                config_file = character_dir / "character_config.json"
                
                if config_file.exists():
                    try:
                        with open(config_file, 'r', encoding='utf-8') as f:
                            config_data = json.load(f)
                        
                        character_info = CharacterInfo(**config_data)
                        
                        # 动态更新GPU配置：如果当前环境与配置文件不一致，更新配置
                        current_gpu_env = os.environ.get("CUDA_VISIBLE_DEVICES")
                        if current_gpu_env and character_info.config.gpu_id != current_gpu_env:
                            old_gpu_id = character_info.config.gpu_id
                            character_info.config.gpu_id = current_gpu_env
                            logger.info(f"角色 {character_name} GPU配置已更新: {old_gpu_id} -> {current_gpu_env}")
                            # 保存更新后的配置
                            self._save_character_config(character_name, character_info)
                        
                        characters_db[character_name] = character_info
                        
                        # 更新音频数量和状态
                        self.update_character_audio_count(character_name)
                        self._update_character_status(character_name)
                        
                        # 自动检查模型状态
                        if self._check_models_exist(character_name):
                            logger.info(f"✅ 角色 {character_name} 已训练完成")
                            # 将模型同步到角色目录
                            self._sync_character_models_dir(character_name)
                        else:
                            logger.info(f"⚠️  角色 {character_name} 尚未训练完成")
                        
                        logger.info(f"加载角色: {character_name}")
                        
                    except Exception as e:
                        logger.warning(f"加载角色配置失败 {character_name}: {e}")
    
    def _load_default_character(self):
        """加载默认角色设置"""
        global default_character
        default_file = self.work_dir / "default_character.txt"
        
        if default_file.exists():
            try:
                with open(default_file, 'r', encoding='utf-8') as f:
                    character_name = f.read().strip()
                
                if character_name in characters_db:
                    default_character = character_name
                    characters_db[character_name].is_default = True
                    logger.info(f"加载默认角色: {character_name}")
                else:
                    logger.warning(f"默认角色不存在: {character_name}")
                    
            except Exception as e:
                logger.warning(f"加载默认角色配置失败: {e}")
    
    def _save_character_config(self, character_name: str, character_info: CharacterInfo):
        """保存角色配置"""
        character_dir = self.get_character_dir(character_name)
        config_file = character_dir / "character_config.json"
        
        config_data = character_info.dict()
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False, default=str)
    
    def _save_default_character(self):
        """保存默认角色设置"""
        global default_character
        default_file = self.work_dir / "default_character.txt"
        
        if default_character:
            with open(default_file, 'w', encoding='utf-8') as f:
                f.write(default_character)
    
    def _update_character_status(self, character_name: str):
        """更新角色状态"""
        character_info = characters_db[character_name]
        
        # 检查音频处理状态
        if character_name in audio_processing_db:
            character_info.audio_processing_status = audio_processing_db[character_name].status
        
        # 检查训练状态
        if character_name in training_db:
            character_info.training_status = training_db[character_name].status
        
        # 检查模型是否存在
        character_info.model_exists = self._check_models_exist(character_name)
    
    def _check_models_exist(self, character_name: str) -> bool:
        """检查模型是否存在"""
        # 首先检查training_db中的路径
        training_info = training_db.get(character_name)
        if training_info and training_info.gpt_model_path and training_info.sovits_model_path:
            gpt_exists = Path(training_info.gpt_model_path).exists()
            sovits_exists = Path(training_info.sovits_model_path).exists()
            if gpt_exists and sovits_exists:
                return True
        
        # 如果training_db中没有或路径无效，直接查找文件系统
        return self._find_models_in_filesystem(character_name)
    
    def _find_models_in_filesystem(self, character_name: str) -> bool:
        """在文件系统中查找模型文件"""
        try:
            # 查找GPT模型
            gpt_weights_dir = self.base_dir / "GPT_weights_v2ProPlus"
            if gpt_weights_dir.exists():
                gpt_files = list(gpt_weights_dir.glob(f"{character_name}*.ckpt"))
                if gpt_files:
                    # 选择最新的GPT模型
                    latest_gpt = max(gpt_files, key=lambda x: x.stat().st_mtime)
                    
                    # 查找SoVITS模型
                    sovits_weights_dir = self.base_dir / "SoVITS_weights_v2ProPlus"
                    if sovits_weights_dir.exists():
                        sovits_files = list(sovits_weights_dir.glob(f"{character_name}*.pth"))
                        if sovits_files:
                            # 选择最新的SoVITS模型
                            latest_sovits = max(sovits_files, key=lambda x: x.stat().st_mtime)
                            
                            # 更新training_db
                            if character_name not in training_db:
                                training_db[character_name] = TrainingInfo(
                                    character_name=character_name,
                                    status=ProcessingStatus.COMPLETED,
                                    gpt_model_path=str(latest_gpt),
                                    sovits_model_path=str(latest_sovits)
                                )
                            else:
                                training_db[character_name].gpt_model_path = str(latest_gpt)
                                training_db[character_name].sovits_model_path = str(latest_sovits)
                                training_db[character_name].status = ProcessingStatus.COMPLETED
                            
                            logger.info(f"✅ 发现已训练的模型: {character_name}")
                            logger.info(f"   GPT模型: {latest_gpt}")
                            logger.info(f"   SoVITS模型: {latest_sovits}")
                            # 同步模型到角色目录
                            self._sync_character_models_dir(character_name)
                            return True
            
            return False
            
        except Exception as e:
            logger.warning(f"查找模型文件失败 {character_name}: {e}")
            return False
    
    def _sync_character_models_dir(self, character_name: str):
        """将已发现/训练完成的模型同步到角色的 models 目录（使用符号链接，若不支持则复制）"""
        if character_name not in training_db:
            return
        training_info = training_db[character_name]
        if not training_info.gpt_model_path or not training_info.sovits_model_path:
            return
        gpt_src = Path(training_info.gpt_model_path)
        sovits_src = Path(training_info.sovits_model_path)
        if not gpt_src.exists() or not sovits_src.exists():
            return
        models_dir = self.get_character_models_dir(character_name)
        models_dir.mkdir(parents=True, exist_ok=True)
        gpt_dst = models_dir / gpt_src.name
        sovits_dst = models_dir / sovits_src.name
        
        # 建立链接或复制
        for src, dst in [(gpt_src, gpt_dst), (sovits_src, sovits_dst)]:
            try:
                if dst.exists():
                    # 已存在则跳过
                    continue
                # 优先创建符号链接
                dst.symlink_to(src)
            except Exception:
                # 回退为复制
                try:
                    shutil.copy2(src, dst)
                except Exception as copy_err:
                    logger.warning(f"同步模型到角色目录失败 {character_name}: {copy_err}")
        
        # 同步完成后，更新training_db中的模型路径为角色目录下的路径
        if gpt_dst.exists() and sovits_dst.exists():
            training_info.gpt_model_path = str(gpt_dst)
            training_info.sovits_model_path = str(sovits_dst)
            logger.info(f"✅ 模型路径已更新为角色目录: {character_name}")
            logger.info(f"   GPT模型: {gpt_dst}")
            logger.info(f"   SoVITS模型: {sovits_dst}")
    
    def _build_training_config(self, character_name: str) -> Dict[str, Any]:
        """构建训练配置"""
        character_info = characters_db[character_name]
        config = character_info.config
        
        # 查找ASR输出文件
        transcripts_dir = self.get_character_transcripts_dir(character_name)
        asr_output = self._find_asr_output_file(str(transcripts_dir))
        
        if config.enable_denoise:
            wav_dir = str(self.get_character_denoised_audio_dir(character_name))
        else:
            wav_dir = str(self.get_character_sliced_audio_dir(character_name))
        
        # 获取模型路径配置
        model_paths = get_model_paths()
        
        return {
            "ASR_OUTPUT": asr_output,
            "DENOISED_DIR": wav_dir,
            "EXP_NAME": character_name,
            "EXP_DIR": str(self.get_character_experiments_dir(character_name)),
            "BERT_DIR": model_paths["bert_dir"],
            "CNHUBERT_DIR": model_paths["cnhubert_dir"],
            "PRETRAINED_SV": model_paths["pretrained_sv"],
            "BATCH_SIZE": config.batch_size,
            "EPOCHS_S2": config.epochs_s2,
            "EPOCHS_S1": config.epochs_s1,
            "GPU_ID": config.gpu_id,
            "LANGUAGE": config.language,
            "IS_HALF": True,
            "VERSION": "v2ProPlus"
        }
    
    def _find_asr_output_file(self, asr_output_path: str) -> str:
        """智能查找ASR输出文件"""
        asr_path = Path(asr_output_path)
        
        if asr_path.is_file():
            return str(asr_path)
        
        if asr_path.is_dir():
            # 按优先级查找文件
            for pattern in ["*.list", "*.txt", "*.tsv"]:
                files = list(asr_path.glob(pattern))
                if files:
                    return str(files[0])
            
            raise FileNotFoundError(f"在ASR输出目录中找不到转录文件: {asr_output_path}")
        
        raise FileNotFoundError(f"ASR输出路径不存在: {asr_output_path}")
    
    def _build_step_env(self, config: Dict[str, Any]) -> Dict[str, str]:
        """构建步骤执行环境变量"""
        env = os.environ.copy()
        env.update({
            "inp_text": config["ASR_OUTPUT"],
            "inp_wav_dir": config["DENOISED_DIR"],
            "exp_name": config["EXP_NAME"],
            "opt_dir": config["EXP_DIR"],
            "bert_pretrained_dir": config["BERT_DIR"],
            "cnhubert_base_dir": config["CNHUBERT_DIR"],
            "sv_path": config["PRETRAINED_SV"],
            "s2config_path": "GPT_SoVITS/configs/s2v2ProPlus.json",
            "is_half": "True",
            "BATCH_SIZE": str(config["BATCH_SIZE"]),
            "EPOCHS_S2": str(config["EPOCHS_S2"]),
            "S2_VERSION": "v2ProPlus",
            "_CUDA_VISIBLE_DEVICES": config["GPU_ID"]
        })
        return env
    
    def _find_trained_models(self, character_name: str):
        """查找训练生成的模型文件"""
        training_info = training_db[character_name]
        
        try:
            # 查找GPT模型 - 按角色名称查找
            gpt_weights_dir = self.base_dir / "GPT_weights_v2ProPlus"
            if gpt_weights_dir.exists():
                gpt_files = list(gpt_weights_dir.glob(f"{character_name}*.ckpt"))
                if gpt_files:
                    # 选择最新的GPT模型
                    latest_gpt = max(gpt_files, key=lambda x: x.stat().st_mtime)
                    training_info.gpt_model_path = str(latest_gpt)
                    logger.info(f"✅ 找到GPT模型: {latest_gpt}")
                else:
                    logger.warning(f"未找到角色 {character_name} 的GPT模型")
            
            # 查找SoVITS模型 - 按角色名称查找
            sovits_weights_dir = self.base_dir / "SoVITS_weights_v2ProPlus"
            if sovits_weights_dir.exists():
                sovits_files = list(sovits_weights_dir.glob(f"{character_name}*.pth"))
                if sovits_files:
                    # 选择最新的SoVITS模型
                    latest_sovits = max(sovits_files, key=lambda x: x.stat().st_mtime)
                    training_info.sovits_model_path = str(latest_sovits)
                    logger.info(f"✅ 找到SoVITS模型: {latest_sovits}")
                else:
                    logger.warning(f"未找到角色 {character_name} 的SoVITS模型")
                    
        except Exception as e:
            logger.warning(f"查找训练模型失败 {character_name}: {e}")
    
    async def _build_inference_params(self, character_name: str, request: InferenceRequest) -> Dict[str, Any]:
        """构建推理参数"""
        training_info = training_db[character_name]
        config = characters_db[character_name].config
        
        # 确定参考音频
        ref_audio = request.ref_audio
        if not ref_audio:
            # 自动选择参考音频
            if config.enable_denoise:
                audio_dir = self.get_character_denoised_audio_dir(character_name)
            else:
                audio_dir = self.get_character_sliced_audio_dir(character_name)
            
            audio_files = list(audio_dir.glob("*.wav"))
            if audio_files:
                ref_audio = str(audio_files[0])
            else:
                raise ValueError(f"找不到参考音频文件: {character_name}")
        
        # 确定参考文本
        ref_text = request.ref_text
        if not ref_text:
            # 从ASR输出中提取
            transcripts_dir = self.get_character_transcripts_dir(character_name)
            asr_output = self._find_asr_output_file(str(transcripts_dir))
            
            ref_audio_name = Path(ref_audio).stem
            ref_text = self._extract_ref_text_from_asr(asr_output, ref_audio_name)
        
        # 构建输出路径 - 创建唯一的输出目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir_name = f"{character_name}_{timestamp}"
        output_dir = self.inference_output_dir / output_dir_name
        output_dir.mkdir(exist_ok=True)
        
        # 设置输出路径为目录，inference_cli.py会在其中创建output.wav
        output_path = str(output_dir)
        
        # 创建目标文本文件
        target_text_file = self.inference_output_dir / f"target_text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(target_text_file, 'w', encoding='utf-8') as f:
            f.write(request.target_text)
        
        # 创建参考文本文件
        ref_text_file = self.inference_output_dir / f"ref_text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(ref_text_file, 'w', encoding='utf-8') as f:
            f.write(ref_text)
        
        # 获取模型路径配置
        model_paths = get_model_paths()
        
        # 动态获取GPU设置：优先使用当前环境的CUDA_VISIBLE_DEVICES，确保与服务启动环境一致
        current_gpu_env = os.environ.get("CUDA_VISIBLE_DEVICES")
        gpu_number = config.gpu_id
        
        logger.info(f"推理使用GPU设备: {gpu_number} (环境变量: {current_gpu_env}, 角色配置: {config.gpu_id})")
        
        return {
            "gpt_model": training_info.gpt_model_path,
            "sovits_model": training_info.sovits_model_path,
            "ref_audio": ref_audio,
            "ref_text": str(ref_text_file),
            "ref_language": request.ref_language,
            "target_text": str(target_text_file),
            "target_language": request.target_language,
            "output_path": output_path,
            "bert_path": model_paths["bert_dir"],
            "cnhubert_base_path": model_paths["cnhubert_dir"],
            "gpu_number": gpu_number,
            "is_half": True
        }
    
    def _extract_ref_text_from_asr(self, asr_output: str, ref_audio_name: str) -> str:
        """从ASR输出中提取参考文本"""
        try:
            with open(asr_output, 'r', encoding='utf-8') as f:
                for line in f:
                    if ref_audio_name in line:
                        if '|' in line:
                            # .list格式
                            parts = line.strip().split('|')
                            if len(parts) >= 4:
                                return parts[3]
                        elif '\t' in line:
                            # .tsv格式
                            parts = line.strip().split('\t')
                            if len(parts) >= 2:
                                return parts[1]
                        else:
                            # 其他格式
                            return line.strip()
                            
            return "这是一个参考音频的文本内容。"
            
        except Exception as e:
            logger.warning(f"提取参考文本失败: {e}")
            return "这是一个参考音频的文本内容。"

# 初始化服务
training_service = CharacterBasedTrainingService()

# ==================== API路由 ====================

# 角色管理API
@app.post("/api/v1/characters", response_model=CharacterInfo)
async def create_character(request: CharacterCreateRequest):
    """创建角色"""
    try:
        character_info = training_service.create_character(request.character_name, request.config)
        return character_info
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/v1/characters", response_model=List[CharacterInfo])
async def list_characters():
    """列出所有角色"""
    return training_service.list_characters()

@app.get("/api/v1/characters/{character_name}", response_model=CharacterInfo)
async def get_character(character_name: str):
    """获取角色信息"""
    try:
        return training_service.get_character(character_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.put("/api/v1/characters/{character_name}", response_model=CharacterInfo)
async def rename_character(character_name: str, request: CharacterRenameRequest):
    """重命名角色"""
    try:
        return training_service.rename_character(character_name, request.new_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/api/v1/characters/{character_name}")
async def delete_character(character_name: str):
    """删除角色"""
    try:
        success = training_service.delete_character(character_name)
        return {"message": "角色删除成功", "success": success}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/api/v1/characters/{character_name}/set_default")
async def set_default_character(character_name: str):
    """设置默认角色"""
    try:
        success = training_service.set_default_character(character_name)
        return {"message": "默认角色设置成功", "success": success}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/api/v1/default_character")
async def get_default_character():
    """获取默认角色"""
    default_char = training_service.get_default_character()
    return {"default_character": default_char}

# 音频上传API
@app.post("/api/v1/characters/{character_name}/audio/upload")
async def upload_audio(character_name: str, file: UploadFile = File(...)):
    """上传音频文件"""
    try:
        character_info = training_service.get_character(character_name)
    except ValueError:
        raise HTTPException(status_code=404, detail="角色不存在")
    
    # 获取角色音频目录
    audio_dir = training_service.get_character_raw_audio_dir(character_name)
    file_path = audio_dir / file.filename
    
    # 保存文件
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # 更新音频数量
    training_service.update_character_audio_count(character_name)
    
    return {"message": "音频上传成功", "filename": file.filename, "path": str(file_path)}

# 音频处理API
@app.post("/api/v1/characters/{character_name}/audio/process", response_model=AudioProcessingInfo)
async def start_audio_processing(character_name: str, background_tasks: BackgroundTasks):
    """开始音频处理"""
    try:
        processing_info = await training_service.start_audio_processing(character_name)
        return processing_info
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/v1/characters/{character_name}/audio/status", response_model=AudioProcessingInfo)
async def get_audio_processing_status(character_name: str):
    """获取音频处理状态"""
    if character_name not in audio_processing_db:
        raise HTTPException(status_code=404, detail="音频处理记录不存在")
    return audio_processing_db[character_name]

# 训练API
@app.post("/api/v1/characters/{character_name}/training/start", response_model=TrainingInfo)
async def start_training(character_name: str, background_tasks: BackgroundTasks):
    """开始训练"""
    try:
        training_info = await training_service.start_training(character_name)
        return training_info
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/v1/characters/{character_name}/training/status", response_model=TrainingInfo)
async def get_training_status(character_name: str):
    """获取训练状态"""
    if character_name not in training_db:
        raise HTTPException(status_code=404, detail="训练记录不存在")
    return training_db[character_name]

# 推理API
@app.post("/api/v1/inference", response_model=InferenceInfo)
async def start_inference(request: InferenceRequest, background_tasks: BackgroundTasks):
    """开始推理"""
    try:
        inference_info = await training_service.start_inference(request)
        return inference_info
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/v1/inference/{inference_id}", response_model=InferenceInfo)
async def get_inference_status(inference_id: str):
    """获取推理状态"""
    if inference_id not in inference_db:
        raise HTTPException(status_code=404, detail="推理记录不存在")
    return inference_db[inference_id]

@app.get("/api/v1/inference")
async def list_inference():
    """列出所有推理记录"""
    return list(inference_db.values())

# 文件下载API
@app.get("/api/v1/characters/{character_name}/download/{filename}")
async def download_character_file(character_name: str, filename: str):
    """下载角色相关文件"""
    try:
        character_info = training_service.get_character(character_name)
    except ValueError:
        raise HTTPException(status_code=404, detail="角色不存在")
    
    # 查找文件
    character_dir = training_service.get_character_dir(character_name)
    search_dirs = [
        character_dir / "models",
        character_dir / "experiments" / character_name,
        character_dir,
        training_service.inference_output_dir
    ]
    
    file_path = None
    for search_dir in search_dirs:
        candidate_path = search_dir / filename
        if candidate_path.exists():
            file_path = candidate_path
            break
    
    if not file_path:
        raise HTTPException(status_code=404, detail="文件不存在")
    
    return FileResponse(file_path, filename=filename)

@app.get("/api/v1/inference/{inference_id}/download")
async def download_inference_result(inference_id: str):
    """下载推理结果"""
    if inference_id not in inference_db:
        raise HTTPException(status_code=404, detail="推理记录不存在")
    
    inference_info = inference_db[inference_id]
    
    if inference_info.status != ProcessingStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="推理尚未完成")
    
    if not inference_info.output_path or not Path(inference_info.output_path).exists():
        raise HTTPException(status_code=404, detail="推理结果文件不存在")
    
    filename = f"{inference_info.character_name}_{inference_id[:8]}.wav"
    return FileResponse(inference_info.output_path, filename=filename)

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "GPT-SoVITS 基于角色的训练服务API",
        "version": "2.0.0",
        "docs_url": "/docs",
        "features": [
            "角色管理",
            "音频处理",
            "模型训练", 
            "语音推理"
        ]
    }

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="GPT-SoVITS训练服务API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python training_service.py                    # 使用默认端口8000
  python training_service.py --port 8216       # 指定端口8216
  python training_service.py -p 9000           # 使用短参数指定端口
  python training_service.py --host 127.0.0.1  # 指定主机地址
  python training_service.py --config          # 显示当前配置
  python training_service.py --help            # 显示帮助信息
        """
    )
    
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=None,
        help="服务端口号 (默认: 从配置文件读取)"
    )
    
    parser.add_argument(
        "-H", "--host",
        type=str,
        default=None,
        help="绑定主机地址 (默认: 从配置文件读取)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="工作进程数 (默认: 从配置文件读取)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["debug", "info", "warning", "error"],
        help="日志级别 (默认: 从配置文件读取)"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="开发模式：代码变更时自动重载"
    )
    
    parser.add_argument(
        "--config",
        action="store_true",
        help="显示当前配置信息"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # 如果只是显示配置，则显示后退出
    if args.config:
        if os.getenv('GPT_SOVITS_SERVER_MODE') == 'standalone':
            from service_config import print_config
        else:
            from .service_config import print_config
        print_config()
        exit(0)
    
    # 导入配置
    if os.getenv('GPT_SOVITS_SERVER_MODE') == 'standalone':
        from service_config import SERVICE_CONFIG
    else:
        from .service_config import SERVICE_CONFIG
    
    # 使用命令行参数覆盖配置文件，如果没有指定则使用配置文件的值
    host = args.host or SERVICE_CONFIG["host"]
    port = args.port or SERVICE_CONFIG["port"]
    workers = args.workers or SERVICE_CONFIG["workers"]
    log_level = args.log_level or SERVICE_CONFIG["log_level"]
    
    logger.info(f"🚀 启动 GPT-SoVITS 训练服务API")
    logger.info(f"📍 服务地址: http://{host}:{port}")
    logger.info(f"📚 API文档: http://{host}:{port}/docs")
    logger.info(f"🔧 工作进程: {workers}")
    logger.info(f"📝 日志级别: {log_level}")
    logger.info(f"🔄 自动重载: {'开启' if args.reload else '关闭'}")
    logger.info("=" * 50)
    
    if args.reload:
        # 使用 reload 模式时，必须使用模块导入字符串
        uvicorn.run(
            "server.training_service:app",
            host=host, 
            port=port,
            workers=1,  # reload 模式下 workers 必须为 1
            log_level=log_level,
            reload=True
        )
    else:
        # 非 reload 模式时，可以直接传递应用对象
        uvicorn.run(
            app, 
            host=host, 
            port=port,
            workers=workers,
            log_level=log_level,
            reload=False
        )
