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

# 任务状态枚举
class TaskStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# 训练步骤枚举
class TrainingStep(str, Enum):
    CONVERT_AUDIO = "convert_audio"
    SLICE_AUDIO = "slice_audio"
    DENOISE_AUDIO = "denoise_audio"
    ASR_TRANSCRIBE = "asr_transcribe"
    EXTRACT_TEXT_FEATURES = "extract_text_features"
    EXTRACT_AUDIO_FEATURES = "extract_audio_features"
    EXTRACT_SPEAKER_VECTORS = "extract_speaker_vectors"
    EXTRACT_SEMANTIC_FEATURES = "extract_semantic_features"
    TRAIN_SOVITS = "train_sovits"
    TRAIN_GPT = "train_gpt"
    TEST_INFERENCE = "test_inference"

# 数据模型
class TaskConfig(BaseModel):
    exp_name: str = Field(default="my_speaker", description="实验名称")
    language: str = Field(default="zh", description="语言设置")
    batch_size: int = Field(default=16, description="批次大小")
    epochs_s2: int = Field(default=50, description="SoVITS训练轮数")
    epochs_s1: int = Field(default=15, description="GPT训练轮数")
    gpu_id: str = Field(default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"), description="GPU设备ID")

class TaskCreateRequest(BaseModel):
    task_name: str = Field(description="任务名称")
    config: TaskConfig = Field(description="训练配置")

class TaskInfo(BaseModel):
    task_id: str
    task_name: str
    status: TaskStatus
    config: TaskConfig
    created_at: datetime
    updated_at: datetime
    current_step: Optional[str] = None
    progress: float = 0.0
    error_message: Optional[str] = None

class StepExecuteRequest(BaseModel):
    params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="步骤特定参数")

# 全局任务存储
tasks_db: Dict[str, TaskInfo] = {}
step_processes: Dict[str, subprocess.Popen] = {}

# FastAPI应用
app = FastAPI(
    title="GPT-SoVITS训练服务",
    description="提供完整的语音克隆训练流程API接口",
    version="1.0.0"
)

class TrainingService:
    """训练服务核心类"""
    
    def __init__(self):
        # 使用配置文件获取路径
        self.base_dir = get_base_path()
        self.work_dir = get_work_dir()
        
        self.step_processor = StepProcessor(str(self.base_dir))
        self.config_generator = ConfigGenerator(str(self.base_dir))
        
        logger.info(f"✅ 初始化完成:")
        logger.info(f"   基础目录: {self.base_dir}")
        logger.info(f"   工作目录: {self.work_dir}")
    
    def get_task_dir(self, task_id: str) -> Path:
        """获取任务工作目录"""
        task_dir = self.work_dir / task_id
        task_dir.mkdir(exist_ok=True)
        return task_dir
    
    def get_task_input_dir(self, task_id: str) -> Path:
        """获取任务输入目录"""
        input_dir = self.get_task_dir(task_id) / "input_audio"
        input_dir.mkdir(exist_ok=True)
        return input_dir
    
    def get_task_output_dir(self, task_id: str) -> Path:
        """获取任务输出目录"""
        output_dir = self.get_task_dir(task_id) / "output"
        output_dir.mkdir(exist_ok=True)
        return output_dir
    
    def create_task_config_file(self, task_id: str, config: TaskConfig) -> Path:
        """创建任务配置文件"""
        task_dir = self.get_task_dir(task_id)
        config_file = task_dir / "task_config.json"
        
        # 获取模型路径配置
        model_paths = get_model_paths()
        
        # 基于原始脚本的配置结构
        task_config = {
            "INPUT_AUDIO": str(self.get_task_input_dir(task_id)),
            "WORK_DIR": str(task_dir),
            "EXP_NAME": config.exp_name,
            "EXP_DIR": str(task_dir / "experiments" / config.exp_name),
            "SLICED_DIR": str(task_dir / "sliced"),
            "DENOISED_DIR": str(task_dir / "denoised"), 
            "ASR_OUTPUT": str(task_dir / "transcripts"),
            "BERT_DIR": model_paths["bert_dir"],
            "CNHUBERT_DIR": model_paths["cnhubert_dir"],
            "PRETRAINED_SV": model_paths["pretrained_sv"],
            "BATCH_SIZE": config.batch_size,
            "EPOCHS_S2": config.epochs_s2,
            "EPOCHS_S1": config.epochs_s1,
            "GPU_ID": config.gpu_id,
            "LANGUAGE": config.language
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(task_config, f, indent=2, ensure_ascii=False)
        
        return config_file
    
    def load_task_config(self, task_id: str) -> Dict[str, Any]:
        """加载任务配置"""
        config_file = self.get_task_dir(task_id) / "task_config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"任务配置文件不存在: {config_file}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    async def execute_step(self, task_id: str, step: TrainingStep, params: Dict[str, Any] = None):
        """执行训练步骤"""
        if task_id not in tasks_db:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        task_info = tasks_db[task_id]
        if task_info.status == TaskStatus.RUNNING:
            raise HTTPException(status_code=400, detail="任务正在运行中")
        
        # 更新任务状态
        task_info.status = TaskStatus.RUNNING
        task_info.current_step = step
        task_info.updated_at = datetime.now()
        
        # 设置当前步骤的起始进度
        step_start_progress = self.get_step_progress(task_id, step)
        task_info.progress = step_start_progress
        logger.info(f"步骤 {step.value} 开始执行，起始进度: {step_start_progress:.1f}%")
        
        try:
            config = self.load_task_config(task_id)
            success = False
            
            # 使用StepProcessor执行具体步骤
            if step == TrainingStep.CONVERT_AUDIO:
                success = await self.step_processor.convert_audio(
                    config["INPUT_AUDIO"],
                    config["WORK_DIR"] + "/converted_wav"
                )
                # 更新配置中的输入目录
                if success:
                    config["INPUT_AUDIO"] = config["WORK_DIR"] + "/converted_wav"
                    self._save_task_config(task_id, config)
                    
            elif step == TrainingStep.SLICE_AUDIO:
                success = await self.step_processor.slice_audio(
                    config["INPUT_AUDIO"],
                    config["SLICED_DIR"],
                    **(params or {})
                )
                
            elif step == TrainingStep.DENOISE_AUDIO:
                success = await self.step_processor.denoise_audio(
                    config["SLICED_DIR"],
                    config["DENOISED_DIR"],
                    params.get("precision", "float16") if params else "float16"
                )
                
            elif step == TrainingStep.ASR_TRANSCRIBE:
                success = await self.step_processor.asr_transcribe(
                    config["DENOISED_DIR"],
                    config["ASR_OUTPUT"],
                    config["LANGUAGE"],
                    params.get("precision", "float16") if params else "float16"
                )
                
            elif step == TrainingStep.EXTRACT_TEXT_FEATURES:
                # 修复：确保ASR输出指向具体的文件而不是目录
                try:
                    asr_output = self._find_asr_output_file(config["ASR_OUTPUT"])
                    if asr_output != config["ASR_OUTPUT"]:
                        logger.info(f"✅ 找到ASR输出文件: {asr_output}")
                        # 更新配置
                        config["ASR_OUTPUT"] = asr_output
                        self._save_task_config(task_id, config)
                except FileNotFoundError as e:
                    logger.error(f"❌ ASR输出文件查找失败: {e}")
                    task_info.status = TaskStatus.FAILED
                    task_info.error_message = str(e)
                    return
                
                env_vars = self._build_step_env(config)
                gpu_ids = config["GPU_ID"]
                parallel_parts = len(gpu_ids.split('-')) if '-' in gpu_ids else 1
                
                success = await self.step_processor.extract_features(
                    "1-get-text.py", env_vars, parallel_parts, gpu_ids
                )
                
                if success:
                    # 合并分片文件
                    await self.step_processor.merge_feature_files(
                        config["EXP_DIR"], "2-name2text-{}.txt", "2-name2text.txt", 
                        has_header=False, parallel_parts=parallel_parts
                    )
                    
            elif step == TrainingStep.EXTRACT_AUDIO_FEATURES:
                # 修复：确保ASR输出指向具体的文件而不是目录
                try:
                    asr_output = self._find_asr_output_file(config["ASR_OUTPUT"])
                    if asr_output != config["ASR_OUTPUT"]:
                        logger.info(f"✅ 找到ASR输出文件: {asr_output}")
                        # 更新配置
                        config["ASR_OUTPUT"] = asr_output
                        self._save_task_config(task_id, config)
                except FileNotFoundError as e:
                    logger.error(f"❌ ASR输出文件查找失败: {e}")
                    task_info.status = TaskStatus.FAILED
                    task_info.error_message = str(e)
                    return
                
                env_vars = self._build_step_env(config)
                gpu_ids = config["GPU_ID"]
                parallel_parts = len(gpu_ids.split('-')) if '-' in gpu_ids else 1
                
                success = await self.step_processor.extract_features(
                    "2-get-hubert-wav32k.py", env_vars, parallel_parts, gpu_ids
                )
                
            elif step == TrainingStep.EXTRACT_SPEAKER_VECTORS:
                # 修复：确保ASR输出指向具体的文件而不是目录
                try:
                    asr_output = self._find_asr_output_file(config["ASR_OUTPUT"])
                    if asr_output != config["ASR_OUTPUT"]:
                        logger.info(f"✅ 找到ASR输出文件: {asr_output}")
                        # 更新配置
                        config["ASR_OUTPUT"] = asr_output
                        self._save_task_config(task_id, config)
                except FileNotFoundError as e:
                    logger.error(f"❌ ASR输出文件查找失败: {e}")
                    task_info.status = TaskStatus.FAILED
                    task_info.error_message = str(e)
                    return
                
                env_vars = self._build_step_env(config)
                gpu_ids = config["GPU_ID"]
                parallel_parts = len(gpu_ids.split('-')) if '-' in gpu_ids else 1
                
                success = await self.step_processor.extract_features(
                    "2-get-sv.py", env_vars, parallel_parts, gpu_ids
                )
                
            elif step == TrainingStep.EXTRACT_SEMANTIC_FEATURES:
                # 修复：确保ASR输出指向具体的文件而不是目录
                try:
                    asr_output = self._find_asr_output_file(config["ASR_OUTPUT"])
                    if asr_output != config["ASR_OUTPUT"]:
                        logger.info(f"✅ 找到ASR输出文件: {asr_output}")
                        # 更新配置
                        config["ASR_OUTPUT"] = asr_output
                        self._save_task_config(task_id, config)
                except FileNotFoundError as e:
                    logger.error(f"❌ ASR输出文件查找失败: {e}")
                    task_info.status = TaskStatus.FAILED
                    task_info.error_message = str(e)
                    return
                
                env_vars = self._build_step_env(config)
                gpu_ids = config["GPU_ID"]
                parallel_parts = len(gpu_ids.split('-')) if '-' in gpu_ids else 1
                
                success = await self.step_processor.extract_features(
                    "3-get-semantic.py", env_vars, parallel_parts, gpu_ids
                )
                
                if success:
                    # 合并分片文件
                    await self.step_processor.merge_feature_files(
                        config["EXP_DIR"], "6-name2semantic-{}.tsv", "6-name2semantic.tsv", 
                        has_header=True, parallel_parts=parallel_parts
                    )
                    
            elif step == TrainingStep.TRAIN_SOVITS:
                # 生成S2配置文件
                s2_config_path = config["WORK_DIR"] + "/config_s2.json"
                self.config_generator.generate_s2_config(config, s2_config_path)
                
                success = await self.step_processor.train_model(
                    "s2_train.py", s2_config_path
                )
                
            elif step == TrainingStep.TRAIN_GPT:
                # 生成S1配置文件
                s1_config_path = config["WORK_DIR"] + "/config_s1.yaml"
                self.config_generator.generate_s1_config(config, s1_config_path)
                
                env_vars = {"hz": "25hz"}
                success = await self.step_processor.train_model(
                    "s1_train.py", s1_config_path, env_vars
                )
                
            elif step == TrainingStep.TEST_INFERENCE:
                inference_params = params or {}
                inference_params.update({
                    "output_path": config["WORK_DIR"] + "/output",
                    "bert_path": config["BERT_DIR"],
                    "cnhubert_base_path": config["CNHUBERT_DIR"],
                    "gpu_number": config["GPU_ID"],
                    "is_half": True
                })
                
                success = await self.step_processor.test_inference(**inference_params)
                
            else:
                raise ValueError(f"不支持的训练步骤: {step}")
            
            if success:
                task_info.status = TaskStatus.COMPLETED
                self._update_progress(task_id, step)
                logger.info(f"步骤 {step.value} 执行成功，任务 {task_id} 进度更新完成")
            else:
                task_info.status = TaskStatus.FAILED
                task_info.error_message = f"步骤 {step} 执行失败"
                logger.error(f"步骤 {step.value} 执行失败，任务 {task_id} 状态更新为失败")
                
        except Exception as e:
            task_info.status = TaskStatus.FAILED
            task_info.error_message = str(e)
        finally:
            task_info.updated_at = datetime.now()
    
    def _save_task_config(self, task_id: str, config: Dict[str, Any]):
        """保存任务配置"""
        config_file = self.get_task_dir(task_id) / "task_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def _find_asr_output_file(self, asr_output_path: str) -> str:
        """智能查找ASR输出文件"""
        asr_path = Path(asr_output_path)
        
        if asr_path.is_file():
            # 如果已经是文件，直接返回
            return str(asr_path)
        
        if asr_path.is_dir():
            # 如果是目录，按优先级查找文件
            # 优先级1: .list文件（标准格式）
            list_files = list(asr_path.glob("*.list"))
            if list_files:
                return str(list_files[0])
            
            # 优先级2: .txt文件
            txt_files = list(asr_path.glob("*.txt"))
            if txt_files:
                return str(txt_files[0])
            
            # 优先级3: .tsv文件
            tsv_files = list(asr_path.glob("*.tsv"))
            if tsv_files:
                return str(tsv_files[0])
            
            # 优先级4: 任何其他文件
            other_files = list(asr_path.glob("*"))
            if other_files:
                return str(other_files[0])
            
            # 如果都找不到，抛出错误
            raise FileNotFoundError(f"在ASR输出目录中找不到转录文件: {asr_output_path}")
        
        # 如果路径不存在，抛出错误
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
    
    def _update_progress(self, task_id: str, completed_step: TrainingStep):
        """更新任务进度"""
        if task_id not in tasks_db:
            return
        
        # 定义步骤顺序和权重（按实际执行顺序）
        step_sequence = [
            TrainingStep.CONVERT_AUDIO,
            TrainingStep.SLICE_AUDIO,
            TrainingStep.DENOISE_AUDIO,
            TrainingStep.ASR_TRANSCRIBE,
            TrainingStep.EXTRACT_TEXT_FEATURES,
            TrainingStep.EXTRACT_AUDIO_FEATURES,
            TrainingStep.EXTRACT_SPEAKER_VECTORS,
            TrainingStep.EXTRACT_SEMANTIC_FEATURES,
            TrainingStep.TRAIN_SOVITS,
            TrainingStep.TRAIN_GPT,
            TrainingStep.TEST_INFERENCE
        ]
        
        # 定义步骤权重（基于实际复杂度和时间）
        step_weights = {
            TrainingStep.CONVERT_AUDIO: 3,        # 音频转换，相对简单
            TrainingStep.SLICE_AUDIO: 5,          # 音频切片，中等复杂度
            TrainingStep.DENOISE_AUDIO: 8,        # 音频降噪，较复杂
            TrainingStep.ASR_TRANSCRIBE: 10,      # 语音识别，复杂
            TrainingStep.EXTRACT_TEXT_FEATURES: 8, # 文本特征提取，中等复杂度
            TrainingStep.EXTRACT_AUDIO_FEATURES: 12, # 音频特征提取，复杂
            TrainingStep.EXTRACT_SPEAKER_VECTORS: 8, # 说话人向量提取，中等复杂度
            TrainingStep.EXTRACT_SEMANTIC_FEATURES: 10, # 语义特征提取，复杂
            TrainingStep.TRAIN_SOVITS: 20,        # SoVITS训练，最复杂
            TrainingStep.TRAIN_GPT: 20,           # GPT训练，最复杂
            TrainingStep.TEST_INFERENCE: 6        # 推理测试，中等复杂度
        }
        
        # 计算已完成步骤的权重
        total_weight = sum(step_weights.values())
        completed_weight = 0
        
        for step in step_sequence:
            if step == completed_step:
                # 当前步骤完成，加上其权重
                completed_weight += step_weights[step]
                break
            else:
                # 之前步骤已完成，加上其权重
                completed_weight += step_weights[step]
        
        # 更新进度
        progress = min(100.0, (completed_weight / total_weight) * 100)
        tasks_db[task_id].progress = progress
        
        logger.info(f"任务 {task_id} 进度更新: {completed_step.value} 完成，总进度: {progress:.1f}%")
    
    def get_step_progress(self, task_id: str, current_step: TrainingStep) -> float:
        """获取当前步骤的进度百分比"""
        # 定义步骤顺序和权重（与_update_progress保持一致）
        step_sequence = [
            TrainingStep.CONVERT_AUDIO,
            TrainingStep.SLICE_AUDIO,
            TrainingStep.DENOISE_AUDIO,
            TrainingStep.ASR_TRANSCRIBE,
            TrainingStep.EXTRACT_TEXT_FEATURES,
            TrainingStep.EXTRACT_AUDIO_FEATURES,
            TrainingStep.EXTRACT_SPEAKER_VECTORS,
            TrainingStep.EXTRACT_SEMANTIC_FEATURES,
            TrainingStep.TRAIN_SOVITS,
            TrainingStep.TRAIN_GPT,
            TrainingStep.TEST_INFERENCE
        ]
        
        step_weights = {
            TrainingStep.CONVERT_AUDIO: 3,
            TrainingStep.SLICE_AUDIO: 5,
            TrainingStep.DENOISE_AUDIO: 8,
            TrainingStep.ASR_TRANSCRIBE: 10,
            TrainingStep.EXTRACT_TEXT_FEATURES: 8,
            TrainingStep.EXTRACT_AUDIO_FEATURES: 12,
            TrainingStep.EXTRACT_SPEAKER_VECTORS: 8,
            TrainingStep.EXTRACT_SEMANTIC_FEATURES: 10,
            TrainingStep.TRAIN_SOVITS: 20,
            TrainingStep.TRAIN_GPT: 20,
            TrainingStep.TEST_INFERENCE: 6
        }
        
        # 计算当前步骤之前的累计权重
        total_weight = sum(step_weights.values())
        completed_weight = 0
        
        for step in step_sequence:
            if step == current_step:
                # 到达当前步骤，返回之前的累计进度
                break
            else:
                completed_weight += step_weights[step]
        
        # 返回当前步骤之前的累计进度
        return (completed_weight / total_weight) * 100

# 初始化服务
training_service = TrainingService()

# API路由
@app.post("/api/v1/task/create", response_model=TaskInfo)
async def create_task(request: TaskCreateRequest):
    """创建新的训练任务"""
    task_id = str(uuid.uuid4())
    
    # 创建任务信息
    task_info = TaskInfo(
        task_id=task_id,
        task_name=request.task_name,
        status=TaskStatus.CREATED,
        config=request.config,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    # 保存任务
    tasks_db[task_id] = task_info
    
    # 创建任务目录和配置文件
    training_service.create_task_config_file(task_id, request.config)
    
    return task_info

@app.get("/api/v1/task/{task_id}", response_model=TaskInfo)
async def get_task(task_id: str):
    """获取任务信息"""
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="任务不存在")
    return tasks_db[task_id]

@app.get("/api/v1/tasks", response_model=List[TaskInfo])
async def list_tasks():
    """列出所有任务"""
    return list(tasks_db.values())

@app.post("/api/v1/task/{task_id}/step/{step}")
async def execute_step(task_id: str, step: TrainingStep, request: StepExecuteRequest, background_tasks: BackgroundTasks):
    """执行训练步骤"""
    background_tasks.add_task(training_service.execute_step, task_id, step, request.params)
    return {"message": f"步骤 {step} 开始执行", "task_id": task_id}

@app.post("/api/v1/task/{task_id}/cancel")
async def cancel_task(task_id: str):
    """取消任务"""
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    # 终止正在运行的进程
    if task_id in step_processes:
        process = step_processes[task_id]
        process.terminate()
        del step_processes[task_id]
    
    # 更新任务状态
    tasks_db[task_id].status = TaskStatus.CANCELLED
    tasks_db[task_id].updated_at = datetime.now()
    
    return {"message": "任务已取消"}

@app.post("/api/v1/task/{task_id}/files/upload")
async def upload_file(task_id: str, file: UploadFile = File(...)):
    """上传训练音频文件"""
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    # 获取任务输入目录
    input_dir = training_service.get_task_input_dir(task_id)
    file_path = input_dir / file.filename
    
    # 保存文件
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    return {"message": "文件上传成功", "filename": file.filename, "path": str(file_path)}

@app.get("/api/v1/task/{task_id}/files/download/{filename}")
async def download_file(task_id: str, filename: str):
    """下载训练结果文件"""
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    # 查找文件
    task_dir = training_service.get_task_dir(task_id)
    file_path = None
    
    # 在各个可能的目录中查找文件
    search_dirs = [
        task_dir / "output",
        task_dir / "experiments" / tasks_db[task_id].config.exp_name,
        task_dir
    ]
    
    for search_dir in search_dirs:
        candidate_path = search_dir / filename
        if candidate_path.exists():
            file_path = candidate_path
            break
    
    if not file_path:
        raise HTTPException(status_code=404, detail="文件不存在")
    
    return FileResponse(file_path, filename=filename)

@app.get("/api/v1/task/{task_id}/logs")
async def get_task_logs(task_id: str):
    """获取任务日志"""
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    # 读取日志文件
    log_file = training_service.get_task_dir(task_id) / "training.log"
    if log_file.exists():
        with open(log_file, 'r', encoding='utf-8') as f:
            logs = f.read()
        return {"logs": logs}
    else:
        return {"logs": "暂无日志"}

@app.delete("/api/v1/task/{task_id}")
async def delete_task(task_id: str):
    """删除任务"""
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    # 取消正在运行的任务
    if tasks_db[task_id].status == TaskStatus.RUNNING:
        await cancel_task(task_id)
    
    # 删除任务目录
    import shutil
    task_dir = training_service.get_task_dir(task_id)
    if task_dir.exists():
        shutil.rmtree(task_dir)
    
    # 从数据库中删除
    del tasks_db[task_id]
    
    return {"message": "任务已删除"}

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "GPT-SoVITS训练服务API",
        "version": "1.0.0",
        "docs_url": "/docs"
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
