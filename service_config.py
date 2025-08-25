#!/usr/bin/env python3
"""
GPT-SoVITS 训练服务配置文件
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# 服务配置
SERVICE_CONFIG = {
    # 基础路径配置
    "base_path": os.getenv("GPT_SOVITS_BASE_PATH", "~/workspace/git/GPT-SoVITS"),
    
    # 服务配置
    "host": os.getenv("SERVICE_HOST", "0.0.0.0"),
    "port": int(os.getenv("SERVICE_PORT", "8000")),
    "workers": int(os.getenv("SERVICE_WORKERS", "1")),
    "log_level": os.getenv("SERVICE_LOG_LEVEL", "info"),
    
    # 工作目录配置
    "work_dir_name": os.getenv("WORK_DIR_NAME", "api_tasks"),
    
    # 预训练模型路径
    "bert_dir": os.getenv("BERT_DIR", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"),
    "cnhubert_dir": os.getenv("CNHUBERT_DIR", "GPT_SoVITS/pretrained_models/chinese-hubert-base"),
    "pretrained_sv": os.getenv("PRETRAINED_SV", "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"),
    
    # 训练配置
    "default_batch_size": int(os.getenv("DEFAULT_BATCH_SIZE", "16")),
    "default_epochs_s2": int(os.getenv("DEFAULT_EPOCHS_S2", "50")),
    "default_epochs_s1": int(os.getenv("DEFAULT_EPOCHS_S1", "15")),
    "default_gpu_id": os.getenv("DEFAULT_GPU_ID", "0"),
    "default_language": os.getenv("DEFAULT_LANGUAGE", "zh"),
}

def resolve_path(path_str: str) -> Path:
    """解析路径字符串，支持 ~ 展开和相对路径"""
    if path_str.startswith("~"):
        # 展开 ~ 符号
        return Path.home() / path_str[1:].lstrip("/")
    elif path_str.startswith("/"):
        # 绝对路径
        return Path(path_str)
    else:
        # 相对路径，相对于当前工作目录
        return Path.cwd() / path_str

def get_base_path() -> Path:
    """获取GPT-SoVITS基础路径"""
    # 首先加载环境变量
    load_dotenv()
    
    base_path = resolve_path(SERVICE_CONFIG["base_path"])
    
    # 检查路径是否存在
    if not base_path.exists():
        # 尝试一些常见的路径
        common_paths = [
            Path.cwd(),  # 当前目录
            Path.home() / "workspace/git/GPT-SoVITS",
            Path.home() / "Documents/GitHub/GPT-SoVITS",
            Path.home() / "git/GPT-SoVITS",
            Path("/home/tongyan.zjy/workspace/git/GPT-SoVITS"),  # 用户提到的路径
        ]
        
        for common_path in common_paths:
            if common_path.exists():
                print(f"⚠️  配置的路径不存在: {base_path}")
                print(f"✅ 使用发现的路径: {common_path}")
                return common_path
        
        # 如果都找不到，使用当前目录
        print(f"⚠️  无法找到GPT-SoVITS目录，使用当前目录: {Path.cwd()}")
        return Path.cwd()
    
    return base_path

def get_work_dir() -> Path:
    """获取工作目录"""
    base_path = get_base_path()
    work_dir = base_path / SERVICE_CONFIG["work_dir_name"]
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir

def get_model_paths() -> dict:
    """获取预训练模型路径"""
    base_path = get_base_path()
    return {
        "bert_dir": str(base_path / SERVICE_CONFIG["bert_dir"]),
        "cnhubert_dir": str(base_path / SERVICE_CONFIG["cnhubert_dir"]),
        "pretrained_sv": str(base_path / SERVICE_CONFIG["pretrained_sv"]),
    }

def print_config():
    """打印当前配置"""
    print("🔧 GPT-SoVITS 训练服务配置:")
    print(f"   基础路径: {get_base_path()}")
    print(f"   工作目录: {get_work_dir()}")
    print(f"   服务地址: {SERVICE_CONFIG['host']}:{SERVICE_CONFIG['port']}")
    print(f"   工作进程: {SERVICE_CONFIG['workers']}")
    print(f"   日志级别: {SERVICE_CONFIG['log_level']}")
    
    model_paths = get_model_paths()
    print("   预训练模型:")
    for key, path in model_paths.items():
        exists = "✅" if Path(path).exists() else "❌"
        print(f"     {key}: {exists} {path}")

if __name__ == "__main__":
    print_config()
