#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成SoVITS训练配置文件
按照webui.py的逻辑生成正确的配置
"""

import json
import sys
import os

def generate_s2_config(
    version="v2ProPlus",
    batch_size=16,
    total_epoch=50,
    exp_name="my_speaker",
    exp_dir=None,
    pretrained_s2G=None,
    pretrained_s2D=None,
    gpu_numbers="0",
    output_path=None
):
    """生成SoVITS训练配置文件"""
    
    # 确定基础配置文件路径
    if version not in {"v2Pro", "v2ProPlus"}:
        config_file = "GPT_SoVITS/configs/s2.json"
    else:
        config_file = f"GPT_SoVITS/configs/s2{version}.json"
    
    # 读取基础配置
    with open(config_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 根据webui.py的逻辑设置训练参数
    data["train"]["batch_size"] = batch_size
    data["train"]["epochs"] = total_epoch
    data["train"]["text_low_lr_rate"] = 0.4
    data["train"]["pretrained_s2G"] = pretrained_s2G
    data["train"]["pretrained_s2D"] = pretrained_s2D
    data["train"]["if_save_latest"] = True
    data["train"]["if_save_every_weights"] = True
    data["train"]["save_every_epoch"] = 4  # 默认保存频率
    data["train"]["gpu_numbers"] = gpu_numbers
    data["train"]["grad_ckpt"] = False
    data["train"]["lora_rank"] = 32  # 对v3/v4有效
    
    # 设置模型版本
    data["model"]["version"] = version
    
    # 设置数据路径
    data["data"]["exp_dir"] = exp_dir
    data["data"]["training_files"] = f"{exp_dir}/2-name2text.txt"
    data["data"]["validation_files"] = f"{exp_dir}/2-name2text.txt"
    # 与 webui.py 一致：将 s2_ckpt_dir 设为实验目录根（s2_dir）
    data["s2_ckpt_dir"] = exp_dir
    
    # 设置实验名称
    data["name"] = exp_name
    data["version"] = version
    
    # 如果没有指定输出路径，打印到stdout
    if output_path is None:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"配置文件已生成: {output_path}")

if __name__ == "__main__":
    # 从环境变量读取参数
    version = os.environ.get("S2_VERSION", "v2ProPlus")
    batch_size = int(os.environ.get("BATCH_SIZE", "16"))
    total_epoch = int(os.environ.get("EPOCHS_S2", "50"))
    exp_name = os.environ.get("exp_name", "my_speaker")
    exp_dir = os.environ.get("opt_dir", "")
    pretrained_s2G = os.environ.get("pretrained_s2G", "")
    pretrained_s2D = os.environ.get("pretrained_s2D", "")
    gpu_numbers = os.environ.get("_CUDA_VISIBLE_DEVICES", "0")
    output_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    generate_s2_config(
        version=version,
        batch_size=batch_size,
        total_epoch=total_epoch,
        exp_name=exp_name,
        exp_dir=exp_dir,
        pretrained_s2G=pretrained_s2G,
        pretrained_s2D=pretrained_s2D,
        gpu_numbers=gpu_numbers,
        output_path=output_path
    )
