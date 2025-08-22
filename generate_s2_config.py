#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成SoVITS训练配置文件
按照webui.py中open1Ba的逻辑生成正确的配置
"""

import json
import sys
import os
from config import SoVITS_weight_version2root, exp_root, pretrained_sovits_name

def generate_s2_config(
    version="v2ProPlus",
    batch_size=16,
    total_epoch=50,
    exp_name="my_speaker",
    exp_dir=None,
    pretrained_s2G=None,
    pretrained_s2D=None,
    gpu_numbers="0",
    text_low_lr_rate=0.4,
    if_save_latest=True,
    if_save_every_weights=True,
    save_every_epoch=4,
    if_grad_ckpt=False,
    lora_rank=32,
    is_half=True,
    output_path=None
):
    """生成SoVITS训练配置文件，完全按照webui.py中open1Ba的逻辑"""
    
    # 按照webui.py的逻辑选择基础配置文件
    config_file = (
        "GPT_SoVITS/configs/s2.json"
        if version not in {"v2Pro", "v2ProPlus"}
        else f"GPT_SoVITS/configs/s2{version}.json"
    )
    
    # 加载基础配置文件，与webui.py完全一致
    with open(config_file, 'r', encoding='utf-8') as f:
        data = f.read()
        data = json.loads(data)
    
    # 设置实验目录路径，与webui.py一致
    s2_dir = "%s/%s" % (exp_root, exp_name)
    if exp_dir is not None:
        s2_dir = exp_dir
    
    # 按照webui.py的逻辑处理is_half
    if is_half == False:
        data["train"]["fp16_run"] = False
        batch_size = max(1, batch_size // 2)
    
    # 如果没有指定pretrained_s2G，从config.py中读取（与webui.py逻辑一致）
    if pretrained_s2G is None or pretrained_s2G == "":
        if version in pretrained_sovits_name and pretrained_sovits_name[version]:
            pretrained_s2G = pretrained_sovits_name[version]
            print(f"从config.py读取{version}版本的SoVITS-G预训练模型: {pretrained_s2G}")
        else:
            print(f"警告: 在config.py中找不到{version}版本的SoVITS-G预训练模型")
    
    # 如果没有指定pretrained_s2D，从config.py中读取（与webui.py逻辑一致）
    if pretrained_s2D is None or pretrained_s2D == "":
        if version in pretrained_sovits_name and pretrained_sovits_name[version]:
            pretrained_s2D = pretrained_sovits_name[version].replace("s2G", "s2D")
            print(f"从config.py读取{version}版本的SoVITS-D预训练模型: {pretrained_s2D}")
        else:
            print(f"警告: 在config.py中找不到{version}版本的SoVITS-D预训练模型")
    
    # 按照webui.py的逻辑设置训练参数
    data["train"]["batch_size"] = batch_size
    data["train"]["epochs"] = total_epoch
    data["train"]["text_low_lr_rate"] = text_low_lr_rate
    data["train"]["pretrained_s2G"] = pretrained_s2G
    data["train"]["pretrained_s2D"] = pretrained_s2D
    data["train"]["if_save_latest"] = if_save_latest
    data["train"]["if_save_every_weights"] = if_save_every_weights
    data["train"]["save_every_epoch"] = save_every_epoch
    data["train"]["gpu_numbers"] = gpu_numbers
    data["train"]["grad_ckpt"] = if_grad_ckpt
    data["train"]["lora_rank"] = lora_rank
    data["model"]["version"] = version
    data["data"]["exp_dir"] = data["s2_ckpt_dir"] = s2_dir
    data["save_weight_dir"] = SoVITS_weight_version2root[version]
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
    # 从环境变量读取参数，与webui.py和test_demo.sh一致
    version = os.environ.get("S2_VERSION", "v2ProPlus")
    batch_size = int(os.environ.get("BATCH_SIZE", "16"))
    total_epoch = int(os.environ.get("EPOCHS_S2", "50"))
    exp_name = os.environ.get("exp_name", "my_speaker")
    exp_dir = os.environ.get("opt_dir", "")
    pretrained_s2G = os.environ.get("pretrained_s2G", "")  # 如果为空，会自动从config.py读取
    pretrained_s2D = os.environ.get("pretrained_s2D", "")  # 如果为空，会自动从config.py读取
    gpu_numbers = os.environ.get("_CUDA_VISIBLE_DEVICES", "0")
    output_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    # 从环境变量读取其他参数，使用webui.py中的默认值
    text_low_lr_rate = float(os.environ.get("TEXT_LOW_LR_RATE", "0.4"))
    if_save_latest = os.environ.get("IF_SAVE_LATEST", "True").lower() == "true"
    if_save_every_weights = os.environ.get("IF_SAVE_EVERY_WEIGHTS", "True").lower() == "true"
    save_every_epoch = int(os.environ.get("SAVE_EVERY_EPOCH_S2", "4"))
    if_grad_ckpt = os.environ.get("IF_GRAD_CKPT", "False").lower() == "true"
    lora_rank = int(os.environ.get("LORA_RANK", "32"))
    is_half = os.environ.get("is_half", "True").lower() == "true"
    
    generate_s2_config(
        version=version,
        batch_size=batch_size,
        total_epoch=total_epoch,
        exp_name=exp_name,
        exp_dir=exp_dir,
        pretrained_s2G=pretrained_s2G,
        pretrained_s2D=pretrained_s2D,
        gpu_numbers=gpu_numbers,
        text_low_lr_rate=text_low_lr_rate,
        if_save_latest=if_save_latest,
        if_save_every_weights=if_save_every_weights,
        save_every_epoch=save_every_epoch,
        if_grad_ckpt=if_grad_ckpt,
        lora_rank=lora_rank,
        is_half=is_half,
        output_path=output_path
    )
