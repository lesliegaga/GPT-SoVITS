#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成GPT训练配置文件
按照webui.py中open1Bb的逻辑生成正确的配置
"""

import yaml
import sys
import os
from config import GPT_weight_version2root, exp_root, pretrained_gpt_name

def generate_s1_config(
    version="v2ProPlus",
    batch_size=16,
    total_epoch=15,
    exp_name="my_speaker",
    exp_dir=None,
    pretrained_s1=None,
    gpu_numbers="0",
    save_every_epoch=5,
    if_save_latest=True,
    if_save_every_weights=True,
    if_dpo=False,
    is_half=True,
    output_path=None
):
    """生成GPT训练配置文件，完全按照webui.py中open1Bb的逻辑"""
    
    # 按照webui.py的逻辑选择基础配置文件
    config_file = (
        "GPT_SoVITS/configs/s1longer.yaml" 
        if version == "v1" 
        else "GPT_SoVITS/configs/s1longer-v2.yaml"
    )
    
    # 加载基础配置文件，与webui.py完全一致
    with open(config_file) as f:
        data = f.read()
        data = yaml.load(data, Loader=yaml.FullLoader)
    
    # 设置实验目录路径，与webui.py一致
    s1_dir = "%s/%s" % (exp_root, exp_name)
    if exp_dir is not None:
        s1_dir = exp_dir
    
    # 按照webui.py的逻辑处理is_half
    if is_half == False:
        data["train"]["precision"] = "32"
        batch_size = max(1, batch_size // 2)
    
    # 如果没有指定pretrained_s1，从config.py中读取（与webui.py逻辑一致）
    if pretrained_s1 is None or pretrained_s1 == "":
        if version in pretrained_gpt_name and pretrained_gpt_name[version]:
            pretrained_s1 = pretrained_gpt_name[version]
            print(f"从config.py读取{version}版本的GPT预训练模型: {pretrained_s1}")
        else:
            print(f"警告: 在config.py中找不到{version}版本的GPT预训练模型")
    
    # 按照webui.py的逻辑设置训练参数
    data["train"]["batch_size"] = batch_size
    data["train"]["epochs"] = total_epoch
    data["pretrained_s1"] = pretrained_s1
    data["train"]["save_every_n_epoch"] = save_every_epoch
    data["train"]["if_save_every_weights"] = if_save_every_weights
    data["train"]["if_save_latest"] = if_save_latest
    data["train"]["if_dpo"] = if_dpo
    data["train"]["half_weights_save_dir"] = GPT_weight_version2root[version]
    data["train"]["exp_name"] = exp_name
    data["train_semantic_path"] = "%s/6-name2semantic.tsv" % s1_dir
    data["train_phoneme_path"] = "%s/2-name2text.txt" % s1_dir
    data["output_dir"] = "%s/logs_s1_%s" % (s1_dir, version)
    
    # 如果没有指定输出路径，打印到stdout
    if output_path is None:
        print(yaml.dump(data, default_flow_style=False, allow_unicode=True))
    else:
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        print(f"配置文件已生成: {output_path}")

if __name__ == "__main__":
    # 从环境变量读取参数，与webui.py和test_demo.sh一致
    version = os.environ.get("S2_VERSION", "v2ProPlus")
    batch_size = int(os.environ.get("BATCH_SIZE", "16"))
    total_epoch = int(os.environ.get("EPOCHS_S1", "15"))
    exp_name = os.environ.get("exp_name", "my_speaker")
    exp_dir = os.environ.get("opt_dir", "")
    pretrained_s1 = os.environ.get("pretrained_s1", "")  # 如果为空，会自动从config.py读取
    gpu_numbers = os.environ.get("_CUDA_VISIBLE_DEVICES", "0")
    output_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    # 从环境变量读取其他参数，使用webui.py中的默认值
    save_every_epoch = int(os.environ.get("SAVE_EVERY_EPOCH_S1", "5"))
    if_save_latest = os.environ.get("IF_SAVE_LATEST", "True").lower() == "true"
    if_save_every_weights = os.environ.get("IF_SAVE_EVERY_WEIGHTS", "True").lower() == "true"
    if_dpo = os.environ.get("IF_DPO", "False").lower() == "true"
    is_half = os.environ.get("is_half", "True").lower() == "true"
    
    generate_s1_config(
        version=version,
        batch_size=batch_size,
        total_epoch=total_epoch,
        exp_name=exp_name,
        exp_dir=exp_dir,
        pretrained_s1=pretrained_s1,
        gpu_numbers=gpu_numbers,
        save_every_epoch=save_every_epoch,
        if_save_latest=if_save_latest,
        if_save_every_weights=if_save_every_weights,
        if_dpo=if_dpo,
        is_half=is_half,
        output_path=output_path
    )