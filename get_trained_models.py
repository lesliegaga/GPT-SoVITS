#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
获取训练好的模型路径
复用webui.py中get_weights_names的逻辑，确保与webui.py的处理完全一致
"""

import os
import sys
import json
import re

# 添加当前目录到Python路径，以便导入config模块
sys.path.insert(0, os.getcwd())

try:
    from config import GPT_weight_version2root, SoVITS_weight_version2root
except ImportError:
    print("错误：无法导入config模块，请确保在GPT-SoVITS根目录下运行此脚本")
    sys.exit(1)

def get_final_step_model(directory, file_extension):
    """
    获取指定目录下最终step的训练模型路径
    
    Args:
        directory: 模型目录路径
        file_extension: 文件扩展名（.ckpt或.pth）
    
    Returns:
        str: 最终step的模型文件路径，如果没有找到则返回空字符串
    """
    if not os.path.exists(directory):
        return ""
    
    try:
        # 获取目录下所有指定扩展名的文件
        files = [f for f in os.listdir(directory) if f.endswith(file_extension)]
        if not files:
            return ""
        
        # 解析文件名中的step信息，找到最终的step
        final_model = ""
        max_step = -1
        
        for file in files:
            # 尝试从文件名中提取step信息
            # 支持多种命名格式：
            # - my_speaker-e15.ckpt (epoch格式，GPT模型)
            # - my_speaker_e48_s336.pth (epoch_step格式，SoVITS模型)
            # - my_speaker_step_1000.ckpt (step格式)
            # - my_speaker_1000.ckpt (纯数字格式)
            
            step = -1
            
            # 尝试匹配epoch格式：my_speaker-e15.ckpt (GPT模型)
            epoch_match = re.search(r'-e(\d+)', file)
            if epoch_match:
                step = int(epoch_match.group(1))
            
            # 尝试匹配epoch_step格式：my_speaker_e48_s336.pth (SoVITS模型)
            if step == -1:
                epoch_step_match = re.search(r'_e(\d+)_s(\d+)', file)
                if epoch_step_match:
                    # 对于SoVITS模型，使用step值（第二个数字）作为排序依据
                    step = int(epoch_step_match.group(2))
            
            # 尝试匹配step格式：my_speaker_step_1000.ckpt
            if step == -1:
                step_match = re.search(r'step_(\d+)', file)
                if step_match:
                    step = int(step_match.group(1))
            
            # 尝试匹配纯数字格式：my_speaker_1000.ckpt
            if step == -1:
                num_match = re.search(r'_(\d+)\.', file)
                if num_match:
                    step = int(num_match.group(1))
            
            # 如果找到更大的step，更新最终模型
            if step > max_step:
                max_step = step
                final_model = file
        
        if final_model:
            return os.path.join(directory, final_model)
        else:
            return ""
            
    except Exception as e:
        print(f"警告：读取目录 {directory} 时发生错误: {e}")
        return ""

def get_trained_models():
    """
    获取训练好的模型路径，使用GPT_weight_version2root[version]和SoVITS_weight_version2root[version]目录下最终step的训练模型
    
    Returns:
        tuple: (gpt_model_path, sovits_model_path)
    """
    try:
        # 从环境变量获取版本信息
        version = os.environ.get("S2_VERSION", "v2ProPlus")
        
        # 获取版本对应的权重目录
        gpt_weight_dir = GPT_weight_version2root.get(version, "")
        sovits_weight_dir = SoVITS_weight_version2root.get(version, "")
        
        if not gpt_weight_dir:
            print(f"警告：找不到版本 {version} 对应的GPT权重目录")
        if not sovits_weight_dir:
            print(f"警告：找不到版本 {version} 对应的SoVITS权重目录")
        
        # 获取最终step的模型路径
        gpt_model = get_final_step_model(gpt_weight_dir, ".ckpt")
        sovits_model = get_final_step_model(sovits_weight_dir, ".pth")
        
        # 验证模型文件是否存在
        if gpt_model and not os.path.exists(gpt_model):
            print(f"警告：GPT模型文件不存在: {gpt_model}")
            gpt_model = ""
            
        if sovits_model and not os.path.exists(sovits_model):
            print(f"警告：SoVITS模型文件不存在: {sovits_model}")
            sovits_model = ""
        
        return gpt_model, sovits_model
        
    except Exception as e:
        print(f"错误：获取模型路径时发生异常: {e}")
        return "", ""

def main():
    """主函数，输出JSON格式的模型路径"""
    gpt_model, sovits_model = get_trained_models()
    
    result = {
        "gpt_model": gpt_model,
        "sovits_model": sovits_model,
        "success": bool(gpt_model and sovits_model)
    }
    
    # 输出JSON格式结果，便于shell脚本解析
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # 如果成功获取模型，返回退出码0；否则返回1
    if result["success"]:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
