#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
获取训练好的模型路径
复用webui.py中get_weights_names的逻辑，确保与webui.py的处理完全一致
"""

import os
import sys
import json

# 添加当前目录到Python路径，以便导入config模块
sys.path.insert(0, os.getcwd())

try:
    from config import get_weights_names
except ImportError:
    print("错误：无法导入config模块，请确保在GPT-SoVITS根目录下运行此脚本")
    sys.exit(1)

def get_trained_models():
    """
    获取训练好的模型路径，完全复用webui.py的逻辑
    
    Returns:
        tuple: (gpt_model_path, sovits_model_path)
    """
    try:
        # 复用webui.py中的get_weights_names函数
        SoVITS_names, GPT_names = get_weights_names()
        
        # 按照webui.py的逻辑选择模型：
        # GPT_names[-1] - 选择最新的GPT模型（列表中的最后一个）
        # SoVITS_names[0] - 选择第一个SoVITS模型
        gpt_model = GPT_names[-1] if GPT_names else ""
        sovits_model = SoVITS_names[0] if SoVITS_names else ""
        
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
