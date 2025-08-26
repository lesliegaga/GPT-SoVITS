#!/usr/bin/env python3
"""
GPT-SoVITS 基于角色的训练服务客户端示例
演示如何通过新的角色管理API进行语音克隆训练流程
"""

import requests
import time
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List

class CharacterBasedGPTSoVITSClient:
    """基于角色的GPT-SoVITS训练服务客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8216"):
        self.base_url = base_url.rstrip('/')
        self.api_base = f"{self.base_url}/api/v1"
        
    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """发送HTTP请求"""
        url = f"{self.api_base}{endpoint}"
        response = requests.request(method, url, **kwargs)
        
        if response.status_code >= 400:
            error_detail = response.json().get('detail', 'Unknown error') if response.content else 'No response'
            raise Exception(f"API错误 ({response.status_code}): {error_detail}")
        
        return response.json() if response.content else {}
    
    # ==================== 角色管理 ====================
    
    def create_character(self, character_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """创建角色"""
        data = {
            "character_name": character_name,
            "config": config
        }
        return self._request("POST", "/characters", json=data)
    
    def list_characters(self) -> List[Dict[str, Any]]:
        """列出所有角色"""
        return self._request("GET", "/characters")
    
    def get_character(self, character_name: str) -> Dict[str, Any]:
        """获取角色信息"""
        return self._request("GET", f"/characters/{character_name}")
    
    def rename_character(self, character_name: str, new_name: str) -> Dict[str, Any]:
        """重命名角色"""
        data = {"new_name": new_name}
        return self._request("PUT", f"/characters/{character_name}", json=data)
    
    def delete_character(self, character_name: str) -> Dict[str, Any]:
        """删除角色"""
        return self._request("DELETE", f"/characters/{character_name}")
    
    def set_default_character(self, character_name: str) -> Dict[str, Any]:
        """设置默认角色"""
        return self._request("POST", f"/characters/{character_name}/set_default")
    
    def get_default_character(self) -> str:
        """获取默认角色"""
        result = self._request("GET", "/default_character")
        return result.get('default_character')
    
    # ==================== 音频管理 ====================
    
    def upload_audio(self, character_name: str, file_path: str) -> Dict[str, Any]:
        """上传音频文件"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        with open(file_path, 'rb') as f:
            files = {'file': (file_path.name, f, 'audio/*')}
            url = f"{self.api_base}/characters/{character_name}/audio/upload"
            response = requests.post(url, files=files)
            
            if response.status_code >= 400:
                error_detail = response.json().get('detail', 'Unknown error')
                raise Exception(f"音频上传失败 ({response.status_code}): {error_detail}")
            
            return response.json()
    
    def start_audio_processing(self, character_name: str) -> Dict[str, Any]:
        """开始音频处理"""
        return self._request("POST", f"/characters/{character_name}/audio/process")
    
    def get_audio_processing_status(self, character_name: str) -> Dict[str, Any]:
        """获取音频处理状态"""
        return self._request("GET", f"/characters/{character_name}/audio/status")
    
    # ==================== 训练管理 ====================
    
    def start_training(self, character_name: str) -> Dict[str, Any]:
        """开始训练"""
        return self._request("POST", f"/characters/{character_name}/training/start")
    
    def get_training_status(self, character_name: str) -> Dict[str, Any]:
        """获取训练状态"""
        return self._request("GET", f"/characters/{character_name}/training/status")
    
    # ==================== 推理管理 ====================
    
    def start_inference(self, target_text: str, character_name: str = None, **kwargs) -> Dict[str, Any]:
        """开始推理"""
        data = {
            "character_name": character_name,
            "target_text": target_text,
            **kwargs
        }
        return self._request("POST", "/inference", json=data)
    
    def get_inference_status(self, inference_id: str) -> Dict[str, Any]:
        """获取推理状态"""
        return self._request("GET", f"/inference/{inference_id}")
    
    def list_inference(self) -> List[Dict[str, Any]]:
        """列出所有推理记录"""
        return self._request("GET", "/inference")
    
    # ==================== 文件下载 ====================
    
    def download_character_file(self, character_name: str, filename: str, save_path: str = None) -> str:
        """下载角色相关文件"""
        url = f"{self.api_base}/characters/{character_name}/download/{filename}"
        response = requests.get(url)
        
        if response.status_code >= 400:
            error_detail = response.json().get('detail', 'Unknown error') if response.content else 'File not found'
            raise Exception(f"文件下载失败 ({response.status_code}): {error_detail}")
        
        if save_path is None:
            save_path = filename
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        return save_path
    
    def download_inference_result(self, inference_id: str, save_path: str = None) -> str:
        """下载推理结果"""
        url = f"{self.api_base}/inference/{inference_id}/download"
        response = requests.get(url)
        
        if response.status_code >= 400:
            error_detail = response.json().get('detail', 'Unknown error') if response.content else 'File not found'
            raise Exception(f"推理结果下载失败 ({response.status_code}): {error_detail}")
        
        if save_path is None:
            save_path = f"inference_{inference_id[:8]}.wav"
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        return save_path
    
    # ==================== 辅助方法 ====================
    
    def wait_for_audio_processing(self, character_name: str, timeout: int = 3600) -> Dict[str, Any]:
        """等待音频处理完成"""
        return self._wait_for_completion(
            lambda: self.get_audio_processing_status(character_name),
            "音频处理",
            timeout
        )
    
    def wait_for_training(self, character_name: str, timeout: int = 7200) -> Dict[str, Any]:
        """等待训练完成"""
        return self._wait_for_completion(
            lambda: self.get_training_status(character_name),
            "训练",
            timeout
        )
    
    def wait_for_inference(self, inference_id: str, timeout: int = 600) -> Dict[str, Any]:
        """等待推理完成"""
        return self._wait_for_completion(
            lambda: self.get_inference_status(inference_id),
            "推理",
            timeout
        )
    
    def _wait_for_completion(self, status_func, task_name: str, timeout: int) -> Dict[str, Any]:
        """通用的等待完成方法"""
        start_time = time.time()
        last_status = None
        
        while time.time() - start_time < timeout:
            try:
                status_info = status_func()
                current_status = status_info['status']
                current_step = status_info.get('current_step', 'unknown')
                progress = status_info.get('progress', 0)
                
                # 打印进度（避免重复打印相同状态）
                status_key = f"{current_status}_{current_step}_{progress}"
                if status_key != last_status:
                    print(f"[{time.strftime('%H:%M:%S')}] {task_name}状态: {current_status} | 步骤: {current_step} | 进度: {progress:.1f}%")
                    last_status = status_key
                
                if current_status in ['completed', 'failed', 'cancelled']:
                    return status_info
                    
                time.sleep(5)  # 每5秒检查一次
                
            except Exception as e:
                print(f"查询{task_name}状态失败: {e}")
                time.sleep(10)
        
        raise TimeoutError(f"等待{task_name}完成超时 ({timeout}秒)")

def example_character_based_training():
    """基于角色的完整训练流程示例"""
    client = CharacterBasedGPTSoVITSClient()
    
    print("=== GPT-SoVITS 基于角色的训练流程示例 ===")
    
    # 1. 创建角色
    print("\n1. 创建训练角色...")
    character_name = "demo_speaker"
    config = {
        "character_name": character_name,
        "language": "zh",
        "batch_size": 16,
        "epochs_s2": 50,
        "epochs_s1": 15,
        "gpu_id": "0",
        "enable_denoise": True
    }
    
    try:
        character = client.create_character(character_name, config)
        print(f"角色创建成功: {character['character_name']}")
        print(f"创建时间: {character['created_at']}")
    except Exception as e:
        print(f"角色创建失败: {e}")
        # 如果角色已存在，继续使用
        try:
            character = client.get_character(character_name)
            print(f"使用现有角色: {character['character_name']}")
        except Exception as e2:
            print(f"获取角色信息失败: {e2}")
            return
    
    # 2. 上传音频文件
    print("\n2. 上传训练音频...")
    audio_dir = "input_audio/test"  # 示例音频目录
    
    if os.path.exists(audio_dir):
        audio_files = []
        for ext in ['wav', 'mp3', 'm4a', 'flac']:
            audio_files.extend(Path(audio_dir).glob(f"*.{ext}"))
        
        if audio_files:
            for audio_file in audio_files[:5]:  # 最多上传5个文件进行演示
                try:
                    result = client.upload_audio(character_name, str(audio_file))
                    print(f"音频上传成功: {result['filename']}")
                except Exception as e:
                    print(f"音频上传失败 {audio_file.name}: {e}")
        else:
            print(f"警告: 在 {audio_dir} 中未找到音频文件")
            print("请将音频文件放在该目录中，或修改audio_dir变量")
            return
    else:
        print(f"警告: 音频目录不存在: {audio_dir}")
        print("请创建目录并放入音频文件，或修改audio_dir变量")
        return
    
    # 3. 开始音频处理
    print("\n3. 开始音频处理...")
    try:
        processing_info = client.start_audio_processing(character_name)
        print(f"音频处理启动: {processing_info['status']}")
        
        # 等待音频处理完成
        final_processing = client.wait_for_audio_processing(character_name)
        
        if final_processing['status'] == 'completed':
            print(f"✅ 音频处理完成！处理了 {final_processing['processed_audio_count']} 个音频文件")
        else:
            print(f"❌ 音频处理失败: {final_processing.get('error_message', '未知错误')}")
            return
            
    except Exception as e:
        print(f"音频处理异常: {e}")
        return
    
    # 4. 开始模型训练
    print("\n4. 开始模型训练...")
    try:
        training_info = client.start_training(character_name)
        print(f"模型训练启动: {training_info['status']}")
        
        # 等待训练完成
        final_training = client.wait_for_training(character_name)
        
        if final_training['status'] == 'completed':
            print("✅ 模型训练完成！")
            print(f"GPT模型: {final_training.get('gpt_model_path', '未找到')}")
            print(f"SoVITS模型: {final_training.get('sovits_model_path', '未找到')}")
        else:
            print(f"❌ 模型训练失败: {final_training.get('error_message', '未知错误')}")
            return
            
    except Exception as e:
        print(f"模型训练异常: {e}")
        return
    
    # 5. 设置为默认角色
    print("\n5. 设置默认角色...")
    try:
        result = client.set_default_character(character_name)
        print(f"默认角色设置成功: {result['message']}")
    except Exception as e:
        print(f"设置默认角色失败: {e}")
    
    # 6. 进行推理测试
    print("\n6. 推理测试...")
    try:
        inference_request = {
            "target_text": "这是一个测试文本，用于验证训练好的模型效果。",
            "ref_language": "中文",
            "target_language": "中文"
        }
        
        inference_info = client.start_inference(**inference_request)
        inference_id = inference_info['inference_id']
        print(f"推理测试启动: {inference_id}")
        
        # 等待推理完成
        final_inference = client.wait_for_inference(inference_id)
        
        if final_inference['status'] == 'completed':
            print("✅ 推理测试完成！")
            
            # 下载推理结果
            try:
                output_file = client.download_inference_result(inference_id, "generated_audio.wav")
                print(f"推理结果已下载: {output_file}")
            except Exception as e:
                print(f"推理结果下载失败: {e}")
        else:
            print(f"❌ 推理测试失败: {final_inference.get('error_message', '未知错误')}")
            
    except Exception as e:
        print(f"推理测试异常: {e}")
    
    print(f"\n✅ 完整流程完成！角色 '{character_name}' 已完成训练并可用于推理。")

def example_character_management():
    """角色管理示例"""
    client = CharacterBasedGPTSoVITSClient()
    
    print("=== 角色管理示例 ===")
    
    # 列出现有角色
    print("\n1. 列出现有角色...")
    characters = client.list_characters()
    print(f"现有角色数量: {len(characters)}")
    
    if characters:
        print("角色列表:")
        for char in characters:
            status = "✅ 默认" if char['is_default'] else ""
            print(f"  - {char['character_name']} | 音频数量: {char['audio_count']} | "
                  f"音频处理: {char['audio_processing_status']} | "
                  f"训练状态: {char['training_status']} {status}")
    
    # 创建新角色示例
    print("\n2. 创建新角色...")
    new_character_name = "test_character"
    config = {
        "character_name": new_character_name,
        "language": "zh",
        "batch_size": 8,
        "epochs_s2": 30,
        "epochs_s1": 10,
        "gpu_id": "0",
        "enable_denoise": False
    }
    
    try:
        character = client.create_character(new_character_name, config)
        print(f"角色创建成功: {character['character_name']}")
        
        # 重命名角色
        print("\n3. 重命名角色...")
        new_name = "renamed_character"
        renamed_char = client.rename_character(new_character_name, new_name)
        print(f"角色重命名成功: {renamed_char['character_name']}")
        
        # 删除角色
        print("\n4. 删除角色...")
        result = client.delete_character(new_name)
        print(f"角色删除: {result['message']}")
        
    except Exception as e:
        print(f"角色管理操作失败: {e}")
    
    # 设置默认角色
    if characters:
        print("\n5. 设置默认角色...")
        try:
            first_char = characters[0]['character_name']
            result = client.set_default_character(first_char)
            print(f"默认角色设置: {result['message']}")
            
            default_char = client.get_default_character()
            print(f"当前默认角色: {default_char}")
        except Exception as e:
            print(f"设置默认角色失败: {e}")

def example_inference_only():
    """仅推理示例（使用已训练好的模型）"""
    client = CharacterBasedGPTSoVITSClient()
    
    print("=== 推理示例 ===")
    
    # 获取默认角色
    try:
        default_char = client.get_default_character()
        if not default_char:
            print("❌ 未设置默认角色，请先完成训练流程")
            return
        
        print(f"使用默认角色进行推理: {default_char}")
        
        # 进行多次推理测试
        test_texts = [
            "欢迎使用GPT-SoVITS语音克隆系统。",
            "这是第二个测试文本，用于验证模型的稳定性。",
            "今天天气真不错，适合出去散步。"
        ]
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n{i}. 推理测试: {text}")
            
            try:
                inference_info = client.start_inference(
                    target_text=text,
                    character_name=default_char,  # 明确指定角色
                    ref_language="中文",
                    target_language="中文"
                )
                
                inference_id = inference_info['inference_id']
                print(f"推理ID: {inference_id}")
                
                # 等待推理完成
                final_result = client.wait_for_inference(inference_id)
                
                if final_result['status'] == 'completed':
                    # 下载结果
                    output_file = f"inference_result_{i}.wav"
                    client.download_inference_result(inference_id, output_file)
                    print(f"✅ 推理完成，结果保存为: {output_file}")
                else:
                    print(f"❌ 推理失败: {final_result.get('error_message')}")
                    
            except Exception as e:
                print(f"推理异常: {e}")
                
    except Exception as e:
        print(f"获取默认角色失败: {e}")

def main():
    """主函数"""
    print("GPT-SoVITS 基于角色的训练服务客户端示例")
    print("1. 完整训练流程演示（角色创建 → 音频处理 → 模型训练 → 推理测试）")
    print("2. 角色管理演示")
    print("3. 仅推理演示（使用已训练好的模型）")
    print("4. 退出")
    
    while True:
        try:
            choice = input("\n请选择 (1-4): ").strip()
            
            if choice == '1':
                example_character_based_training()
                break
            elif choice == '2':
                example_character_management()
                break
            elif choice == '3':
                example_inference_only()
                break
            elif choice == '4':
                print("退出")
                break
            else:
                print("无效选择，请输入 1-4")
                
        except KeyboardInterrupt:
            print("\n\n用户中断，退出")
            break
        except Exception as e:
            print(f"执行出错: {e}")
            break

if __name__ == "__main__":
    main()
