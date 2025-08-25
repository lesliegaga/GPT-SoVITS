#!/usr/bin/env python3
"""
GPT-SoVITS 训练服务客户端示例
演示如何通过API进行完整的语音克隆训练流程
"""

import requests
import time
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

class GPTSoVITSClient:
    """GPT-SoVITS训练服务客户端"""
    
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
    
    def create_task(self, task_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """创建训练任务"""
        data = {
            "task_name": task_name,
            "config": config
        }
        return self._request("POST", "/task/create", json=data)
    
    def get_task(self, task_id: str) -> Dict[str, Any]:
        """获取任务信息"""
        return self._request("GET", f"/task/{task_id}")
    
    def list_tasks(self) -> list:
        """列出所有任务"""
        return self._request("GET", "/tasks")
    
    def upload_file(self, task_id: str, file_path: str) -> Dict[str, Any]:
        """上传文件"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        with open(file_path, 'rb') as f:
            files = {'file': (file_path.name, f, 'audio/wav')}
            url = f"{self.api_base}/task/{task_id}/files/upload"
            response = requests.post(url, files=files)
            
            if response.status_code >= 400:
                error_detail = response.json().get('detail', 'Unknown error')
                raise Exception(f"文件上传失败 ({response.status_code}): {error_detail}")
            
            return response.json()
    
    def download_file(self, task_id: str, filename: str, save_path: str = None) -> str:
        """下载文件"""
        url = f"{self.api_base}/task/{task_id}/files/download/{filename}"
        response = requests.get(url)
        
        if response.status_code >= 400:
            error_detail = response.json().get('detail', 'Unknown error') if response.content else 'File not found'
            raise Exception(f"文件下载失败 ({response.status_code}): {error_detail}")
        
        if save_path is None:
            save_path = filename
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        return save_path
    
    def execute_step(self, task_id: str, step_name: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """执行训练步骤"""
        data = {"params": params or {}}
        return self._request("POST", f"/task/{task_id}/step/{step_name}", json=data)
    
    def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """取消任务"""
        return self._request("POST", f"/task/{task_id}/cancel")
    
    def delete_task(self, task_id: str) -> Dict[str, Any]:
        """删除任务"""
        return self._request("DELETE", f"/task/{task_id}")
    
    def get_logs(self, task_id: str) -> str:
        """获取任务日志"""
        result = self._request("GET", f"/task/{task_id}/logs")
        return result.get('logs', '')
    
    def wait_for_step_completion(self, task_id: str, timeout: int = 3600) -> Dict[str, Any]:
        """等待当前步骤完成"""
        start_time = time.time()
        last_status = None
        
        while time.time() - start_time < timeout:
            try:
                task_info = self.get_task(task_id)
                current_status = task_info['status']
                current_step = task_info.get('current_step', 'unknown')
                progress = task_info.get('progress', 0)
                
                # 打印进度（避免重复打印相同状态）
                status_key = f"{current_status}_{current_step}_{progress}"
                if status_key != last_status:
                    print(f"[{time.strftime('%H:%M:%S')}] 状态: {current_status} | 步骤: {current_step} | 进度: {progress:.1f}%")
                    last_status = status_key
                
                if current_status in ['completed', 'failed', 'cancelled']:
                    return task_info
                    
                time.sleep(5)  # 每5秒检查一次
                
            except Exception as e:
                print(f"查询状态失败: {e}")
                time.sleep(10)
        
        raise TimeoutError(f"等待步骤完成超时 ({timeout}秒)")

def example_full_training():
    """完整训练流程示例"""
    client = GPTSoVITSClient()
    
    print("=== GPT-SoVITS 完整训练流程示例 ===")
    
    # 1. 创建任务
    print("\n1. 创建训练任务...")
    task_config = {
        "exp_name": "demo_speaker",
        "language": "zh",
        "batch_size": 8,  # 小批次用于快速测试
        "epochs_s2": 20,  # 减少训练轮数
        "epochs_s1": 8,
        "gpu_id": "0"
    }
    
    task = client.create_task("语音克隆演示", task_config)
    task_id = task['task_id']
    print(f"任务创建成功: {task_id}")
    print(f"任务名称: {task['task_name']}")
    
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
                    result = client.upload_file(task_id, str(audio_file))
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
    
    # 3. 执行训练步骤
    print("\n3. 开始训练流程...")
    training_steps = [
        "convert_audio",
        "slice_audio",
        "denoise_audio",
        "asr_transcribe",
        "extract_text_features",
        "extract_audio_features",
        "extract_speaker_vectors", 
        "extract_semantic_features",
        "train_sovits",
        "train_gpt"
    ]
    
    for i, step in enumerate(training_steps, 1):
        print(f"\n--- 步骤 {i}/{len(training_steps)}: {step} ---")
        
        try:
            # 启动步骤
            result = client.execute_step(task_id, step)
            print(f"步骤启动: {result['message']}")
            
            # 等待完成
            final_status = client.wait_for_step_completion(task_id)
            
            if final_status['status'] == 'failed':
                print(f"❌ 步骤失败: {final_status.get('error_message', '未知错误')}")
                break
            elif final_status['status'] == 'completed':
                print(f"✅ 步骤完成: {step}")
            else:
                print(f"⚠️  步骤状态异常: {final_status['status']}")
                break
                
        except Exception as e:
            print(f"❌ 步骤执行异常: {e}")
            break
    
    # 4. 检查最终状态
    print("\n4. 检查训练结果...")
    final_task = client.get_task(task_id)
    print(f"最终状态: {final_task['status']}")
    print(f"最终进度: {final_task['progress']:.1f}%")
    
    if final_task['status'] == 'completed':
        print("🎉 训练完成！可以进行推理测试了")
        
        # 可选：进行推理测试
        print("\n5. 推理测试...")
        try:
            inference_params = {
                "target_text": "这是一个测试文本，验证训练效果。"
            }
            
            result = client.execute_step(task_id, "test_inference", inference_params)
            print(f"推理测试启动: {result['message']}")
            
            final_status = client.wait_for_step_completion(task_id)
            if final_status['status'] == 'completed':
                print("✅ 推理测试完成！")
                print("可以下载生成的音频文件：")
                try:
                    client.download_file(task_id, "output.wav", "generated_audio.wav")
                    print("音频文件已下载: generated_audio.wav")
                except Exception as e:
                    print(f"音频下载失败: {e}")
            else:
                print(f"❌ 推理测试失败: {final_status.get('error_message')}")
                
        except Exception as e:
            print(f"推理测试异常: {e}")
    
    else:
        print("❌ 训练未完成")
        if final_task.get('error_message'):
            print(f"错误信息: {final_task['error_message']}")
    
    # 显示日志
    print("\n=== 任务日志 ===")
    try:
        logs = client.get_logs(task_id)
        if logs:
            print(logs[-1000:])  # 显示最后1000个字符
        else:
            print("暂无日志")
    except Exception as e:
        print(f"获取日志失败: {e}")
    
    print(f"\n训练任务ID: {task_id}")
    print("您可以通过以下方式管理任务：")
    print(f"- 查看状态: client.get_task('{task_id}')")
    print(f"- 下载文件: client.download_file('{task_id}', 'filename')")
    print(f"- 删除任务: client.delete_task('{task_id}')")

def example_step_by_step():
    """分步执行示例"""
    client = GPTSoVITSClient()
    
    print("=== 分步执行示例 ===")
    
    # 列出现有任务
    tasks = client.list_tasks()
    print(f"现有任务数量: {len(tasks)}")
    
    if tasks:
        print("最近的任务:")
        for task in tasks[-3:]:  # 显示最近3个任务
            print(f"  {task['task_id'][:8]}... - {task['task_name']} - {task['status']}")
        
        # 使用最新的任务进行演示
        latest_task = tasks[-1]
        task_id = latest_task['task_id']
        print(f"\n使用任务: {task_id[:8]}... - {latest_task['task_name']}")
        
        # 执行单个步骤
        print("\n执行音频切片步骤...")
        try:
            result = client.execute_step(task_id, "slice_audio", {
                "min_length": -30,
                "min_interval": 3000
            })
            print(f"步骤启动: {result['message']}")
            
            # 等待完成
            final_status = client.wait_for_step_completion(task_id, timeout=600)
            print(f"步骤结果: {final_status['status']}")
            
        except Exception as e:
            print(f"步骤执行失败: {e}")
    
    else:
        print("没有现有任务，请先运行 example_full_training()")

def main():
    """主函数"""
    print("GPT-SoVITS 训练服务客户端示例")
    print("1. 完整训练流程演示")
    print("2. 分步执行演示")
    print("3. 退出")
    
    while True:
        try:
            choice = input("\n请选择 (1-3): ").strip()
            
            if choice == '1':
                example_full_training()
                break
            elif choice == '2':
                example_step_by_step()
                break
            elif choice == '3':
                print("退出")
                break
            else:
                print("无效选择，请输入 1-3")
                
        except KeyboardInterrupt:
            print("\n\n用户中断，退出")
            break
        except Exception as e:
            print(f"执行出错: {e}")
            break

if __name__ == "__main__":
    main()
