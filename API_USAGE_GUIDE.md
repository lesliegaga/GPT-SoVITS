# GPT-SoVITS 训练服务 API 使用指南

## 概述

GPT-SoVITS训练服务API提供了完整的语音克隆训练流程接口，将原始的`test_demo.sh`脚本功能封装为RESTful API，支持分步骤执行、进度监控和文件管理。

## 功能特点

- ✅ **分步骤训练**: 将完整训练流程拆分为11个独立的步骤，可单独调用
- ✅ **进度监控**: 实时跟踪训练进度和状态
- ✅ **并行处理**: 支持GPU并行加速特征提取
- ✅ **文件管理**: 提供文件上传下载接口
- ✅ **错误处理**: 完善的错误处理和日志记录
- ✅ **任务管理**: 支持多任务并发和任务取消

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install fastapi uvicorn python-multipart

# 确保FFmpeg已安装（用于音频格式转换）
# macOS: brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg
```

### 2. 启动服务

```bash
# 在GPT-SoVITS项目根目录下执行
python training_service.py
```

服务默认运行在 `http://localhost:8000`

### 3. 访问API文档

打开浏览器访问 `http://localhost:8000/docs` 查看Swagger UI文档

## API 接口详解

### 任务管理

#### 创建训练任务

```http
POST /api/v1/task/create
Content-Type: application/json

{
    "task_name": "我的语音克隆任务",
    "config": {
        "exp_name": "my_speaker",
        "language": "zh",
        "batch_size": 16,
        "epochs_s2": 50,
        "epochs_s1": 15,
        "gpu_id": "0"
    }
}
```

**响应示例**:
```json
{
    "task_id": "123e4567-e89b-12d3-a456-426614174000",
    "task_name": "我的语音克隆任务",
    "status": "created",
    "config": { ... },
    "created_at": "2024-01-15T10:30:00",
    "updated_at": "2024-01-15T10:30:00",
    "progress": 0.0
}
```

#### 查询任务状态

```http
GET /api/v1/task/{task_id}
```

#### 列出所有任务

```http
GET /api/v1/tasks
```

### 文件管理

#### 上传训练音频

```http
POST /api/v1/task/{task_id}/files/upload
Content-Type: multipart/form-data

file: [音频文件]
```

支持的音频格式：WAV, MP3, M4A, AAC, FLAC

#### 下载结果文件

```http
GET /api/v1/task/{task_id}/files/download/{filename}
```

### 训练步骤执行

#### 训练步骤列表

1. **convert_audio** - 音频格式转换
2. **slice_audio** - 音频切片
3. **denoise_audio** - 音频降噪
4. **asr_transcribe** - 语音识别
5. **extract_text_features** - 文本特征提取
6. **extract_audio_features** - 音频特征提取
7. **extract_speaker_vectors** - 说话人向量提取
8. **extract_semantic_features** - 语义特征提取
9. **train_sovits** - SoVITS模型训练
10. **train_gpt** - GPT模型训练
11. **test_inference** - 推理测试

#### 执行训练步骤

```http
POST /api/v1/task/{task_id}/step/{step_name}
Content-Type: application/json

{
    "params": {
        // 步骤特定参数（可选）
    }
}
```

**示例 - 执行音频切片**:
```http
POST /api/v1/task/{task_id}/step/slice_audio
Content-Type: application/json

{
    "params": {
        "min_length": -34,
        "min_interval": 4000,
        "hop_size": 300
    }
}
```

#### 取消任务

```http
POST /api/v1/task/{task_id}/cancel
```

#### 删除任务

```http
DELETE /api/v1/task/{task_id}
```

## 完整训练流程示例

以下是使用Python进行完整训练的示例代码：

```python
import requests
import time
import json

# API基础URL
BASE_URL = "http://localhost:8000/api/v1"

class TrainingClient:
    def __init__(self, base_url=BASE_URL):
        self.base_url = base_url
        
    def create_task(self, task_name, config):
        """创建训练任务"""
        response = requests.post(
            f"{self.base_url}/task/create",
            json={"task_name": task_name, "config": config}
        )
        return response.json()
    
    def upload_audio(self, task_id, audio_file_path):
        """上传音频文件"""
        with open(audio_file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{self.base_url}/task/{task_id}/files/upload",
                files=files
            )
        return response.json()
    
    def execute_step(self, task_id, step_name, params=None):
        """执行训练步骤"""
        data = {"params": params or {}}
        response = requests.post(
            f"{self.base_url}/task/{task_id}/step/{step_name}",
            json=data
        )
        return response.json()
    
    def get_task_status(self, task_id):
        """查询任务状态"""
        response = requests.get(f"{self.base_url}/task/{task_id}")
        return response.json()
    
    def wait_for_completion(self, task_id, timeout=3600):
        """等待任务完成"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_task_status(task_id)
            if status['status'] in ['completed', 'failed']:
                return status
            print(f"进度: {status['progress']:.1f}% - {status['current_step']}")
            time.sleep(10)
        raise TimeoutError("任务执行超时")

# 使用示例
def main():
    client = TrainingClient()
    
    # 1. 创建任务
    task_config = {
        "exp_name": "my_speaker",
        "language": "zh",
        "batch_size": 16,
        "epochs_s2": 20,  # 降低训练轮数以加快测试
        "epochs_s1": 8,
        "gpu_id": "0"
    }
    
    task = client.create_task("语音克隆测试", task_config)
    task_id = task['task_id']
    print(f"任务已创建: {task_id}")
    
    # 2. 上传音频文件
    audio_files = [
        "audio1.wav",
        "audio2.wav", 
        "audio3.wav"
    ]
    
    for audio_file in audio_files:
        result = client.upload_audio(task_id, audio_file)
        print(f"音频文件上传成功: {result['filename']}")
    
    # 3. 执行训练步骤
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
        "train_gpt",
        "test_inference"
    ]
    
    for step in training_steps:
        print(f"开始执行步骤: {step}")
        result = client.execute_step(task_id, step)
        print(f"步骤启动: {result['message']}")
        
        # 等待步骤完成
        final_status = client.wait_for_completion(task_id)
        if final_status['status'] == 'failed':
            print(f"步骤失败: {final_status['error_message']}")
            break
        else:
            print(f"步骤完成: {step}")
    
    print("训练完成！")

if __name__ == "__main__":
    main()
```

## 步骤参数详解

### 音频切片 (slice_audio)

```json
{
    "params": {
        "min_length": -34,      // 最小长度阈值
        "min_interval": 4000,   // 最小间隔
        "hop_size": 300,        // 跳跃大小
        "max_sil_kept": 10,     // 最大静音保留
        "_max": 500,            // 最大值
        "alpha": 0.9,           // Alpha参数
        "n_parts": 0.25,        // 分片数量
        "is_speech": 0,         // 是否为语音
        "num_processes": 1      // 进程数
    }
}
```

### 音频降噪 (denoise_audio)

```json
{
    "params": {
        "precision": "float16"  // 精度设置: float16/float32
    }
}
```

### 语音识别 (asr_transcribe)

```json
{
    "params": {
        "precision": "float16"  // 精度设置
    }
}
```

### 推理测试 (test_inference)

```json
{
    "params": {
        "gpt_model": "path/to/gpt_model.ckpt",
        "sovits_model": "path/to/sovits_model.pth",
        "ref_audio": "path/to/reference.wav",
        "ref_text": "参考文本内容",
        "ref_language": "中文",
        "target_text": "目标合成文本",
        "target_language": "中文"
    }
}
```

## 错误处理

### 常见错误代码

- **404**: 任务不存在
- **400**: 任务正在运行中或参数错误
- **500**: 服务器内部错误

### 错误响应格式

```json
{
    "detail": "错误描述信息"
}
```

### 任务状态说明

- **created**: 任务已创建
- **running**: 任务执行中
- **completed**: 任务完成
- **failed**: 任务失败
- **cancelled**: 任务已取消

## 性能优化建议

### GPU并行设置

对于多GPU环境，可设置GPU ID为多个设备：

```json
{
    "gpu_id": "0-1-2"  // 使用GPU 0, 1, 2进行并行处理
}
```

### 批次大小调整

根据GPU显存调整批次大小：

- RTX 3080 (10GB): batch_size=16
- RTX 4090 (24GB): batch_size=32
- V100 (32GB): batch_size=48

### 训练轮数建议

- 快速测试: S2=20轮, S1=8轮
- 一般质量: S2=50轮, S1=15轮  
- 高质量: S2=100轮, S1=25轮

## 监控和日志

### 查看任务日志

```http
GET /api/v1/task/{task_id}/logs
```

### 监控指标

- 任务进度百分比
- 当前执行步骤
- 错误信息
- 执行时间

## 部署指南

### 生产环境部署

```bash
# 使用Gunicorn部署
pip install gunicorn

gunicorn -w 4 -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 7200 \
  training_service:app
```

### Docker部署

```dockerfile
FROM python:3.9

WORKDIR /app
COPY . /app

RUN pip install fastapi uvicorn python-multipart
RUN apt-get update && apt-get install -y ffmpeg

EXPOSE 8000
CMD ["python", "training_service.py"]
```

### 配置文件

创建 `config.yaml` 配置文件：

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4

paths:
  base_dir: "/path/to/GPT-SoVITS"
  work_dir: "/path/to/workdir"

gpu:
  default_device: "0"
  max_parallel: 4

training:
  default_epochs_s2: 50
  default_epochs_s1: 15
  default_batch_size: 16
```

## 故障排除

### 常见问题

1. **FFmpeg未安装**
   ```bash
   # 安装FFmpeg
   sudo apt install ffmpeg  # Ubuntu
   brew install ffmpeg      # macOS
   ```

2. **GPU内存不足**
   - 减小批次大小
   - 关闭其他GPU程序
   - 使用梯度累积

3. **依赖包缺失**
   ```bash
   pip install -r requirements.txt
   pip install fastapi uvicorn python-multipart
   ```

4. **权限问题**
   ```bash
   chmod +x *.py
   chmod 755 /path/to/workdir
   ```

### 调试模式

启用调试模式查看详细日志：

```python
import logging
logging.basicConfig(level=logging.DEBUG)

uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
```

## 贡献和支持

- 项目地址: [GitHub链接]
- 问题反馈: [Issues页面]
- 文档更新: [Wiki页面]

---

## 附录

### API完整接口列表

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/` | 根路径信息 |
| POST | `/api/v1/task/create` | 创建训练任务 |
| GET | `/api/v1/task/{task_id}` | 获取任务信息 |
| GET | `/api/v1/tasks` | 列出所有任务 |
| POST | `/api/v1/task/{task_id}/step/{step}` | 执行训练步骤 |
| POST | `/api/v1/task/{task_id}/cancel` | 取消任务 |
| DELETE | `/api/v1/task/{task_id}` | 删除任务 |
| POST | `/api/v1/task/{task_id}/files/upload` | 上传文件 |
| GET | `/api/v1/task/{task_id}/files/download/{filename}` | 下载文件 |
| GET | `/api/v1/task/{task_id}/logs` | 获取任务日志 |

### 训练步骤依赖关系

```
convert_audio → slice_audio → denoise_audio → asr_transcribe
                                                    ↓
extract_text_features ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ←
         ↓
extract_audio_features
         ↓  
extract_speaker_vectors
         ↓
extract_semantic_features
         ↓
train_sovits → train_gpt → test_inference
```

### 文件结构说明

```
api_tasks/
├── {task_id}/
│   ├── input_audio/          # 原始音频文件
│   ├── converted_wav/        # 转换后的WAV文件
│   ├── sliced/              # 切片后的音频
│   ├── denoised/            # 降噪后的音频
│   ├── transcripts/         # ASR转录结果
│   ├── experiments/         # 训练实验目录
│   │   └── {exp_name}/      # 具体实验数据
│   ├── output/              # 推理输出结果
│   ├── task_config.json     # 任务配置文件
│   ├── config_s2.json       # SoVITS训练配置
│   ├── config_s1.yaml       # GPT训练配置
│   └── training.log         # 训练日志
```
