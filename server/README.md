# GPT-SoVITS 训练服务（API + 管理）

将 `test_demo.sh` 的完整语音克隆训练流程封装为 RESTful API，支持分步骤执行、进度监控、文件管理与并行加速；提供启动脚本与客户端示例，便于快速集成与部署。

## 功能概览

- ✅ 分步骤训练：11 个独立步骤可单独调用与编排
- ✅ 进度监控：实时任务状态、当前步骤与完成百分比
- ✅ 并行处理：多 GPU 并行的特征提取加速
- ✅ 文件管理：训练数据上传、结果文件下载
- ✅ 错误处理：统一错误返回与任务失败状态
- ✅ 任务管理：创建/查询/取消/删除任务，多任务并发
- ✅ 部署便捷：脚本化启动，支持生产与 Docker

## 目录结构

```
server/
├── __init__.py                 # Python 包初始化
├── training_service.py         # FastAPI 主服务（API）
├── training_steps.py           # 训练步骤处理与配置生成
├── service_config.py           # 服务配置
├── start_service.sh            # 启动脚本（开发/生产/守护）
├── stop_service.sh             # 停止脚本
└── api_tasks/                  # 任务工作目录（按 task_id 分隔）
```

## 快速开始

### 1) 安装依赖
```bash
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart
# 安装 FFmpeg（用于音频转换）
# macOS: brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg
```

### 2) 启动/停止服务
```bash
cd server
# 开发模式
bash start_service.sh
# 生产模式（多进程）
bash start_service.sh -m production -w 4
# 后台运行
bash start_service.sh --daemon

# 停止服务
bash stop_service.sh            # 正常停止
bash stop_service.sh -f         # 强制停止
bash stop_service.sh --cleanup  # 停止并清理
```

### 3) 访问
- 服务地址: `http://localhost:8000`
- 文档: `http://localhost:8000/docs`

## API 端点速览

| 方法 | 路径 | 功能 |
|------|------|------|
| POST | `/api/v1/task/create` | 创建训练任务 |
| GET | `/api/v1/task/{task_id}` | 获取任务信息 |
| GET | `/api/v1/tasks` | 列出所有任务 |
| POST | `/api/v1/task/{task_id}/step/{step}` | 执行训练步骤 |
| POST | `/api/v1/task/{task_id}/cancel` | 取消任务 |
| DELETE | `/api/v1/task/{task_id}` | 删除任务 |
| POST | `/api/v1/task/{task_id}/files/upload` | 上传文件 |
| GET | `/api/v1/task/{task_id}/files/download/{filename}` | 下载文件 |
| GET | `/api/v1/task/{task_id}/logs` | 获取任务日志 |

## 使用示例（Python）

```python
import requests
import time

BASE_URL = "http://localhost:8000/api/v1"

class TrainingClient:
    def __init__(self, base_url=BASE_URL):
        self.base_url = base_url

    def create_task(self, task_name, config):
        return requests.post(f"{self.base_url}/task/create",
                             json={"task_name": task_name, "config": config}).json()

    def upload_audio(self, task_id, audio_file_path):
        with open(audio_file_path, 'rb') as f:
            return requests.post(f"{self.base_url}/task/{task_id}/files/upload",
                                 files={'file': f}).json()

    def execute_step(self, task_id, step_name, params=None):
        return requests.post(f"{self.base_url}/task/{task_id}/step/{step_name}",
                             json={"params": params or {}}).json()

    def get_task_status(self, task_id):
        return requests.get(f"{self.base_url}/task/{task_id}").json()

    def wait_for_completion(self, task_id, timeout=3600):
        start = time.time()
        while time.time() - start < timeout:
            s = self.get_task_status(task_id)
            if s['status'] in ['completed', 'failed']:
                return s
            print(f"进度: {s['progress']:.1f}% - {s['current_step']}")
            time.sleep(10)
        raise TimeoutError('任务执行超时')

def main():
    client = TrainingClient()
    task = client.create_task("语音克隆测试", {
        "exp_name": "my_speaker",
        "language": "zh",
        "batch_size": 16,
        "epochs_s2": 20,
        "epochs_s1": 8,
        "gpu_id": "0"
    })
    task_id = task['task_id']

    for f in ["audio1.wav", "audio2.wav", "audio3.wav"]:
        client.upload_audio(task_id, f)

    for step in [
        "convert_audio", "slice_audio", "denoise_audio", "asr_transcribe",
        "extract_text_features", "extract_audio_features",
        "extract_speaker_vectors", "extract_semantic_features",
        "train_sovits", "train_gpt", "test_inference"
    ]:
        client.execute_step(task_id, step)
        final_status = client.wait_for_completion(task_id)
        if final_status['status'] == 'failed':
            print("失败:", final_status.get('error_message'))
            break

if __name__ == "__main__":
    main()
```

## 训练步骤与依赖

流程概览：
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

### 步骤参数示例
- slice_audio
```json
{
  "params": {
    "min_length": -34,
    "min_interval": 4000,
    "hop_size": 300,
    "max_sil_kept": 10,
    "_max": 500,
    "alpha": 0.9,
    "n_parts": 0.25,
    "is_speech": 0,
    "num_processes": 1
  }
}
```

- denoise_audio / asr_transcribe
```json
{ "params": { "precision": "float16" } }
```

- test_inference
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

## 配置与运行模式

- `service_config.py` 提供服务配置，支持环境变量覆盖
- 默认主机与端口：`0.0.0.0:8000`
- 工作目录：`server/api_tasks/`
- 运行模式：
  - 开发：单进程，热重载
  - 生产：gunicorn + uvicorn 多进程

导入方式：
```python
from server.training_service import app
from server.service_config import print_config
from server.training_steps import StepProcessor, ConfigGenerator
```

或在 `server` 目录下独立运行：
```bash
cd server
export GPT_SOVITS_SERVER_MODE=standalone
python3 training_service.py
```

## 部署

### Gunicorn（生产环境）
```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 7200 \
  training_service:app
```

### Docker
```dockerfile
FROM python:3.9
WORKDIR /app
COPY . /app
RUN pip install fastapi uvicorn python-multipart
RUN apt-get update && apt-get install -y ffmpeg
EXPOSE 8000
CMD ["python", "training_service.py"]
```

### 可选配置文件（示例）
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

## 监控、日志与故障排除

### 查看日志与监控
```bash
tail -f logs/service.log
curl http://localhost:8000/api/v1/task/{task_id}/logs
curl http://localhost:8000/api/v1/task/{task_id}
curl http://localhost:8000/api/v1/tasks
```

### 常见问题
- FFmpeg 未安装：安装系统级 FFmpeg
- GPU 内存不足：减小 batch_size、关闭其他 GPU 程序、使用梯度累积
- 依赖缺失：`pip install -r requirements.txt` 并补充 `fastapi uvicorn python-multipart`
- 权限问题：`chmod +x *.py && chmod 755 /path/to/workdir`

### 调试模式
```python
import logging
logging.basicConfig(level=logging.DEBUG)
uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
```

## 性能与安全

- 多 GPU：示例 `{ "gpu_id": "0-1-2" }`
- 批次大小：按显存调整（如 24GB 可设 32）
- 并行：特征提取步骤支持多 GPU 并行，异步任务执行
- 安全：限制与校验上传文件、任务隔离与权限控制、错误信息过滤、资源使用监控

## 规划（可选）
- 实时训练进度 WebSocket 推送
- 分布式训练与模型版本管理
- 结果可视化与监控面板
- Redis 缓存、数据库持久化、负载均衡与容器化

## 支持
- 文档: `http://localhost:8000/docs`
- 示例: 参见仓库 `client_example.py`
