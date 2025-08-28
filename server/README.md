# GPT-SoVITS 训练服务（基于角色的API）

基于角色管理的语音克隆训练服务，将完整的语音克隆训练流程封装为 RESTful API，支持角色创建、音频处理、模型训练、语音推理等完整流程；提供启动脚本与客户端示例，便于快速集成与部署。

## 功能概览

- ✅ 角色管理：创建/删除/重命名角色，设置默认角色
- ✅ 音频处理：音频上传、格式转换、切片、降噪、转录
- ✅ 训练流程：特征提取、SoVITS训练、GPT训练的完整流程
- ✅ 语音推理：基于训练模型的文本转语音合成
- ✅ 状态监控：实时处理状态、进度监控、错误处理
- ✅ 文件管理：音频文件列表、删除、下载等操作
- ✅ 持久化检查：基于文件系统的状态验证
- ✅ 智能状态管理：文件变更时自动状态失效
- ✅ 并行处理：多 GPU 并行的特征提取加速
- ✅ 部署便捷：脚本化启动，支持生产与 Docker

## 目录结构

```
server/
├── __init__.py                 # Python 包初始化
├── training_service.py         # FastAPI 主服务（基于角色的API）
├── training_steps.py           # 训练步骤处理与配置生成
├── service_config.py           # 服务配置
├── start_service.sh            # 启动脚本（开发/生产/守护）
├── stop_service.sh             # 停止脚本
└── work_dir/                   # 工作目录
    ├── characters/             # 角色数据目录（按角色名分隔）
    │   └── {character_name}/   # 单个角色目录
    │       ├── raw_audio/      # 原始音频文件
    │       ├── converted_audio/# 转换后音频
    │       ├── sliced_audio/   # 切片音频
    │       ├── denoised_audio/ # 降噪音频
    │       ├── transcripts/    # 转录文件
    │       ├── experiments/    # 训练实验数据
    │       └── models/         # 训练好的模型
    └── inference_output/       # 推理输出目录
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

### 角色管理
| 方法 | 路径 | 功能 |
|------|------|------|
| POST | `/api/v1/characters` | 创建角色 |
| GET | `/api/v1/characters` | 列出所有角色 |
| GET | `/api/v1/characters/{character_name}` | 获取角色信息 |
| PUT | `/api/v1/characters/{character_name}` | 重命名角色 |
| DELETE | `/api/v1/characters/{character_name}` | 删除角色 |
| POST | `/api/v1/characters/{character_name}/set_default` | 设置默认角色 |
| GET | `/api/v1/default_character` | 获取默认角色 |

### 音频管理
| 方法 | 路径 | 功能 |
|------|------|------|
| POST | `/api/v1/characters/{character_name}/audio/upload` | 上传音频文件 |
| GET | `/api/v1/characters/{character_name}/audio/files` | 获取音频文件列表 |
| DELETE | `/api/v1/characters/{character_name}/audio/files/{filename}` | 删除指定音频文件 |
| POST | `/api/v1/characters/{character_name}/audio/process` | 开始音频处理 |
| GET | `/api/v1/characters/{character_name}/audio/status` | 获取音频处理状态 |
| GET | `/api/v1/characters/{character_name}/audio/check_status` | 基于文件检查音频处理状态 |

### 训练管理
| 方法 | 路径 | 功能 |
|------|------|------|
| POST | `/api/v1/characters/{character_name}/training/start` | 开始训练 |
| GET | `/api/v1/characters/{character_name}/training/status` | 获取训练状态 |
| GET | `/api/v1/characters/{character_name}/training/check_status` | 基于文件检查训练状态 |
| POST | `/api/v1/characters/{character_name}/training/clean` | 清理训练模型 |

### 推理管理
| 方法 | 路径 | 功能 |
|------|------|------|
| POST | `/api/v1/inference` | 开始推理 |
| GET | `/api/v1/inference/{inference_id}` | 获取推理状态 |
| GET | `/api/v1/inference` | 列出所有推理记录 |

### 文件下载
| 方法 | 路径 | 功能 |
|------|------|------|
| GET | `/api/v1/characters/{character_name}/download/{filename}` | 下载角色相关文件 |
| GET | `/api/v1/inference/{inference_id}/download` | 下载推理结果 |

## 使用示例（Python）

```python
import requests
import time

BASE_URL = "http://localhost:8000/api/v1"

class GPTSoVITSClient:
    def __init__(self, base_url=BASE_URL):
        self.base_url = base_url

    def create_character(self, character_name, config):
        """创建角色"""
        return requests.post(f"{self.base_url}/characters",
                             json={"character_name": character_name, "config": config}).json()

    def upload_audio(self, character_name, audio_file_path):
        """上传音频文件"""
        with open(audio_file_path, 'rb') as f:
            return requests.post(f"{self.base_url}/characters/{character_name}/audio/upload",
                                 files={'file': f}).json()

    def get_audio_files(self, character_name):
        """获取音频文件列表"""
        return requests.get(f"{self.base_url}/characters/{character_name}/audio/files").json()

    def delete_audio_file(self, character_name, filename):
        """删除音频文件"""
        return requests.delete(f"{self.base_url}/characters/{character_name}/audio/files/{filename}").json()

    def start_audio_processing(self, character_name):
        """开始音频处理"""
        return requests.post(f"{self.base_url}/characters/{character_name}/audio/process").json()

    def start_training(self, character_name):
        """开始训练"""
        return requests.post(f"{self.base_url}/characters/{character_name}/training/start").json()

    def get_audio_status(self, character_name):
        """获取音频处理状态"""
        return requests.get(f"{self.base_url}/characters/{character_name}/audio/status").json()

    def get_training_status(self, character_name):
        """获取训练状态"""
        return requests.get(f"{self.base_url}/characters/{character_name}/training/status").json()

    def check_audio_status_from_files(self, character_name):
        """基于文件检查音频处理状态"""
        return requests.get(f"{self.base_url}/characters/{character_name}/audio/check_status").json()

    def check_training_status_from_files(self, character_name):
        """基于文件检查训练状态"""
        return requests.get(f"{self.base_url}/characters/{character_name}/training/check_status").json()

    def start_inference(self, character_name, target_text, ref_audio=None, ref_text=None):
        """开始推理"""
        data = {
            "character_name": character_name,
            "target_text": target_text,
            "ref_language": "中文",
            "target_language": "中文"
        }
        if ref_audio:
            data["ref_audio"] = ref_audio
        if ref_text:
            data["ref_text"] = ref_text
        return requests.post(f"{self.base_url}/inference", json=data).json()

    def get_inference_status(self, inference_id):
        """获取推理状态"""
        return requests.get(f"{self.base_url}/inference/{inference_id}").json()

    def wait_for_completion(self, character_name, process_type="training", timeout=3600):
        """等待处理完成"""
        start = time.time()
        while time.time() - start < timeout:
            if process_type == "audio":
                status = self.get_audio_status(character_name)
            else:
                status = self.get_training_status(character_name)
            
            if status['status'] in ['completed', 'failed', 'outdated']:
                return status
            print(f"进度: {status.get('progress', 0):.1f}% - {status.get('current_step', 'unknown')}")
            time.sleep(10)
        raise TimeoutError('处理超时')

def main():
    client = GPTSoVITSClient()
    
    # 1. 创建角色
    character_name = "my_speaker"
    character = client.create_character(character_name, {
        "character_name": character_name,
        "language": "zh",
        "batch_size": 16,
        "epochs_s2": 50,
        "epochs_s1": 15,
        "gpu_id": "0",
        "enable_denoise": True
    })
    print(f"角色创建成功: {character['character_name']}")

    # 2. 上传音频文件
    for audio_file in ["audio1.wav", "audio2.wav", "audio3.wav"]:
        try:
            result = client.upload_audio(character_name, audio_file)
            print(f"上传成功: {result['filename']}")
        except FileNotFoundError:
            print(f"文件不存在: {audio_file}")

    # 3. 开始音频处理
    client.start_audio_processing(character_name)
    audio_status = client.wait_for_completion(character_name, "audio")
    
    if audio_status['status'] == 'completed':
        print("音频处理完成")
        
        # 4. 开始训练
        client.start_training(character_name)
        training_status = client.wait_for_completion(character_name, "training")
        
        if training_status['status'] == 'completed':
            print("训练完成")
            
            # 5. 测试推理
            inference = client.start_inference(character_name, "你好，这是一个测试语音。")
            print(f"推理ID: {inference['inference_id']}")
            
        else:
            print("训练失败:", training_status.get('error_message'))
    else:
        print("音频处理失败:", audio_status.get('error_message'))

if __name__ == "__main__":
    main()
```

## 训练流程

### 完整流程概览
```
1. 角色创建 → 2. 音频上传 → 3. 音频处理 → 4. 模型训练 → 5. 语音推理
```

### 详细步骤说明

#### 1. 角色管理阶段
- **创建角色**: 设置角色名称、语言、训练参数等
- **配置管理**: GPU设置、批次大小、训练轮数等
- **默认角色**: 可设置默认角色用于推理

#### 2. 音频处理阶段 (自动化)
当调用音频处理API时，系统会自动执行以下步骤：
```
原始音频 → 格式转换 → 音频切片 → 降噪处理 → 语音转录
```
- **格式转换**: 统一转换为WAV格式
- **音频切片**: 智能分割长音频文件
- **降噪处理**: 可选的音频降噪（可在角色配置中开启/关闭）
- **语音转录**: 自动生成转录文本

#### 3. 模型训练阶段 (自动化)
当调用训练API时，系统会自动执行以下步骤：
```
文本特征提取 → 音频特征提取 → 说话人向量提取 → 语义特征提取 → SoVITS训练 → GPT训练
```
- **特征提取**: 多种特征的并行提取
- **SoVITS训练**: 语音合成模型训练
- **GPT训练**: 语言模型训练

#### 4. 推理阶段
- **语音合成**: 基于训练好的模型进行文本转语音
- **参考音频**: 可指定参考音频和文本，或自动选择
- **多语言**: 支持中文、英文、日文等多种语言

### 状态管理

#### 处理状态类型
- `pending`: 等待处理
- `running`: 正在处理  
- `completed`: 处理完成
- `failed`: 处理失败
- `cancelled`: 已取消
- `outdated`: 已过期（文件列表已变更）

#### 状态检查方式
1. **内存状态检查**: 基于运行时状态的快速检查
2. **文件系统检查**: 基于持久化文件的可靠检查
3. **自动状态失效**: 音频文件变更时自动标记相关状态为过期

### 角色配置示例
```json
{
  "character_name": "my_speaker",
  "language": "zh",
  "batch_size": 16,
  "epochs_s2": 50,
  "epochs_s1": 15, 
  "gpu_id": "0",
  "enable_denoise": true
}
```

### 推理请求示例
```json
{
  "character_name": "my_speaker",
  "target_text": "你好，欢迎使用GPT-SoVITS语音克隆系统。",
  "ref_audio": "/path/to/reference.wav",
  "ref_text": "这是参考音频的文本",
  "ref_language": "中文",
  "target_language": "中文"
}
```

## 配置与运行模式

- `service_config.py` 提供服务配置，支持环境变量覆盖
- 默认主机与端口：`0.0.0.0:8000`
- 工作目录：`work_dir/` (包含角色数据和推理输出)
- 运行模式：
  - 开发：单进程，热重载  
  - 生产：gunicorn + uvicorn 多进程

### 导入方式
作为模块导入：
```python
from server.training_service import app, CharacterBasedTrainingService
from server.service_config import print_config, get_base_path, get_work_dir
from server.training_steps import StepProcessor, ConfigGenerator
```

独立运行：
```bash
cd server
export GPT_SOVITS_SERVER_MODE=standalone
python3 training_service.py
```

### 启动参数
```bash
# 指定端口
python3 training_service.py --port 8216

# 指定主机地址
python3 training_service.py --host 127.0.0.1

# 开发模式（自动重载）
python3 training_service.py --reload

# 查看当前配置
python3 training_service.py --config
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
# 查看服务日志
tail -f logs/service.log

# 查看所有角色
curl http://localhost:8000/api/v1/characters

# 查看特定角色信息
curl http://localhost:8000/api/v1/characters/{character_name}

# 查看音频处理状态
curl http://localhost:8000/api/v1/characters/{character_name}/audio/status

# 查看训练状态  
curl http://localhost:8000/api/v1/characters/{character_name}/training/status

# 基于文件检查状态
curl http://localhost:8000/api/v1/characters/{character_name}/audio/check_status
curl http://localhost:8000/api/v1/characters/{character_name}/training/check_status

# 查看推理记录
curl http://localhost:8000/api/v1/inference
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

## 新功能特性

### 持久化状态检查
- **文件系统检查**: 基于实际文件存在情况检查处理状态
- **状态自动失效**: 音频文件变更时自动标记相关状态为过期
- **智能状态管理**: 区分内存状态和持久化状态

### 音频文件管理
- **文件列表查询**: 获取详细的音频文件信息（大小、时长、创建时间等）
- **文件删除**: 支持删除指定音频文件并自动更新状态
- **智能推理**: 自动选择参考音频和提取参考文本

### 角色管理增强
- **角色重命名**: 支持角色重命名并保持数据完整性
- **默认角色**: 设置默认角色简化推理调用
- **配置持久化**: 角色配置自动保存和加载

## 规划（可选）
- 实时训练进度 WebSocket 推送
- 分布式训练与模型版本管理  
- 结果可视化与监控面板
- Redis 缓存、数据库持久化、负载均衡与容器化
- 音频文件批量管理和预处理
- 训练进度更细粒度监控

## 支持
- **API文档**: `http://localhost:8000/docs`
- **交互式文档**: `http://localhost:8000/redoc`
- **示例代码**: 参见本文档中的Python客户端示例
- **配置查看**: `python3 training_service.py --config`
