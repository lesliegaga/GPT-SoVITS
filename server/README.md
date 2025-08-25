# GPT-SoVITS 训练服务

本目录包含了GPT-SoVITS训练服务的所有相关文件。

## 目录结构

```
server/
├── __init__.py                 # Python包初始化文件
├── training_service.py         # 主要API服务，基于FastAPI
├── training_steps.py           # 训练步骤处理器和配置生成器
├── service_config.py           # 服务配置文件
├── start_service.sh            # 服务启动脚本
├── stop_service.sh             # 服务停止脚本
├── SERVICE_SUMMARY.md          # 服务功能说明文档
├── API_USAGE_GUIDE.md          # API使用指南
└── api_tasks/                  # API任务工作目录
```

## 快速开始

### 1. 启动服务

```bash
cd server
bash start_service.sh
```

### 2. 停止服务

```bash
cd server
bash stop_service.sh
```

### 3. 从Python代码中使用

```python
# 从根目录导入
from server.training_service import app
from server.service_config import print_config
from server.training_steps import StepProcessor, ConfigGenerator

# 或者直接进入server目录使用
cd server
python3 training_service.py
```

## 配置说明

- `service_config.py`: 包含所有服务配置，支持环境变量覆盖
- 默认端口: 8000
- 默认主机: 0.0.0.0
- 工作目录: api_tasks/

## 支持的模式

- **开发模式**: 单进程，支持热重载
- **生产模式**: 多进程，使用gunicorn + uvicorn

## 依赖要求

- Python 3.8+
- FastAPI
- uvicorn
- gunicorn (生产模式)
- 其他依赖见requirements.txt

## 导入机制

本服务支持两种导入模式：

### 1. 包模式（推荐）
从项目根目录导入，使用相对导入：
```python
from server.training_service import app
from server.service_config import print_config
from server.training_steps import StepProcessor, ConfigGenerator
```

### 2. 独立模式
在server目录下直接运行，使用绝对导入：
```bash
cd server
export GPT_SOVITS_SERVER_MODE=standalone
python3 training_service.py
```

启动脚本会自动设置正确的环境变量。

## 注意事项

1. 所有文件现在都在server目录下，使用智能导入机制
2. 启动脚本会自动检测环境依赖并设置正确的导入模式
3. 支持后台运行和进程管理
4. 包含完整的日志记录和错误处理
5. 无需手动设置环境变量，启动脚本会自动处理
