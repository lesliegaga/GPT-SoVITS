#!/bin/bash

# GPT-SoVITS 训练服务启动脚本
# 使用说明：bash start_service.sh [选项]

set -e

# 默认配置
HOST="0.0.0.0"
PORT="8000"
WORKERS="1"
MODE="development"
LOG_LEVEL="info"
RELOAD="false"

# 函数定义
show_help() {
    cat << EOF
GPT-SoVITS 训练服务启动脚本

用法: bash start_service.sh [选项]

选项:
    -h, --help          显示此帮助信息
    -H, --host HOST     绑定主机地址 (默认: 0.0.0.0)
    -p, --port PORT     绑定端口 (默认: 8000)
    -w, --workers NUM   工作进程数 (默认: 1, 仅生产模式)
    -m, --mode MODE     运行模式: development/production (默认: development)
    -l, --log LOG       日志级别: debug/info/warning/error (默认: info)
    --install-deps      安装Python依赖包
    --check-env         检查环境依赖
    --daemon            后台运行服务

示例:
    bash start_service.sh                           # 开发模式启动
    bash start_service.sh -p 8080                   # 指定端口
    bash start_service.sh -m production -w 4        # 生产模式，4个工作进程
    bash start_service.sh --install-deps            # 安装依赖后启动
    bash start_service.sh --daemon                  # 后台运行
EOF
}

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
}

log_warn() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARN: $1"
}

# 检查环境依赖
check_environment() {
    log_info "检查环境依赖..."
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 未安装"
        exit 1
    fi
    log_info "Python: $(python3 --version)"
    
    # 检查pip
    if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
        log_error "pip 未安装"
        exit 1
    fi
    
    # 检查FFmpeg
    if ! command -v ffmpeg &> /dev/null; then
        log_warn "FFmpeg 未安装，音频格式转换功能将不可用"
        log_warn "安装方法："
        log_warn "  macOS: brew install ffmpeg"
        log_warn "  Ubuntu: sudo apt install ffmpeg"
        log_warn "  CentOS: sudo yum install ffmpeg"
    else
        log_info "FFmpeg: $(ffmpeg -version | head -1)"
    fi
    
    # 检查CUDA（可选）
    if command -v nvidia-smi &> /dev/null; then
        log_info "NVIDIA GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
    else
        log_warn "未检测到NVIDIA GPU，将使用CPU进行训练"
    fi
    
    # 检查项目文件
    if [[ ! -f "training_service.py" ]]; then
        log_error "training_service.py 文件不存在"
        exit 1
    fi
    
    if [[ ! -f "training_steps.py" ]]; then
        log_error "training_steps.py 文件不存在"
        exit 1
    fi
    
    log_info "环境检查完成"
}

# 安装Python依赖
install_dependencies() {
    log_info "安装Python依赖包..."
    
    # 确定pip命令
    PIP_CMD="pip"
    if command -v pip3 &> /dev/null; then
        PIP_CMD="pip3"
    fi
    
    # 安装基础依赖
    $PIP_CMD install fastapi uvicorn python-multipart
    
    # 如果存在requirements.txt，安装其中的依赖
    if [[ -f "requirements.txt" ]]; then
        log_info "安装requirements.txt中的依赖..."
        $PIP_CMD install -r requirements.txt
    fi
    
    log_info "依赖安装完成"
}

# 启动开发模式
start_development() {
    log_info "启动开发模式服务..."
    log_info "服务地址: http://${HOST}:${PORT}"
    log_info "API文档: http://${HOST}:${PORT}/docs"
    log_info "按 Ctrl+C 停止服务"
    
    python3 training_service.py
}

# 启动生产模式
start_production() {
    log_info "启动生产模式服务..."
    
    # 检查gunicorn
    if ! command -v gunicorn &> /dev/null; then
        log_warn "gunicorn 未安装，尝试安装..."
        pip install gunicorn
    fi
    
    log_info "服务地址: http://${HOST}:${PORT}"
    log_info "工作进程数: ${WORKERS}"
    log_info "API文档: http://${HOST}:${PORT}/docs"
    
    gunicorn -w $WORKERS \
        -k uvicorn.workers.UvicornWorker \
        --bind ${HOST}:${PORT} \
        --timeout 7200 \
        --log-level $LOG_LEVEL \
        --access-logfile - \
        --error-logfile - \
        training_service:app
}

# 后台运行服务
start_daemon() {
    log_info "启动后台服务..."
    
    # 创建日志目录
    mkdir -p logs
    
    if [[ "$MODE" == "production" ]]; then
        # 生产模式后台运行
        gunicorn -w $WORKERS \
            -k uvicorn.workers.UvicornWorker \
            --bind ${HOST}:${PORT} \
            --timeout 7200 \
            --log-level $LOG_LEVEL \
            --access-logfile logs/access.log \
            --error-logfile logs/error.log \
            --pid logs/training_service.pid \
            --daemon \
            training_service:app
        
        log_info "生产模式服务已启动"
        log_info "PID文件: logs/training_service.pid"
    else
        # 开发模式后台运行
        nohup python3 training_service.py > logs/service.log 2>&1 &
        echo $! > logs/training_service.pid
        log_info "开发模式服务已启动"
        log_info "PID: $(cat logs/training_service.pid)"
    fi
    
    log_info "日志目录: logs/"
    log_info "停止服务: bash stop_service.sh"
}

# 解析命令行参数
INSTALL_DEPS=false
CHECK_ENV=false
DAEMON=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -H|--host)
            HOST="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -w|--workers)
            WORKERS="$2"
            shift 2
            ;;
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -l|--log)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --install-deps)
            INSTALL_DEPS=true
            shift
            ;;
        --check-env)
            CHECK_ENV=true
            shift
            ;;
        --daemon)
            DAEMON=true
            shift
            ;;
        *)
            log_error "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 执行主逻辑
main() {
    log_info "=== GPT-SoVITS 训练服务启动 ==="
    log_info "运行模式: $MODE"
    log_info "绑定地址: ${HOST}:${PORT}"
    
    # 检查环境
    if [[ "$CHECK_ENV" == true ]] || [[ "$INSTALL_DEPS" == true ]]; then
        check_environment
    fi
    
    # 安装依赖
    if [[ "$INSTALL_DEPS" == true ]]; then
        install_dependencies
    fi
    
    # 启动服务
    if [[ "$DAEMON" == true ]]; then
        start_daemon
    elif [[ "$MODE" == "production" ]]; then
        start_production
    else
        start_development
    fi
}

# 脚本入口
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
