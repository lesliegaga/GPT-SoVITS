#!/bin/bash

# GPT-SoVITS 训练服务停止脚本

set -e

# 日志函数
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
}

log_warn() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARN: $1"
}

# 显示帮助信息
show_help() {
    cat << EOF
GPT-SoVITS 训练服务停止脚本

用法: bash stop_service.sh [选项]

选项:
    -h, --help      显示此帮助信息
    -f, --force     强制停止所有相关进程
    -k, --kill      使用SIGKILL信号强制终止
    --cleanup       清理日志和临时文件

示例:
    bash stop_service.sh           # 正常停止服务
    bash stop_service.sh -f        # 强制停止所有相关进程
    bash stop_service.sh --cleanup # 停止服务并清理文件
EOF
}

# 停止PID文件中的进程
stop_pid_file() {
    local pid_file="$1"
    local service_name="$2"
    
    if [[ -f "$pid_file" ]]; then
        local pid=$(cat "$pid_file")
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            log_info "停止 $service_name (PID: $pid)..."
            if [[ "$FORCE_KILL" == true ]]; then
                kill -9 "$pid"
            else
                kill -TERM "$pid"
                # 等待进程优雅退出
                local count=0
                while kill -0 "$pid" 2>/dev/null && [[ $count -lt 30 ]]; do
                    sleep 1
                    ((count++))
                done
                
                # 如果还没退出，强制终止
                if kill -0 "$pid" 2>/dev/null; then
                    log_warn "进程未能优雅退出，强制终止..."
                    kill -9 "$pid" 2>/dev/null || true
                fi
            fi
            log_info "$service_name 已停止"
        else
            log_warn "$service_name PID文件存在但进程不运行"
        fi
        rm -f "$pid_file"
    else
        log_warn "$service_name PID文件不存在: $pid_file"
    fi
}

# 强制停止所有相关进程
force_stop_all() {
    log_info "查找并停止所有相关进程..."
    
    # 查找training_service.py进程
    local pids=$(pgrep -f "training_service.py" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        log_info "发现training_service.py进程: $pids"
        for pid in $pids; do
            log_info "停止进程 $pid..."
            if [[ "$FORCE_KILL" == true ]]; then
                kill -9 "$pid" 2>/dev/null || true
            else
                kill -TERM "$pid" 2>/dev/null || true
            fi
        done
    fi
    
    # 查找gunicorn进程
    local gunicorn_pids=$(pgrep -f "gunicorn.*server.training_service" 2>/dev/null || true)
    if [[ -n "$gunicorn_pids" ]]; then
        log_info "发现gunicorn进程: $gunicorn_pids"
        for pid in $gunicorn_pids; do
            log_info "停止gunicorn进程 $pid..."
            if [[ "$FORCE_KILL" == true ]]; then
                kill -9 "$pid" 2>/dev/null || true
            else
                kill -TERM "$pid" 2>/dev/null || true
            fi
        done
    fi
    
    # 查找uvicorn进程
    local uvicorn_pids=$(pgrep -f "uvicorn.*training_service" 2>/dev/null || true)
    if [[ -n "$uvicorn_pids" ]]; then
        log_info "发现uvicorn进程: $uvicorn_pids"
        for pid in $uvicorn_pids; do
            log_info "停止uvicorn进程 $pid..."
            if [[ "$FORCE_KILL" == true ]]; then
                kill -9 "$pid" 2>/dev/null || true
            else
                kill -TERM "$pid" 2>/dev/null || true
            fi
        done
    fi
    
    # 等待进程退出
    if [[ "$FORCE_KILL" != true ]]; then
        sleep 2
        # 检查是否还有残留进程
        local remaining=$(pgrep -f "training_service|gunicorn.*server.training_service|uvicorn.*training_service" 2>/dev/null || true)
        if [[ -n "$remaining" ]]; then
            log_warn "发现残留进程，强制终止: $remaining"
            for pid in $remaining; do
                kill -9 "$pid" 2>/dev/null || true
            done
        fi
    fi
}

# 清理文件
cleanup_files() {
    log_info "清理临时文件和日志..."
    
    # 清理日志文件
    if [[ -d "logs" ]]; then
        log_info "清理日志目录..."
        rm -rf logs/*.log logs/*.pid
        # 保留目录但清空内容
        find logs -type f -name "*.log" -delete 2>/dev/null || true
        find logs -type f -name "*.pid" -delete 2>/dev/null || true
    fi
    
    # 清理临时API任务目录（可选）
    if [[ -d "api_tasks" ]] && [[ "$CLEANUP_TASKS" == true ]]; then
        log_warn "清理API任务目录..."
        read -p "是否确认删除所有API任务数据？(y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf api_tasks/*
            log_info "API任务数据已清理"
        else
            log_info "跳过API任务数据清理"
        fi
    fi
    
    # 清理Python缓存
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    
    log_info "文件清理完成"
}

# 检查服务状态
check_status() {
    log_info "检查服务状态..."
    
    local running=false
    
    # 检查PID文件
    if [[ -f "logs/training_service.pid" ]]; then
        local pid=$(cat "logs/training_service.pid")
        if kill -0 "$pid" 2>/dev/null; then
            log_info "服务运行中 (PID: $pid)"
            running=true
        fi
    fi
    
    # 检查进程
    local pids=$(pgrep -f "training_service|gunicorn.*server.training_service|uvicorn.*training_service" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        log_info "发现相关进程: $pids"
        running=true
    fi
    
    if [[ "$running" == false ]]; then
        log_info "未发现运行中的服务"
    fi
    
    # 检查端口占用
    local port_usage=$(netstat -tlnp 2>/dev/null | grep ":8000 " || true)
    if [[ -n "$port_usage" ]]; then
        log_info "端口8000占用情况:"
        echo "$port_usage"
    fi
}

# 解析命令行参数
FORCE_STOP=false
FORCE_KILL=false
CLEANUP=false
CLEANUP_TASKS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -f|--force)
            FORCE_STOP=true
            shift
            ;;
        -k|--kill)
            FORCE_KILL=true
            shift
            ;;
        --cleanup)
            CLEANUP=true
            shift
            ;;
        --cleanup-tasks)
            CLEANUP_TASKS=true
            shift
            ;;
        *)
            log_error "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 主函数
main() {
    log_info "=== GPT-SoVITS 训练服务停止 ==="
    
    # 检查状态
    check_status
    
    # 停止服务
    if [[ "$FORCE_STOP" == true ]]; then
        force_stop_all
    else
        # 尝试通过PID文件优雅停止
        stop_pid_file "logs/training_service.pid" "训练服务"
    fi
    
    # 清理文件
    if [[ "$CLEANUP" == true ]]; then
        cleanup_files
    fi
    
    log_info "服务停止完成"
}

# 脚本入口
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
