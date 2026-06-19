#!/usr/bin/env bash
# =============================================================
#  知识问答系统 启动脚本 v3.0
#  用法: ./start.sh <command> [options]
#
#  命令:
#    start   [options]  启动服务（后台运行）
#    stop               停止所有服务
#    restart [options]  重启服务
#    status             查看运行状态
#    logs [backend|frontend]  查看日志
#
#  选项 (仅 start/restart):
#    --install          启动前安装/更新 Python 依赖
#    --build-frontend   构建前端 (npm install + npm run build)
#    --dev              开发模式（前端热更新）
#    --foreground       前台运行（调试用，Ctrl+C 退出）
# =============================================================
set -e

# ---------- 颜色定义 ----------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# ---------- 项目路径 ----------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PID_DIR="$SCRIPT_DIR/.pids"
LOG_DIR="$SCRIPT_DIR/logs"
BACKEND_PID_FILE="$PID_DIR/backend.pid"
FRONTEND_PID_FILE="$PID_DIR/frontend.pid"
BACKEND_LOG="$LOG_DIR/backend.log"
FRONTEND_LOG="$LOG_DIR/frontend.log"

# ---------- 辅助函数 ----------
create_dirs() {
    mkdir -p "$PID_DIR" "$LOG_DIR"
}

# 检查进程是否存活
is_running() {
    local pid_file="$1"
    if [ -f "$pid_file" ]; then
        local pid
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
        rm -f "$pid_file"
    fi
    return 1
}

# 杀掉单个进程（先 SIGTERM，5 秒后 SIGKILL）
kill_process() {
    local name="$1"
    local pid_file="$2"

    if [ ! -f "$pid_file" ]; then
        return 0
    fi

    local pid
    pid=$(cat "$pid_file")

    if ! kill -0 "$pid" 2>/dev/null; then
        rm -f "$pid_file"
        return 0
    fi

    echo -e "   停止 ${name} (PID: $pid)..."

    # 先发 SIGTERM（优雅退出）
    kill "$pid" 2>/dev/null

    # 等待最多 5 秒
    local waited=0
    while kill -0 "$pid" 2>/dev/null && [ $waited -lt 5 ]; do
        sleep 0.5
        waited=$((waited + 1))
    done

    # 还没退出就 SIGKILL
    if kill -0 "$pid" 2>/dev/null; then
        echo -e "   ${YELLOW}进程未响应，强制终止...${NC}"
        kill -9 "$pid" 2>/dev/null
        # 同时杀掉子进程（uvicorn 会 fork worker）
        pkill -P "$pid" 2>/dev/null || true
        sleep 0.5
    fi

    rm -f "$pid_file"
    echo -e "   ${GREEN}✓${NC} ${name} 已停止"
}

# 清理所有残留进程
cleanup_all() {
    echo -e "${YELLOW}⏹  正在停止所有服务...${NC}"
    kill_process "后端 API" "$BACKEND_PID_FILE"
    kill_process "前端 Dev Server" "$FRONTEND_PID_FILE"

    # 清理可能残留的 uvicorn 进程
    if pgrep -f "uvicorn api.main" > /dev/null 2>&1; then
        echo -e "   清理残留 uvicorn 进程..."
        pkill -f "uvicorn api.main" 2>/dev/null || true
    fi

    # 等待所有相关进程真正退出（最多 5 秒）
    echo -n "   等待进程退出"
    local waited=0
    while pgrep -f "uvicorn api.main" > /dev/null 2>&1 && [ $waited -lt 10 ]; do
        sleep 0.5
        waited=$((waited + 1))
        echo -n "."
    done
    echo ""

    # 确认 Milvus Lite 锁已释放（LOCK 文件不再被 flock）
    local lock_file="$SCRIPT_DIR/data/milvus.db/LOCK"
    if [ -f "$lock_file" ]; then
        # 尝试获取排他锁来验证锁是否已释放
        if command -v flock &> /dev/null; then
            if ! flock -n "$lock_file" true 2>/dev/null; then
                echo -e "   ${YELLOW}等待 Milvus 锁释放...${NC}"
                sleep 2
            fi
        fi
    fi

    echo -e "${GREEN}✅ 所有服务已停止${NC}"
}

# 信号处理：确保 Ctrl+C 也能清理（前台模式）
trap 'echo ""; cleanup_all; exit 0' SIGINT SIGTERM

# ---------- Java 环境 (PySpark) ----------
export JAVA_HOME=/opt/jdk-17.0.2
export PATH=$JAVA_HOME/bin:$PATH
export JAVA_TOOL_OPTIONS="-XX:-UseContainerSupport"  # WSL2 cgroupv2 兼容
export PYSPARK_PYTHON=/home/yhwz/miniconda3/envs/kbqa/bin/python3
export PYSPARK_DRIVER_PYTHON=/home/yhwz/miniconda3/envs/kbqa/bin/python3

# ---------- 检查 Python ----------
find_python() {
    local conda_py="/home/yhwz/miniconda3/envs/kbqa/bin/python3"
    if [ -f "$conda_py" ]; then
        echo "$conda_py"
    elif command -v python3 &> /dev/null; then
        echo "python3"
    else
        echo ""
    fi
}

# ---------- 检查 Node.js ----------
has_node() {
    command -v node &> /dev/null
}

# ---------- 等待端口就绪 ----------
wait_port() {
    local port="$1"
    local name="$2"
    local pid="$3"
    local timeout="${4:-30}"

    echo -n "   等待 ${name} 就绪"
    for i in $(seq 1 "$timeout"); do
        if curl -s "http://127.0.0.1:$port/health" > /dev/null 2>&1 || \
           curl -s "http://127.0.0.1:$port" > /dev/null 2>&1; then
            echo ""
            return 0
        fi
        if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
            echo ""
            echo -e "   ${RED}❌ ${name} 启动失败，查看日志: ${BACKEND_LOG}${NC}"
            return 1
        fi
        sleep 1
        echo -n "."
    done
    echo ""
    echo -e "   ${YELLOW}⚠  等待超时，服务可能仍在启动中${NC}"
    return 0
}

# ================================================================
#  命令: start
# ================================================================
do_start() {
    local PYTHON
    PYTHON=$(find_python)
    if [ -z "$PYTHON" ]; then
        echo -e "${RED}❌ 未找到 python3，请先安装 Python 3.9+${NC}"
        exit 1
    fi

    create_dirs

    # 检查是否已在运行
    if is_running "$BACKEND_PID_FILE"; then
        local old_pid
        old_pid=$(cat "$BACKEND_PID_FILE")
        echo -e "${YELLOW}⚠  后端服务已在运行中 (PID: $old_pid)${NC}"
        echo -e "   请先执行 ${CYAN}./start.sh stop${NC} 或 ${CYAN}./start.sh restart${NC}"
        exit 1
    fi

    # 打印横幅
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════╗"
    echo "║       🧠 知识问答系统 (RAG QA)          ║"
    echo "║           启动脚本 v3.0                 ║"
    echo "╚══════════════════════════════════════════╝"
    echo -e "${NC}"

    echo -e "${GREEN}✓${NC} Python: $($PYTHON --version 2>&1)"
    if has_node; then
        echo -e "${GREEN}✓${NC} Node.js: $(node --version 2>&1)"
    fi

    # 检查/创建 .env
    if [ ! -f .env ] && [ -f .env.example ]; then
        cp .env.example .env
        echo -e "${YELLOW}⚠  已从 .env.example 创建 .env${NC}"
    fi

    # 创建数据目录
    mkdir -p data/uploads data/processed data/raw/images data/sessions logs

    # 安装 Python 依赖
    if [ "$INSTALL_DEPS" = true ]; then
        echo -e "\n${CYAN}📦 安装/更新 Python 依赖...${NC}"
        $PYTHON -m pip install -r requirements.txt -q
        echo -e "${GREEN}✓${NC} Python 依赖安装完成"
    fi

    # 构建前端
    if [ "$BUILD_FRONTEND" = true ] && has_node; then
        echo -e "\n${CYAN}🔨 构建前端...${NC}"
        cd frontend
        if [ ! -d "node_modules" ]; then
            echo "   安装 npm 依赖..."
            npm install --silent
        fi
        echo "   构建生产版本..."
        npm run build
        cd "$SCRIPT_DIR"
        echo -e "${GREEN}✓${NC} 前端构建完成"
    fi

    # 读取端口
    API_PORT=${API_PORT:-8000}
    DEV_PORT=${DEV_PORT:-5173}
    if [ -f .env ]; then
        local env_port
        env_port=$(grep -E '^API_PORT=' .env 2>/dev/null | cut -d'=' -f2 | tr -d ' ')
        [ -n "$env_port" ] && API_PORT="$env_port"
    fi

    # ---------- 启动后端 ----------
    echo -e "\n${CYAN}🚀 启动后端 API 服务 (端口: $API_PORT)...${NC}"

    if [ "$FOREGROUND" = true ]; then
        # 前台模式
        echo -e "   ${YELLOW}前台运行模式，按 Ctrl+C 退出${NC}"
        if [ "$DEV_MODE" = true ] && has_node; then
            echo -e "${CYAN}🎨 启动前端开发服务 (端口: $DEV_PORT)...${NC}"
            cd frontend
            npm run dev -- --port "$DEV_PORT" &
            FRONTEND_PID=$!
            echo "$FRONTEND_PID" > "$FRONTEND_PID_FILE"
            cd "$SCRIPT_DIR"
        fi
        $PYTHON -m uvicorn api.main:app --host 0.0.0.0 --port "$API_PORT" --reload
    else
        # 后台模式 — 后端
        nohup $PYTHON -m uvicorn api.main:app --host 0.0.0.0 --port "$API_PORT" \
            > "$BACKEND_LOG" 2>&1 &
        echo $! > "$BACKEND_PID_FILE"
        echo -e "   ${GREEN}✓${NC} 后端已启动 (PID: $!, 日志: logs/backend.log)"

        wait_port "$API_PORT" "后端 API" "$(cat "$BACKEND_PID_FILE")"

        # 后台模式 — 前端 dev server
        if [ "$DEV_MODE" = true ] && has_node; then
            echo -e "\n${CYAN}🎨 启动前端开发服务 (端口: $DEV_PORT)...${NC}"
            cd frontend
            nohup npm run dev -- --port "$DEV_PORT" > "$FRONTEND_LOG" 2>&1 &
            echo $! > "$FRONTEND_PID_FILE"
            echo -e "   ${GREEN}✓${NC} 前端已启动 (PID: $!, 日志: logs/frontend.log)"
            cd "$SCRIPT_DIR"
            wait_port "$DEV_PORT" "前端 Dev Server" "$(cat "$FRONTEND_PID_FILE")" 20
        fi

        # 启动完成
        echo ""
        echo -e "${GREEN}══════════════════════════════════════════${NC}"
        echo -e "${GREEN} ✅ 服务已启动！${NC}"
        echo -e "${GREEN}══════════════════════════════════════════${NC}"
        if [ "$DEV_MODE" = true ]; then
            echo -e " 🔗 前端 (dev):  http://127.0.0.1:$DEV_PORT"
            echo -e " 🔗 后端 API:    http://127.0.0.1:$API_PORT"
        else
            echo -e " 🔗 Web 界面:    http://127.0.0.1:$API_PORT"
        fi
        echo -e " 📖 Swagger UI:  http://127.0.0.1:$API_PORT/docs"
        echo ""
        echo -e " ${CYAN}查看日志:${NC}    ./start.sh logs"
        echo -e " ${CYAN}查看状态:${NC}    ./start.sh status"
        echo -e " ${CYAN}停止服务:${NC}    ./start.sh stop"
        echo -e " ${CYAN}重启服务:${NC}    ./start.sh restart"
        echo ""
    fi
}

# ================================================================
#  命令: stop
# ================================================================
do_stop() {
    create_dirs
    cleanup_all
}

# ================================================================
#  命令: restart
# ================================================================
do_restart() {
    echo -e "${CYAN}🔄 重启服务...${NC}"
    do_stop
    # 等待 OS 完全释放文件描述符和锁（Milvus Lite 需要）
    sleep 3
    do_start
}

# ================================================================
#  命令: status
# ================================================================
do_status() {
    create_dirs

    echo -e "${CYAN}📊 服务状态${NC}"
    echo "─────────────────────────────────"

    if is_running "$BACKEND_PID_FILE"; then
        local bp
        bp=$(cat "$BACKEND_PID_FILE")
        echo -e " 后端 API:      ${GREEN}运行中${NC} (PID: $bp)"
    else
        echo -e " 后端 API:      ${RED}未运行${NC}"
    fi

    if is_running "$FRONTEND_PID_FILE"; then
        local fp
        fp=$(cat "$FRONTEND_PID_FILE")
        echo -e " 前端 Dev:      ${GREEN}运行中${NC} (PID: $fp)"
    else
        echo -e " 前端 Dev:      ${RED}未运行${NC}（生产模式无需前端进程）"
    fi

    echo "─────────────────────────────────"

    # 端口检查
    local api_port=${API_PORT:-8000}
    if [ -f .env ]; then
        local p
        p=$(grep -E '^API_PORT=' .env 2>/dev/null | cut -d'=' -f2 | tr -d ' ')
        [ -n "$p" ] && api_port="$p"
    fi

    if curl -s "http://127.0.0.1:$api_port/health" > /dev/null 2>&1; then
        echo -e " 端口 $api_port:     ${GREEN}可访问${NC}"
    else
        echo -e " 端口 $api_port:     ${RED}不可访问${NC}"
    fi
}

# ================================================================
#  命令: logs
# ================================================================
do_logs() {
    local target="${1:-all}"

    if [ "$target" = "backend" ]; then
        if [ -f "$BACKEND_LOG" ]; then
            echo -e "${CYAN}━━━ 后端日志 (最近 50 行) ━━━${NC}"
            tail -n 50 "$BACKEND_LOG"
        else
            echo -e "${YELLOW}⚠  后端日志文件不存在${NC}"
        fi
    elif [ "$target" = "frontend" ]; then
        if [ -f "$FRONTEND_LOG" ]; then
            echo -e "${CYAN}━━━ 前端日志 (最近 50 行) ━━━${NC}"
            tail -n 50 "$FRONTEND_LOG"
        else
            echo -e "${YELLOW}⚠  前端日志文件不存在${NC}"
        fi
    else
        if [ -f "$BACKEND_LOG" ]; then
            echo -e "${CYAN}━━━ 后端日志 (最近 30 行) ━━━${NC}"
            tail -n 30 "$BACKEND_LOG"
            echo ""
        fi
        if [ -f "$FRONTEND_LOG" ]; then
            echo -e "${CYAN}━━━ 前端日志 (最近 30 行) ━━━${NC}"
            tail -n 30 "$FRONTEND_LOG"
        fi
    fi

    echo ""
    echo -e "${CYAN}💡 实时跟踪日志:${NC}"
    echo -e "   后端:  ${YELLOW}tail -f logs/backend.log${NC}"
    echo -e "   前端:  ${YELLOW}tail -f logs/frontend.log${NC}"
    echo -e "   全部:  ${YELLOW}tail -f logs/backend.log logs/frontend.log${NC}"
}

# ================================================================
#  参数解析
# ================================================================
COMMAND=""
INSTALL_DEPS=false
BUILD_FRONTEND=false
DEV_MODE=false
FOREGROUND=false

# 第一个非 -- 开头的参数作为命令
for arg in "$@"; do
    case $arg in
        start|stop|restart|status|logs)
            COMMAND="$arg"
            ;;
        --install)
            INSTALL_DEPS=true
            ;;
        --build-frontend)
            BUILD_FRONTEND=true
            ;;
        --dev)
            DEV_MODE=true
            ;;
        --foreground)
            FOREGROUND=true
            ;;
        -h|--help)
            echo "用法: ./start.sh <命令> [选项]"
            echo ""
            echo "命令:"
            echo "  start              启动服务（后台运行）"
            echo "  stop               停止所有服务"
            echo "  restart            重启服务"
            echo "  status             查看运行状态"
            echo "  logs [backend|frontend]  查看日志"
            echo ""
            echo "选项 (仅 start/restart):"
            echo "  --install          安装/更新 Python 依赖"
            echo "  --build-frontend   构建前端"
            echo "  --dev              开发模式（前端热更新）"
            echo "  --foreground       前台运行（调试用）"
            echo ""
            echo "示例:"
            echo "  ./start.sh start                       # 启动（后台）"
            echo "  ./start.sh start --install --build-frontend  # 首次使用"
            echo "  ./start.sh stop                        # 停止"
            echo "  ./start.sh restart                     # 重启"
            echo "  ./start.sh logs                        # 查看日志"
            echo "  ./start.sh logs backend                # 仅查看后端日志"
            echo "  ./start.sh status                      # 查看状态"
            exit 0
            ;;
        *)
            if [ -z "$COMMAND" ]; then
                echo -e "${YELLOW}⚠  未知命令 '$arg'，使用 --help 查看帮助${NC}"
                exit 1
            fi
            ;;
    esac
done

# 默认命令
if [ -z "$COMMAND" ]; then
    echo -e "${YELLOW}请指定命令，例如:${NC}"
    echo -e "  ${CYAN}./start.sh start${NC}    启动服务"
    echo -e "  ${CYAN}./start.sh stop${NC}     停止服务"
    echo -e "  ${CYAN}./start.sh restart${NC}  重启服务"
    echo -e "  ${CYAN}./start.sh status${NC}   查看状态"
    echo -e "  ${CYAN}./start.sh logs${NC}     查看日志"
    echo -e "\n使用 ${CYAN}./start.sh --help${NC} 查看完整帮助"
    exit 0
fi

# 执行命令
case "$COMMAND" in
    start)   do_start   ;;
    stop)    do_stop    ;;
    restart) do_restart ;;
    status)  do_status  ;;
    logs)    do_logs "$@" ;;
esac
