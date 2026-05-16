#!/bin/bash
# Phase I 项目跨用户名迁移脚本.
# 在 *新 Mac* 上, 把项目和 Claude Code 对话记录从旧用户名适配过来.
#
# 使用方法:
#   1. 把整个项目目录 (含 .claude/projects/*) 复制到新 Mac
#   2. 改下面两个变量 (OLD_USER, NEW_USER)
#   3. 给执行权限: chmod +x migrate_to_new_user.sh
#   4. 跑: ./migrate_to_new_user.sh
#
# 脚本会修改:
#   A. ~/.claude/projects/ 目录名 (让 claude /resume 找得到对话)
#   B. ~/.claude/projects/<dir>/*.jsonl 文件内容 (历史里的旧路径)
#   C. train.sh (4 处硬编码路径)
#   D. vla_runner.py (1 处 DEFAULT_CHECKPOINT)
#
# 不会改: config.yaml (USB 串口名你得手动改) / Python venv (重装即可)

set -euo pipefail

# ===== 改这两行 =====
OLD_USER="chenyuying"
NEW_USER="$(whoami)"   # 自动读新 Mac 的当前用户名
# ===================

PROJECT_DIR="$HOME/Downloads/lerobot/challenge_pkg/robot_zhenbang_challenge"

echo "=========================================="
echo "  Phase I 跨用户名迁移"
echo "  OLD: $OLD_USER"
echo "  NEW: $NEW_USER"
echo "  Project: $PROJECT_DIR"
echo "=========================================="
echo ""

if [ "$OLD_USER" = "$NEW_USER" ]; then
    echo "✅ 用户名相同, 无需迁移. 退出."
    exit 0
fi

# -----------------------------------------------------------
# A. Claude Code 对话目录改名
# -----------------------------------------------------------
echo "[A] 迁移 Claude Code 对话历史..."

OLD_CLAUDE_DIR="$HOME/.claude/projects/-Users-${OLD_USER}-Downloads-lerobot"
NEW_CLAUDE_DIR="$HOME/.claude/projects/-Users-${NEW_USER}-Downloads-lerobot"

if [ -d "$OLD_CLAUDE_DIR" ]; then
    if [ -d "$NEW_CLAUDE_DIR" ]; then
        echo "  ⚠️  新路径已存在: $NEW_CLAUDE_DIR"
        echo "      备份为 ${NEW_CLAUDE_DIR}.bak.$(date +%s)"
        mv "$NEW_CLAUDE_DIR" "${NEW_CLAUDE_DIR}.bak.$(date +%s)"
    fi
    mv "$OLD_CLAUDE_DIR" "$NEW_CLAUDE_DIR"
    echo "  ✓ 重命名: -Users-${OLD_USER}-... → -Users-${NEW_USER}-..."
else
    echo "  ⚠️  没找到旧 Claude 对话目录, 跳过"
    echo "      (你是否复制过 ~/.claude/projects/ ?)"
fi

# -----------------------------------------------------------
# B. jsonl 内容里的旧路径
# -----------------------------------------------------------
if [ -d "$NEW_CLAUDE_DIR" ]; then
    echo ""
    echo "[B] 替换 jsonl 文件内的旧路径..."
    COUNT=$(find "$NEW_CLAUDE_DIR" -name "*.jsonl" 2>/dev/null | wc -l | tr -d ' ')
    echo "  找到 $COUNT 个 jsonl 文件"

    if [ "$COUNT" -gt 0 ]; then
        # macOS sed 用 -i ''  (空字符串是必须的)
        find "$NEW_CLAUDE_DIR" -name "*.jsonl" -exec \
            sed -i '' "s|/Users/${OLD_USER}|/Users/${NEW_USER}|g" {} +
        echo "  ✓ 已替换所有 /Users/${OLD_USER} → /Users/${NEW_USER}"
    fi
fi

# -----------------------------------------------------------
# C. train.sh
# -----------------------------------------------------------
echo ""
echo "[C] 改 train.sh..."
if [ -f "$PROJECT_DIR/train.sh" ]; then
    cp "$PROJECT_DIR/train.sh" "$PROJECT_DIR/train.sh.bak"
    sed -i '' "s|/Users/${OLD_USER}|/Users/${NEW_USER}|g" "$PROJECT_DIR/train.sh"
    echo "  ✓ train.sh 改完 (备份: train.sh.bak)"
else
    echo "  ⚠️  train.sh 不存在, 跳过"
fi

# -----------------------------------------------------------
# D. vla_runner.py
# -----------------------------------------------------------
echo ""
echo "[D] 改 vla_runner.py..."
if [ -f "$PROJECT_DIR/vla_runner.py" ]; then
    cp "$PROJECT_DIR/vla_runner.py" "$PROJECT_DIR/vla_runner.py.bak"
    sed -i '' "s|/Users/${OLD_USER}|/Users/${NEW_USER}|g" "$PROJECT_DIR/vla_runner.py"
    echo "  ✓ vla_runner.py 改完 (备份: vla_runner.py.bak)"
else
    echo "  ⚠️  vla_runner.py 不存在, 跳过"
fi

# -----------------------------------------------------------
# 验证
# -----------------------------------------------------------
echo ""
echo "=========================================="
echo "  验证 — 检查还有没有残留 /Users/${OLD_USER}"
echo "=========================================="
echo ""

echo "[项目代码]"
REMAINING=$(grep -rln "/Users/${OLD_USER}" "$PROJECT_DIR" 2>/dev/null | grep -v __pycache__ | grep -v '\.bak$' || true)
if [ -z "$REMAINING" ]; then
    echo "  ✅ 项目代码干净"
else
    echo "  ⚠️  以下文件仍有旧路径:"
    echo "$REMAINING" | sed 's/^/      /'
fi

echo ""
echo "[Claude 对话]"
if [ -d "$NEW_CLAUDE_DIR" ]; then
    JSONL_COUNT=$(find "$NEW_CLAUDE_DIR" -name "*.jsonl" 2>/dev/null | wc -l | tr -d ' ')
    REMAINING_JSONL=$(grep -l "/Users/${OLD_USER}" "$NEW_CLAUDE_DIR"/*.jsonl 2>/dev/null | wc -l | tr -d ' ')
    echo "  jsonl 文件总数: $JSONL_COUNT"
    echo "  仍含旧路径的:   $REMAINING_JSONL  (应该是 0)"
fi

# -----------------------------------------------------------
# 还需要手动做什么
# -----------------------------------------------------------
echo ""
echo "=========================================="
echo "  ⚠️  还需要你手动做的事"
echo "=========================================="
cat <<EOF

  1. config.yaml — 改 USB 串口名:
     ls /dev/tty.usbmodem*
     # 把输出的路径填到 robot.port 那一行

  2. 环境变量:
     export DASHSCOPE_API_KEY="sk-..."
     (加到 ~/.zshrc 永久生效)

  3. 重装 Python 环境:
     cd $PROJECT_DIR
     python -m venv .venv
     source .venv/bin/activate
     pip install -r requirements.txt
     pip install lerobot

  4. (可选) 重新校准:
     如果机械臂搬位置了, 跑 python calibrate.py 重做 calibration.npz

  5. 测试 Phase I:
     python -c "import primitives, cap_agent; print('Phase I import OK')"
     python main.py

  6. 验证对话恢复:
     claude        # 然后按 Ctrl+R 或 /resume

EOF
echo "=========================================="
echo "  ✅ 自动迁移部分完成"
echo "=========================================="
