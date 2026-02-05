#!/bin/zsh

# AgFrame Evaluation Runner CI Hook
# 自动在 git commit 前运行评估测试

set -e

SCRIPT_DIR="$(cd "$(dirname "${0}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo "🧪 AgFrame Evaluation Runner"
echo "========================================"

# 检查是否有测试变更
if [[ -n "$(git diff --name-only -- 'tests/' '!**/fixtures/' 2>/dev/null)" ]]; then
    echo "📝 检测到测试文件变更，运行集成测试..."

    cd "$PROJECT_ROOT"

    # 检查虚拟环境
    if [[ -d ".venv" ]]; then
        source .venv/bin/activate
    fi

    # 运行测试
    echo "Running pytest on tests/..."
    python -m pytest tests/ -v --tb=short --color=yes

    TEST_EXIT_CODE=$?

    if [[ $TEST_EXIT_CODE -ne 0 ]]; then
        echo "❌ 测试失败，请修复后重试"
        exit 1
    fi

    echo "✅ 所有测试通过"
else
    echo "ℹ️  未检测到测试文件变更，跳过评估"
fi

echo "========================================"
echo "✨ Evaluation Complete"
echo "========================================"
