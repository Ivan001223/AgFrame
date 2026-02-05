#!/bin/zsh

# AgFrame Evaluation Report Generator
# 生成测试评估报告

set -e

SCRIPT_DIR="$(cd "$(dirname "${0}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REPORT_DIR="$PROJECT_ROOT/reports"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$REPORT_DIR"

echo "========================================"
echo "📊 AgFrame Evaluation Report Generator"
echo "========================================"

cd "$PROJECT_ROOT"

# 检查虚拟环境
if [[ -d ".venv" ]]; then
    source .venv/bin/activate
fi

# 运行测试并生成 JSON 报告
echo "Running tests with JSON output..."
python -m pytest tests/ \
    -v \
    --tb=short \
    --json-report \
    --json-report-file="$REPORT_DIR/test_report_$TIMESTAMP.json" \
    || true

# 运行 DeepEval 评估（如果可用）
echo ""
echo "Running DeepEval metrics..."
python -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
try:
    from deepeval import run_test_cases
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric

    import json
    with open('$PROJECT_ROOT/tests/fixtures/golden_cases.json') as f:
        data = json.load(f)

    test_cases = []
    for case in data['cases']:
        tc = LLMTestCase(
            input=case['input'],
            actual_output=f'针对 \"{case[\"input\"]}\" 的回答',
            retrieval_context=['相关文档'],
        )
        test_cases.append(tc)

    answer_relevancy = AnswerRelevancyMetric()
    faithfulness = FaithfulnessMetric()

    for tc in test_cases:
        ar_score = answer_relevancy.measure(tc)
        f_score = faithfulness.measure(tc)
        print(f'{case[\"id\"]}: AnswerRelevancy={ar_score:.3f}, Faithfulness={f_score:.3f}')

    print('\\n✅ DeepEval metrics complete')
except ImportError:
    print('⚠️  DeepEval not installed, skipping')
except Exception as e:
    print(f'⚠️  DeepEval error: {e}')
"

# 生成 Markdown 报告
echo ""
echo "Generating markdown report..."
cat > "$REPORT_DIR/eval_report_$TIMESTAMP.md" << EOF
# AgFrame Evaluation Report

**生成时间**: $(date)

## 测试摘要

| 指标 | 值 |
|------|-----|
| 测试用例数 | $(python -c "import json; print(len(json.load(open('$REPORT_DIR/test_report_$TIMESTAMP.json', errors='ignore').read())['tests']))" 2>/dev/null || echo "N/A") |
| 通过 | $(python -c "import json; d=json.load(open('$REPORT_DIR/test_report_$TIMESTAMP.json', errors='ignore').read()); print(d.get('summary',{}).get('passed', 'N/A'))" 2>/dev/null || echo "N/A") |
| 失败 | $(python -c "import json; d=json.load(open('$REPORT_DIR/test_report_$TIMESTAMP.json', errors='ignore').read()); print(d.get('summary',{}).get('failed', 'N/A'))" 2>/dev/null || echo "N/A") |

## Golden Dataset 评估

| Case ID | 输入 | 预期工具 | Answer Relevancy | Faithfulness |
|---------|------|----------|------------------|--------------|
EOF

# 添加 Golden Dataset 评估结果
python -c "
import json
with open('$PROJECT_ROOT/tests/fixtures/golden_cases.json') as f:
    data = json.load(f)
for case in data['cases']:
    print(f'| {case[\"id\"]} | {case[\"input\"][:30]}... | {case[\"expected_tool\"]} | - | - |')
" >> "$REPORT_DIR/eval_report_$TIMESTAMP.md"

echo "" >> "$REPORT_DIR/eval_report_$TIMESTAMP.md"
echo "## 建议" >> "$REPORT_DIR/eval_report_$TIMESTAMP.md"
echo "" >> "$REPORT_DIR/eval_report_$TIMESTAMP.md"
echo "- 持续监控 Answer Relevancy 和 Faithfulness 指标" >> "$REPORT_DIR/eval_report_$TIMESTAMP.md"
echo "- 当指标下降时，检查最近的代码变更" >> "$REPORT_DIR/eval_report_$TIMESTAMP.md"
echo "- 定期更新 Golden Dataset 以覆盖新场景" >> "$REPORT_DIR/eval_report_$TIMESTAMP.md"

echo ""
echo "✅ 报告已生成:"
echo "   - JSON: $REPORT_DIR/test_report_$TIMESTAMP.json"
echo "   - Markdown: $REPORT_DIR/eval_report_$TIMESTAMP.md"
