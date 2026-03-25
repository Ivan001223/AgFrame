#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REPORT_DIR="$PROJECT_ROOT/reports"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$REPORT_DIR"

echo "========================================"
echo "AgFrame Evaluation Report Generator"
echo "========================================"

cd "$PROJECT_ROOT"

echo "Running tests with JSON output..."
uv run python -m pytest tests/ \
  -v \
  --tb=short \
  --json-report \
  --json-report-file="$REPORT_DIR/test_report_$TIMESTAMP.json" \
  || true

echo ""
echo "Running DeepEval metrics..."
uv run python -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
try:
    from deepeval import run_test_cases
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
    import json
    with open('$PROJECT_ROOT/tests/fixtures/golden_cases.json', encoding='utf-8') as f:
        data = json.load(f)
    test_cases = []
    for case in data['cases']:
        tc = LLMTestCase(
            input=case['input'],
            actual_output=f'Answer for {case[\"input\"]}',
            retrieval_context=['related document'],
        )
        test_cases.append(tc)
    answer_relevancy = AnswerRelevancyMetric()
    faithfulness = FaithfulnessMetric()
    for case, tc in zip(data['cases'], test_cases):
        ar_score = answer_relevancy.measure(tc)
        f_score = faithfulness.measure(tc)
        print(f'{case[\"id\"]}: AnswerRelevancy={ar_score:.3f}, Faithfulness={f_score:.3f}')
    print('\nDeepEval metrics complete')
except ImportError:
    print('DeepEval not installed, skipping')
except Exception as e:
    print(f'DeepEval error: {e}')
"

echo ""
echo "Generating markdown report..."
cat > "$REPORT_DIR/eval_report_$TIMESTAMP.md" << EOF
# AgFrame Evaluation Report

Generated: $(date)

## Test Summary

| Metric | Value |
|------|-----|
| Test cases | $(uv run python -c "import json; print(json.load(open('$REPORT_DIR/test_report_$TIMESTAMP.json', encoding='utf-8')).get('summary', {}).get('total', 'N/A'))" 2>/dev/null || echo "N/A") |
| Passed | $(uv run python -c "import json; print(json.load(open('$REPORT_DIR/test_report_$TIMESTAMP.json', encoding='utf-8')).get('summary', {}).get('passed', 'N/A'))" 2>/dev/null || echo "N/A") |
| Failed | $(uv run python -c "import json; print(json.load(open('$REPORT_DIR/test_report_$TIMESTAMP.json', encoding='utf-8')).get('summary', {}).get('failed', 'N/A'))" 2>/dev/null || echo "N/A") |

## Golden Dataset

| Case ID | Input | Expected Tool | Answer Relevancy | Faithfulness |
|---------|------|----------|------------------|--------------|
EOF

uv run python -c "
import json
with open('$PROJECT_ROOT/tests/fixtures/golden_cases.json', encoding='utf-8') as f:
    data = json.load(f)
for case in data['cases']:
    print(f'| {case[\"id\"]} | {case[\"input\"][:30]}... | {case[\"expected_tool\"]} | - | - |')
" >> "$REPORT_DIR/eval_report_$TIMESTAMP.md"

cat >> "$REPORT_DIR/eval_report_$TIMESTAMP.md" << EOF

## Recommendations

- Monitor answer relevancy and faithfulness over time
- Investigate regressions after retrieval or prompt changes
- Expand the golden dataset as new scenarios are added
EOF

echo ""
echo "Report generated:"
echo "  - JSON: $REPORT_DIR/test_report_$TIMESTAMP.json"
echo "  - Markdown: $REPORT_DIR/eval_report_$TIMESTAMP.md"
