#!/bin/zsh
set -e

SCRIPT_DIR="$(cd "$(dirname "${0}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$PROJECT_ROOT/reports/test_suite/$TIMESTAMP"

mkdir -p "$OUT_DIR"
cd "$PROJECT_ROOT"

python -m pytest \
  --json-report \
  --json-report-file="$OUT_DIR/pytest.json" \
  --cov=app.runtime.prompts \
  --cov=app.infrastructure.utils \
  --cov=app.server.api \
  --cov-config=.coveragerc \
  --cov-report=term-missing \
  --cov-report=xml:"$OUT_DIR/coverage.xml" \
  --cov-fail-under=80

python "$PROJECT_ROOT/scripts/perf_bench.py" --out "$OUT_DIR/perf.json"
python "$PROJECT_ROOT/scripts/security_scan.py" --out "$OUT_DIR/security.json"
python "$PROJECT_ROOT/scripts/generate_test_report.py" \
  --pytest-json "$OUT_DIR/pytest.json" \
  --coverage-xml "$OUT_DIR/coverage.xml" \
  --perf-json "$OUT_DIR/perf.json" \
  --security-json "$OUT_DIR/security.json" \
  --out "$OUT_DIR/report.md" \
  --defects "$OUT_DIR/defects.md"

echo "$OUT_DIR"
