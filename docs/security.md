# Security Notes

<div align="center">
  <a href="security-cn.md">中文文档</a>
</div>

## 1. Configuration Baseline

- `auth.secret_key` must be a random value of 32 characters or more.
- `database.password` must be a strong password.
- When `server.cors_allow_credentials=true`, `server.cors_origins` is prohibited from containing `"*"`.
- **Strictly prohibited** to commit real keys and production connection strings in the repository.

## 2. Pre-flight Checks

```bash
uv run python -c "from app.infrastructure.config.settings import settings; settings.validate_security(); print('security-ok')"
```

If there are high-risk configurations, the service will refuse to continue running during the startup phase and exit with an error.

## 3. Security Scan

```bash
uv run python scripts/security_scan.py --out reports/security.json
```

Judgment rules:

- Any security tool missing: Gate failure.
- High-risk issues or dependency vulnerabilities exist: Gate failure.
- All passed: Gate passed.