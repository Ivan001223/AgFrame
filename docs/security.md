# Security Notes

<div align="center">
  <a href="security-cn.md">中文文档</a>
</div>

## Configuration Baseline

- `auth.secret_key` must be a random value with length >= 32
- `database.password` must not use insecure defaults and should be at least 8 characters when set
- `server.cors_allow_credentials=true` cannot be used with `server.cors_origins=["*"]`
- never commit real API keys, database passwords, or production connection strings

## Startup Validation

Run the same validation used by application startup:

```bash
./.venv/bin/python -c "from app.infrastructure.config.settings import settings; settings.validate_security(); print('security-ok')"
```

Validation behavior:

- rejects insecure JWT secrets
- rejects insecure database passwords
- warns when `llm.api_key` is empty
- rejects invalid CORS credential configuration

## Security Scan

```bash
./.venv/bin/python scripts/security_scan.py --out reports/security.json
```

Gate rules:

- missing required security tools -> fail
- high-severity findings or dependency vulnerabilities -> fail
- otherwise -> pass

## Release Checklist

- verify startup validation passes
- verify `.env` and `configs/config.json` do not contain production secrets committed to git
- verify Docker Compose overrides do not reintroduce weak defaults
- verify documentation and changelog reflect any security-sensitive config changes
