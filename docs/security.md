# 安全最小文档

## 1. 配置基线

- `auth.secret_key` 必须为 32 位以上随机值
- `database.password` 必须为强密码
- `server.cors_allow_credentials=true` 时，`server.cors_origins` 禁止包含 `"*"`
- 不在仓库提交真实密钥与生产连接串

## 2. 启动前自检

```bash
uv run python -c "from app.infrastructure.config.settings import settings; settings.validate_security(); print('security-ok')"
```

若存在高风险配置，服务会在启动阶段拒绝继续运行。

## 3. 安全扫描

```bash
uv run python scripts/security_scan.py --out reports/security.json
```

判定规则：

- 任一安全工具缺失：门禁失败
- 高危问题或依赖漏洞存在：门禁失败
- 全部通过：门禁通过
