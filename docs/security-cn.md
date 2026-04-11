# 安全说明

<div align="center">
  <a href="security.md">English</a>
</div>

## 配置基线

- `auth.secret_key` 必须是长度不少于 32 的随机值
- `database.password` 不能使用不安全默认值，且设置时建议至少 8 位
- `server.cors_allow_credentials=true` 不能与 `server.cors_origins=["*"]` 同时使用
- 严禁把真实 API Key、数据库密码或生产连接串提交到仓库

## 启动校验

执行与应用启动相同的安全校验：

```bash
./.venv/bin/python -c "from app.infrastructure.config.settings import settings; settings.validate_security(); print('security-ok')"
```

校验行为包括：

- 拒绝不安全的 JWT 密钥
- 拒绝不安全的数据库密码
- 在 `llm.api_key` 为空时给出警告
- 拒绝错误的 CORS 凭证配置

## 安全扫描

```bash
./.venv/bin/python scripts/security_scan.py --out reports/security.json
```

门禁规则：

- 缺少必要安全工具 -> 失败
- 存在高危问题或依赖漏洞 -> 失败
- 其余情况 -> 通过

## 发布前检查清单

- 确认启动校验通过
- 确认 `.env` 与 `configs/config.json` 未提交生产密钥
- 确认 Docker Compose 覆盖项没有重新引入弱默认值
- 确认安全相关配置变更已同步更新文档与变更记录
