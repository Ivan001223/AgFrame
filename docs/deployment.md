# 部署最小文档

## 1. 前置条件

- Python 3.11.15
- 建议固定使用仓库声明的 Python 版本（见 `.python-version`）
- uv
- Docker 与 Docker Compose
- PostgreSQL/Redis 端口可用（本地默认 `5432`、`6379`）

## 2. 初始化配置

```bash
uv python install 3.11.15
uv sync --no-dev
cp configs/config.example.json configs/config.json
```

如果当前部署需要本地 embedding / reranker / OCR 模型，再补装：

```bash
uv sync --no-dev --group local-inference
```

如果需要高精度 PDF/Office 文档解析，再补装：

```bash
uv sync --no-dev --group document-ai
```

必须设置以下项：

- `auth.secret_key`：至少 32 位随机字符串
- `database.url`：生产库连接串
- `database.password`：强密码
- `llm.api_key`：云模型调用凭证（若使用云模型）

## 3. 启动依赖

```bash
docker-compose up -d
docker-compose ps
```

## 4. 启动服务

```bash
uv run python -m app.server.main
```

默认地址：

- API: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`

## 5. 停止服务

```bash
docker-compose down
```
