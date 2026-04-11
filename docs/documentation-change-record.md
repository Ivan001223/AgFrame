# 文档变更记录

## 记录范围

- 记录日期：`2026-04-08`
- 记录目标：对齐现有文档与当前仓库代码、配置、依赖与运行拓扑
- 维护范围：对应 [documentation-package-index.md](./documentation-package-index.md) 中列出的项目级文档

## 变更明细

| 文件 | 问题 | 事实来源 | 修复内容 |
| --- | --- | --- | --- |
| `README.md` | Python 版本与 worker 拓扑滞后 | `pyproject.toml`、`docker-compose.yml`、`scripts/start-worker.sh` | 更新为 Python 3.12 约束，补齐 `worker-ingest`、`worker-resume` 与三类 worker 职责 |
| `README-CN.md` | Python 版本、`.env.example` 默认值、手动启动说明过时 | `pyproject.toml`、`.env.example`、`scripts/start-worker.sh` | 修正 `dev-stub` 默认值，改为脚本化手动启动说明，补齐三类 worker 描述 |
| `docs/deployment.md` | 部署要求与默认服务列表过时 | `pyproject.toml`、`docker-compose.yml` | 更新 Python 约束、默认服务列表与 manual startup 对三类 worker 的要求 |
| `docs/deployment-cn.md` | 中文部署说明未同步三类 worker 与 Python 版本 | `pyproject.toml`、`docker-compose.yml` | 同步 Python 3.12、三类 worker 与手动启动约束 |
| `docs/api.md` | 缺少知识库、文件访问、文档下载与知识库绑定接口 | `app/server/api/*.py`、前端 hooks | 补齐 `/knowledge-bases`、`/files/*`、`/uploads/*`、`GET /documents/{doc_id}/download`、`PUT /documents/{doc_id}/knowledge-base` |
| `docs/api-cn.md` | 中文 API 文档存在同类缺口 | `app/server/api/*.py`、前端 hooks | 同步补齐中文接口说明 |
| `docs/testing-cn.md` | 前端校验缺少 `npm run typecheck` | `frontend/package.json`、`scripts/run_test_suite.sh` | 补齐 typecheck 门禁步骤 |
| `frontend/README.md` | 路由与接口接入说明不完整 | `frontend/src/app`、`frontend/src/domains/*/hooks.ts` | 补齐 `/register`、`/knowledge/[docId]`、知识库接口与文档下载/绑定接口 |
| `frontend/README-CN.md` | 中文前端说明不完整 | `frontend/src/app`、`frontend/src/domains/*/hooks.ts` | 同步补齐中文路由和接口说明 |
| `CHANGELOG.md` | 发布记录停留在 `0.1.1` | `pyproject.toml`、`frontend/package.json`、当前文档集 | 新增 `0.3.1` 发布条目并移除旧的发布前占位描述 |

## 未修改但已复核

以下文档在本轮未发生正文调整，但已完成事实核对与结构复查：

- `docs/testing.md`
- `docs/security.md`
- `docs/security-cn.md`
- `docs/frontend-architecture.md`
- `docs/frontend-architecture-cn.md`
- `docs/rag-architecture.md`
- `docs/rag-architecture-cn.md`
- `docs/documentation-governance.md`
- `docs/documentation-governance-cn.md`
