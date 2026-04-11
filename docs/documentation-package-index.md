# 文档包索引

## 审计范围

- 校验日期：`2026-04-08`
- 审计对象：仓库内 19 份项目级文档
- 一级事实源：应用路由、配置模型、依赖清单、Docker Compose、启动脚本、测试脚本
- 维护范围：本页“文档清单”中列出的项目级文档，以及本页列出的审计产物

## 状态说明

- `已校准`：本轮已执行内容修订，并与当前事实源重新对齐
- `已复核`：本轮未改动正文，但已完成事实核对与结构复查

## 文档清单

| 分类 | 文件 | 状态 |
| --- | --- | --- |
| 项目总览 | `README.md` | 已校准 |
| 项目总览 | `README-CN.md` | 已校准 |
| 发布说明 | `CHANGELOG.md` | 已校准 |
| API | `docs/api.md` | 已校准 |
| API | `docs/api-cn.md` | 已校准 |
| 部署 | `docs/deployment.md` | 已校准 |
| 部署 | `docs/deployment-cn.md` | 已校准 |
| 测试 | `docs/testing.md` | 已复核 |
| 测试 | `docs/testing-cn.md` | 已校准 |
| 安全 | `docs/security.md` | 已复核 |
| 安全 | `docs/security-cn.md` | 已复核 |
| 前端架构 | `docs/frontend-architecture.md` | 已复核 |
| 前端架构 | `docs/frontend-architecture-cn.md` | 已复核 |
| RAG 架构 | `docs/rag-architecture.md` | 已复核 |
| RAG 架构 | `docs/rag-architecture-cn.md` | 已复核 |
| 文档治理 | `docs/documentation-governance.md` | 已复核 |
| 文档治理 | `docs/documentation-governance-cn.md` | 已复核 |
| 前端子系统 | `frontend/README.md` | 已校准 |
| 前端子系统 | `frontend/README-CN.md` | 已校准 |

## 本轮重点同步项

- Python 运行时统一为 `>=3.12,<3.13`
- Docker Compose 与手动启动拓扑统一为 backend + 3 类 worker
- API 文档补齐知识库、文档下载、文档知识库绑定、文件访问接口
- 前端文档补齐 `/register`、`/knowledge/[docId]` 与 `npm run typecheck`
- 发布说明补齐当前版本条目与文档巡检产物

## 审计产物

- 说明：以下 3 份文件为本轮文档审计产物，当前按中文单语维护

- 文档包索引：`docs/documentation-package-index.md`
- 变更记录表：`docs/documentation-change-record.md`
- 质量检查报告：`docs/documentation-quality-report.md`
