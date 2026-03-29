# AgFrame 路线图 (Roadmap)

## P0 (核心闭环)

- [x] 知识库管理最小闭环第一步：补充文档列表、文档详情、删除文档接口
- [x] 会话中心第一步：会话搜索、详情、标题重命名
- [x] 记忆控制台第一步：查看画像、查看长期记忆、删除记忆项
- [x] 记忆控制台第二步：手动更新画像、手动新增记忆项
- [x] 上传链路增强第一步：重复文档提示
- [x] 上传链路增强第二步：任务重试接口、进度状态细化
- [x] 文档管理第二步：文件名搜索、内容预览、重建索引
- [x] 健康检查第二步：向量库、LLM、Embedding、Hybrid RAG/裁剪策略探针
- [x] 健康检查增强第一步：数据库、Redis 就绪检查

## P1 (Agent & 运营增强)

- [x] 人工审批闭环：审批后恢复执行的服务端封装接口，当前已由 Harness 与 checkpoint 集成链路承载
- [ ] 检索运营面板：命中质量、引用质量、失败问题回放
- [ ] 文档管理增强：重建索引、按标签/来源筛选、解析结果预览
- [ ] 用户设置增强：模型偏好、回答风格、检索策略
- [ ] Context pruning 评测：方法对比、节省量、耗时与质量回放

## P2 (扩展与管理)

- [ ] Agent 工具扩展：结构化网页抓取、表格分析、受控代码执行
- [ ] 管理后台：配额、审计日志、租户治理、配置面板
- [ ] 端到端验收脚本：注册 -> 上传 -> 检索 -> 对话 -> 历史 -> 记忆

## Context Pruning TODO

### Now
- [x] 在检索阶段接入 candidate pruning
- [x] 在 prompt 组装阶段接入 prompt pruning
- [x] 支持 `heuristic` / `reranker` / `auto` 三种裁剪方法
- [x] `reranker` 模式已收敛为轻量本地 ranker，实现无模型依赖
- [x] 在工作台展示 candidate/prompt 两层裁剪统计
- [x] 记录每层 `saved chars` 与 `saved %`

### Next
- [ ] 基准评测：比较 `heuristic` / `reranker(lightweight)` / `auto` 的耗时与节省量
- [ ] 追踪落盘：把 pruning telemetry 纳入测试报告或运营报表
- [ ] 管理入口：在设置页显示当前 pruning method 与阈值
- [ ] 质量评估：抽样对比不同 pruning method 对回答质量的影响
- [ ] 术语收敛：将 UI/文档中的 `reranker` 命名逐步迁移到 `lightweight_ranker`
- [ ] 数据集化：准备一批真实知识库片段用于稳定复现实验

### Later
- [ ] 针对代码块、日志块、表格块做类型感知裁剪
- [ ] 把 focus hint 自动生成从启发式升级为结构化 planner 输出
- [ ] 引入 repo/file 级别的局部上下文预算分配
