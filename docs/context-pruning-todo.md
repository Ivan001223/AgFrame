# Context Pruning TODO

## Now

- [x] 在检索阶段接入 candidate pruning
- [x] 在 prompt 组装阶段接入 prompt pruning
- [x] 支持 `heuristic` / `reranker` / `auto` 三种裁剪方法
- [x] 在工作台展示 candidate/prompt 两层裁剪统计
- [x] 记录每层 `saved chars` 与 `saved %`

## Next

- [ ] 基准评测：比较 `heuristic` / `reranker` / `auto` 的耗时与节省量
- [ ] 追踪落盘：把 pruning telemetry 纳入测试报告或运营报表
- [ ] 管理入口：在设置页显示当前 pruning method 与阈值
- [ ] 质量评估：抽样对比不同 pruning method 对回答质量的影响
- [ ] 降级策略：当 reranker 超时或不可用时记录 fallback 原因
- [ ] 数据集化：准备一批真实知识库片段用于稳定复现实验

## Later

- [ ] 针对代码块、日志块、表格块做类型感知裁剪
- [ ] 把 focus hint 自动生成从启发式升级为结构化 planner 输出
- [ ] 引入 repo/file 级别的局部上下文预算分配
