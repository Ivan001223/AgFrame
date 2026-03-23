# AgFrame 前端架构设计 (Frontend Architecture)

## 1. 架构概览 (Architecture Overview)

AgFrame 前端致力于构建一个**高性能、面向生产环境的 AI 运维与交互工作台**。有别于传统的对话演示 Demo，本工作台采用了高内聚、低耦合的分层架构设计，将复杂的 AI 调度、文档 RAG 流程与长期记忆管理整合在统一的现代化交互界面中。

前端充分利用了服务端渲染 (SSR) 和现代 React 生态，将业务规则与状态校验后置于服务端，前端专注于 **界面编排 (Orchestration)**、**视图渲染 (Rendering)** 和 **交互状态 (Interaction State)** 的极致体验。

## 2. 核心技术栈 (Technology Stack)

基于当前最优的工程实践，AgFrame 前端采用了以下现代化技术栈：

- **核心框架**: [Next.js 15 (App Router)](https://nextjs.org/) 提供路由编排与服务端渲染（SSR/RSC）支持。
- **开发语言**: [TypeScript](https://www.typescriptlang.org/) 保障类型的安全与领域模型的严谨性。
- **UI 库**: 基于 React 19，结合 [Tailwind CSS](https://tailwindcss.com/) 实现原子化和高度定制化的样式引擎，底层组件构建于无头 UI 库 [Radix UI](https://www.radix-ui.com/)。
- **数据流与缓存**: [TanStack Query (React Query)](https://tanstack.com/query/latest) 负责服务端状态的获取、缓存与同步。
- **客户端状态**: [Zustand](https://github.com/pmndrs/zustand) 处理轻量级的 UI 临时状态（如侧边栏收拢、临时队列等）。
- **表单与校验**: 借助 [React Hook Form](https://react-hook-form.com/) 与 [Zod](https://zod.dev/) 构建复杂、高性能的动态表单和客户端强校验。
- **数据可视化**: 结合 [TanStack Table](https://tanstack.com/table/latest) 高效渲染海量数据表格，并通过 [Recharts](https://recharts.org/) 呈现各类数据诊断和统计报表。

## 3. 分层架构设计 (Layered Design)

前端架构从逻辑与职责隔离的角度，规划为了四个渐进叠加的层次：

### 3.1 应用壳层 (App Shell Layer)
负责整个前端应用的骨架，包括路由分发、权限控制 (Auth Bootstrap)、全局导航栏与全局的错误边界捕捉 (Error Boundaries)。

### 3.2 功能模块层 (Feature Layer)
按产品业务垂直拆分，每个子模块（如：知识库、任务队列、对话面板）独立内聚自己独有的页面容器、用户工作流组合逻辑。

### 3.3 领域层 (Domain Layer)
与后端微服务/API对接的核心地带。负责定义并输出带有严格类型的 API 客户端、抽象出 View Model（视图模型，避免页面直接裸处理后端返回格式）、以及封装 Query/Mutation 的自定义 Hooks。

### 3.4 共享层 (Shared Layer)
包含系统全局通用的原子组件体系，诸如响应式表格组件、任务状态徽章、通用的文件拖拽区域支持组件，以及高度一致的业务级错误展现占位层。

## 4. 产品领域模块 (Domain Modules)

为支持庞大的后台管理能力，整个工作台划分为以下核心领域模块：

- 🧠 **对话工作台 (Chat Workbench)**：集成 LangServe 协议的流式对话系统，支持中断审批与追问。
- 📚 **知识库与 RAG 控制中心 (Knowledge Base)**：负责文档异步队列入库、全量向量索引重建操作与文档快照管理。
- ⚡ **任务与事件运维 (Task Operations)**：面向高并发系统中的异步任务观测，提供任务失败诊断、事件降级流调度与主动重试追踪。
- 👥 **记忆控制台 (Memory Console)**：负责用户偏好的管理与权限控制，可修改底层 LLM 构建出的长期对话画像特征。
- 💬 **会话中心 (Conversation Center)**：对话历史片段和审计的管理。
- ⚙️ **系统与安全配置 (Settings)**：提供动态的环境提示词分配策略和企业/个人级的安全风控配置面板。

## 5. 状态管理理念 (State Management Philosophy)

为了防止前端单页应用中状态的混乱，AgFrame 采用了 **"状态分离" (State Segregation)** 的最佳实践模式：

- **Server State (服务端状态)**：靠 TanStack Query 请求并缓存外部数据源，实现了文档、任务状态、会话历史等基于失效 (Invalidation) 和轮询 (Polling) 策略的高效自动刷新。
- **UI State (视图交互状态)**：应用自身产生的一些生命周期极短的独立状态（如下拉框打开、筛选按钮激活等），下放至 Zustand 中按模块管理，确保不污染全局域。
- **Form State (表单临时状态)**：不把表单值污染进全局 Store，统一在局部的 React Hook Form 进行闭环控制，直至提交给 Domain Client 层。

## 6. HTTP 规范与观测链路 (HTTP & Observability)

### 6.1 统一 Fetch 封装
通过一套自定义的基础 HTTP 客户端实例全局管控安全令牌的注入操作。拦截统一的网络报错并抽象为 `ApiError` 强类型错误。这从根源上杜绝了各页面分散处理 `401 Unauthorized` 认证失效或者网络中断的问题。

### 6.2 全链路观测与埋点预留
从前端触发的文件上传任务开始一直到持久化调度结束，前端全程参与埋点和溯源。不仅透传自定义的 `X-Request-ID`，并将上传失败、入库延迟等复杂事件动作抛送至底层的事件分析组件内，从而形成跨端端侧到服务端的完整异常回溯监控视图。
