# AgFrame Roadmap

<div align="center">
  <a href="roadmap-cn.md">中文文档</a>
</div>

## P0 (Core Closed Loop)

- [x] Knowledge base management minimal closed loop step 1: Supplement document list, document details, and delete document interfaces
- [x] Conversation center step 1: Conversation search, details, title renaming
- [x] Memory console step 1: View profile, view long-term memory, delete memory items
- [x] Memory console step 2: Manually update profile, manually add new memory items
- [x] Upload pipeline enhancement step 1: Duplicate document prompt
- [x] Upload pipeline enhancement step 2: Task retry interface, refined progress status
- [x] Document management step 2: File name search, content preview, rebuild index
- [x] Health check step 2: Vector database, LLM, Embedding, Hybrid RAG/pruning strategy probes
- [x] Health check enhancement step 1: Database, Redis readiness checks

## P1 (Agent & Operations Enhancement)

- [x] Human approval closed loop: Server-side encapsulated interface for resuming execution after approval, now backed by the current Harness and checkpoint integration
- [ ] Retrieval operations panel: Hit quality, citation quality, failure case replay
- [ ] Document management enhancement: Rebuild index, filter by tag/source, parsing result preview
- [ ] User settings enhancement: Model preferences, answer style, retrieval strategy
- [ ] Context pruning evaluation: Method comparison, savings, time consumption, and quality replay

## P2 (Extension & Management)

- [ ] Agent tool extensions: Structured web scraping, table analysis, controlled code execution
- [ ] Admin backend: Quotas, audit logs, tenant governance, configuration panels
- [ ] End-to-end acceptance scripts: Registration -> Upload -> Retrieval -> Chat -> History -> Memory

## Context Pruning TODO

### Now
- [x] Integrate candidate pruning in the retrieval phase
- [x] Integrate prompt pruning in the prompt assembly phase
- [x] Support three pruning methods: `heuristic` / `reranker` / `auto`
- [x] The `reranker` mode has converged to a lightweight local ranker, achieving model independence
- [x] Display candidate/prompt two-layer pruning statistics in the workbench
- [x] Record `saved chars` and `saved %` for each layer

### Next
- [ ] Benchmark evaluation: compare the time consumption and savings of `heuristic` / `reranker(lightweight)` / `auto`
- [ ] Telemetry persistence: include pruning telemetry in test reports or operation reports
- [ ] Management portal: display the current pruning method and threshold on the settings page
- [ ] Quality evaluation: sample and compare the impact of different pruning methods on answer quality
- [ ] Terminology convergence: gradually migrate the `reranker` naming in UI/docs to `lightweight_ranker`
- [ ] Dataset creation: prepare a batch of real knowledge base fragments for stable reproducible experiments

### Later
- [ ] Implement type-aware pruning for code blocks, log blocks, and table blocks
- [ ] Upgrade automatic focus hint generation from heuristics to structured planner output
- [ ] Introduce repo/file-level local context budget allocation
