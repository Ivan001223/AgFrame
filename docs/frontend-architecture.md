# AgFrame Frontend Architecture Design

<div align="center">
  <a href="frontend-architecture-cn.md">中文文档</a>
</div>

## 1. Architecture Overview

The AgFrame frontend is dedicated to building a **high-performance, production-oriented AI operations and interactive workbench**. Unlike traditional conversational demos, this workbench adopts a highly cohesive, loosely coupled layered architecture design, integrating workflow scheduling, lightweight Hybrid RAG, long-term memory management, and operational observability into a unified modern interactive interface.

The frontend fully utilizes Server-Side Rendering (SSR) and the modern React ecosystem, deferring business rules and state validation to the server side. The frontend focuses on the ultimate experience of **Orchestration**, **Rendering**, and **Interaction State**.

## 2. Core Technology Stack

Based on the current best engineering practices, the AgFrame frontend adopts the following modern technology stack:

- **Core Framework**: [Next.js 15 (App Router)](https://nextjs.org/) provides routing orchestration and server-side rendering (SSR/RSC) support.
- **Development Language**: [TypeScript](https://www.typescriptlang.org/) ensures type safety and the rigor of domain models.
- **UI Library**: Based on React 19, combined with [Tailwind CSS](https://tailwindcss.com/) to implement an atomic and highly customizable styling engine, with underlying components built on the headless UI library [Radix UI](https://www.radix-ui.com/).
- **Data Flow and Caching**: [TanStack Query (React Query)](https://tanstack.com/query/latest) is responsible for fetching, caching, and synchronizing server state.
- **Client State**: [Zustand](https://github.com/pmndrs/zustand) handles lightweight temporary UI states (e.g., sidebar collapse, temporary queues).
- **Forms and Validation**: Leveraging [React Hook Form](https://react-hook-form.com/) and [Zod](https://zod.dev/) to build complex, high-performance dynamic forms and strong client-side validation.
- **Data Visualization**: Combined with [TanStack Table](https://tanstack.com/table/latest) to efficiently render massive data tables, and using [Recharts](https://recharts.org/) to present various data diagnostics and statistical reports.

## 3. Layered Architecture Design

From the perspective of logic and responsibility isolation, the frontend architecture is planned into four progressively overlaid layers:

### 3.1 App Shell Layer
Responsible for the skeleton of the entire frontend application, including route distribution, permission control (Auth Bootstrap), global navigation bar, and global Error Boundaries.

### 3.2 Feature Layer
Vertically split by product business, each sub-module (e.g., knowledge base, task queue, chat panel) independently coalesces its own unique page container and user workflow composition logic.

### 3.3 Domain Layer
The core area docking with backend microservices/APIs. It is responsible for defining and outputting strictly typed API clients, abstracting View Models (to avoid pages directly handling backend response formats), and encapsulating custom Hooks for Queries/Mutations.

### 3.4 Shared Layer
Contains the system's globally common atomic component system, such as responsive table components, task status badges, universal file drag-and-drop area support components, and highly consistent business-level error presentation placeholder layers.

## 4. Product Domain Modules

To support huge backend management capabilities, the entire workbench is divided into the following core domain modules:

- 🧠 **Chat Workbench**: Integrates a streaming dialogue system based on the LangServe protocol, supporting interrupt approval and follow-up questions.
- 📚 **Knowledge Base & RAG Control Center**: Responsible for asynchronous queue ingestion of documents, Hybrid RAG index management, document preview, and rebuild operations.
- ⚡ **Task and Event Operations**: Oriented towards asynchronous task observation in high-concurrency systems, providing task failure diagnostics, event fallback flow scheduling, and active retry tracking.
- 👥 **Memory Console**: Responsible for managing user preferences and permission control, able to modify the long-term conversational profile features built by the underlying LLM.
- 💬 **Conversation Center**: Management of dialogue history fragments and audits.
- ⚙️ **System and Security Settings**: Provides dynamic environment prompt allocation strategies and enterprise/personal level security risk control configuration panels.

## 5. State Management Philosophy

To prevent state chaos in frontend single-page applications, AgFrame adopts the best practice pattern of **"State Segregation"**:

- **Server State**: Relies on TanStack Query to request and cache external data sources, implementing efficient automatic refresh based on Invalidation and Polling strategies for documents, task states, conversation history, etc.
- **UI State**: Some independent states with extremely short lifecycles generated by the application itself (such as dropdown box opening, filter button activation, etc.) are delegated to Zustand for modular management, ensuring the global scope is not polluted.
- **Form State**: Form values are not polluted into the global Store; they are uniformly and locally controlled in a closed loop by React Hook Form until submitted to the Domain Client layer.

## 6. HTTP Specifications and Observability Pipeline

### 6.1 Unified Fetch Encapsulation
Global control of security token injection operations through a set of custom underlying HTTP client instances. Intercepts unified network errors and abstracts them into strongly typed `ApiError`. This fundamentally eliminates the problem of distributed handling of `401 Unauthorized` authentication failures or network interruptions across various pages.

### 6.2 Full-Link Observability and Buried Point Reservation
From the file upload task triggered by the frontend to the end of persistence scheduling, the frontend participates in tracking and tracing throughout the process. It not only passes through custom `X-Request-ID` but also throws complex event actions such as upload failures and storage delays into the underlying event analysis components, thereby forming a complete exception retrospective monitoring view from the client side to the server side.