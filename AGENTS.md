# AGENTS.md — `orbitals` Python Library

This document provides context and guidance for AI coding agents and developers working on this codebase.

## About Orbitals

Orbitals is Principled Intelligence's solution for enforcing AI governance in production AI systems (chatbots, agents, and more). At its core, it provides an ecosystem of modular guardrails—individual control layers that validate user inputs and AI outputs to enforce different dimensions of AI governance.

> **Why "Orbitals"?** It's an analogy with atomic orbitals: just as atomic orbitals surround and protect the nucleus, our guardrails—which we call *orbitals*—surround and protect the AI system.

Orbitals organizes guardrails across the AI interaction lifecycle:

| Type | Role |
|------|------|
| **Input Orbitals** | Filter out harmful content, off-topic requests, jailbreak attempts, or policy-violating inputs before they are processed by the AI |
| **Output Orbitals** | Validate—or redact—AI-generated responses before they reach the user, catching problematic content, factual inaccuracies, and safety or policy violations |

These guardrails are available as a **pay-per-use SaaS**, invokable via REST APIs either singularly or as part of customer-specific end-to-end guardrailing pipelines. Some guardrails are **open-sourced** (and open-weighted) and released within the `orbitals` Python library as part of Principled Intelligence's mission; the library also contains a Python SDK for invoking the hosted REST endpoints.

## About This Project

This repository is the `orbitals` open-source Python library. It serves two complementary purposes:

1. **Open-Source Guardrails**: Fully open-source, open-weight implementations of selected orbitals. These run locally, invoking local LLMs (via vLLM, Hugging Face Pipelines, or other backend).
2. **SDK for Hosted Orbitals**: Lightweight Python wrappers that call the hosted REST endpoints exposed by Principled Intelligence API.

Users of this library interact with both kinds of orbitals through the same consistent interface, regardless of whether execution happens locally or on the hosted service.

## Current Guardrails

### ScopeGuard

Given the specifications of an AI service and a user query, `orbitals.ScopeGuard` classifies the query into one of five classes:

| Class | Meaning |
|-------|---------|
| **Directly Supported** | The query is clearly within the assistant's capabilities |
| **Potentially Supported** | The query could plausibly be handled by the assistant |
| **Out of Scope** | The query is outside the assistant's defined role |
| **Restricted** | The query cannot be handled due to a specific constraint |
| **Chit Chat** | A social interaction not related to the assistant's function |

Optionally, `orbitals.ScopeGuard` can also generate supporting evidence (spans from the AI Service Description) for its decision. 

Open-source release available.

## Project Structure

The library is implemented entirely in Python, using `uv` as the package manager.

```
src/orbitals/
├── cli/                  # Root `orbitals` CLI (Typer app; imports sub-apps from each orbital)
└── scope_guard/          # ScopeGuard orbital
    ├── guards/           # Guard implementations
    │   │                 #   BaseScopeGuard (base class)
    │   │                 #   ScopeGuard (sync: validate + batch_validate)
    │   │                 #   AsyncScopeGuard (async counterpart)
    ├── cli/              # ScopeGuard CLI commands
    └── serving/          # FastAPI serving implementation
```

## Coding Conventions

- Use **Google-style docstrings** for all public functions and classes.
- Use `uv` for all dependency management; do not use `pip` directly.
