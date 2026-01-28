# AGENTS.md Guide for `orbitals`

This document provides guidance for code agents and developers working on the `orbitals` library.

## What is `orbitals`?

`orbitals` is an ecosystem of LLM guardrails, developed at Principled Intelligence.

### What is `orbitals.ScopeGuard`?

Given the specifications of an AI assistant and a user query, `orbitals.ScopeGuard` assigns a label for the user query, chosen from the following five classes:
*   **Directly Supported**: The query is clearly within the assistant's capabilities.
*   **Potentially Supported**: The query could plausibly be handled by the assistant.
*   **Out of Scope**: The query is outside the assistant's defined role.
*   **Restricted**: The query cannot be handled due to a specific constraint.
*   **Chit Chat**: The query is a social interaction not related to the assistant's function.
Optionally, `orbitals.ScopeGuard` can also generate supporting evidences for its decision.

This is done by prompting **local LLMs**.
    
## Project Structure

The API implementation is entirely in Python, using `uv` as the package manager. We are adopting a mono-repo approach, with the `packages/` folder structured as follows:

/src/transformers: 
/models: 

* `src/orbitals/`: This contains the core source code for the library.
    * `src/orbitals/cli`: code for root `orbitals` cli. It's a Typer app that imports and adds the apps defined by each orbital.
    * `src/orbitals/scope_guard`:
        * `src/orbitals/scope_guard/guards/`: Code for individual scope guards. Starting from the `BaseScopeGuard` root class, we find two subclasses, `ScopeGuard` (sync guard and batch-guard methods) and `AsyncScopeGuard` (async counterpart).
        * `src/orbitals/scope_guard/cli/`: This contains the CLI commands of `scope-guard`.
        * `src/orbitals/scope_guard/serving/`: This contains the FastAPI serving implementation.

## Coding Conventions

* Use Google-style syntax for documentation