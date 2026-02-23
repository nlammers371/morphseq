
# Gemini Integration Guide for Claude

This document outlines how to leverage the integrated Gemini models (Flash and Pro) as sub-agents to enhance your capabilities.

## Core Principle: Delegate, Don't Do

Your primary role is to **coordinate and synthesize**. For tasks that are repetitive, large-scale, or require deep strategic analysis, you should delegate to the specialized Gemini sub-agents.

## Sub-Agent Quick Reference

| Sub-Agent              | When to Use                                                                                                 | Example Invocation                                                                      |
|------------------------|-------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| `gemini-flash`         | **Bulk search, code scans, pattern matching, large file analysis.** Fast, cheap, and has a 1M token context. | `Task: Use gemini-flash to find all instances of the deprecated function 'getUser'.`      |
| `gemini-pro`           | **Architectural analysis, strategic refactoring, complex logic.** Slower, more expensive, but deeper reasoning. | `Task: Use gemini-pro to analyze the authentication flow and suggest security improvements.` |
| `parallel-coordinator` | **Multi-faceted tasks that can be broken down.** When you need to search multiple locations or analyze different aspects of the codebase simultaneously. | `Task: Use parallel-coordinator to search for 'API_KEY' in the 'backend' and 'frontend' directories simultaneously.` |

## Your Workflow

1.  **Analyze the User's Request:** Identify the core intent. Is it a simple edit, a broad search, or a request for strategic advice?
2.  **Select the Right Tool:**
    *   **Simple, localized task?** Handle it yourself.
    *   **Large-scale search or analysis?** Delegate to `gemini-flash`.
    *   **Complex, strategic question?** Delegate to `gemini-pro`.
    *   **Multiple, independent sub-tasks?** Delegate to `parallel-coordinator`.
3.  **Formulate the Task:** Clearly define the sub-agent's task. Be specific.
4.  **Synthesize the Results:** Once the sub-agent(s) return their findings, your job is to synthesize them into a single, coherent response for the user.

## Best Practices

*   **Be proactive:** Don't wait for the user to suggest a deep search. If you see the need, delegate.
*   **Think in parallel:** Break down large problems into smaller, parallelizable chunks for the `parallel-coordinator`.
*   **Trust your agents:** The Gemini agents are wrappers for powerful models. Trust their output, but always verify and synthesize.
*   **Manage costs:** Be mindful that `gemini-pro` is more expensive. Use it for tasks that truly require its deep reasoning capabilities.
