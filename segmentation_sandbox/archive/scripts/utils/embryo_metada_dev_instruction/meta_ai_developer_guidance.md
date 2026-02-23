# Meta AI Developer Guidance

This document outlines best practices for steering AI-assisted development in VS Code. Use these guidelines to keep your agents organized, efficient, and focused on core functionality.

---

## 1. Plan in Phases and Modules  
- Break work into discrete phases (e.g., setup, core features, integration, testing, optimization).  
- Within each phase, group related functionality into modules.  

## 2. Keep an Implementation Log  
- Record each completed task with date, file(s) modified, and a brief description.  
- Before starting a new phase, review the log to confirm completed steps and remaining work.  

## 3. Generate and Follow Instructions  
- For each module or feature, write a short “implementation instructions” snippet.  
- Let the agent execute one step at a time, then pause to verify results.  

## 4. Minimize Compute and I/O  
- Avoid loading entire folders or huge files; parse only what’s needed.  
- Use intelligent file queries (e.g., `Path.glob`, search by function name) instead of brute-force reads.  

## 5. Focus on Core Functionality  
- Prioritize correctness of core APIs before adding bells and whistles.  
- Defer non-critical enhancements (e.g., performance tuning) until after basic features pass tests.  

## 6. Incremental Testing  
- After implementing or modifying a piece of functionality, run focused unit tests.  
- Only when unit tests pass, run integration tests on real data.  

## 7. Code Reviews and Cleanups  
- At the end of each phase, reorganize files:  
  - Move archived or experimental scripts into an `/archive` folder.  
  - Consolidate core utilities and parsers into `/scripts/utils`.  
- Update the project overview document to reflect any structural changes.  

---

### Example Prompt for Your AI Agents

```
You are an AI developer in VS Code.  
1. Read the implementation log and identify the next unfinished task.  
2. Load only the target file or module needed for that task.  
3. Modify or add code according to the step’s instructions.  
4. Run the relevant unit test to confirm success.  
5. Append the outcome to the implementation log.  
6. Pause and wait for the next instruction.
```

Use this meta-level guidance to coordinate multiple AI agents efficiently and keep your codebase clean and well-documented.<!-- filepath: /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/scripts/utils/embryo_metada_dev_instruction/meta_ai_developer_guidance.md -->

# Meta AI Developer Guidance

This document outlines best practices for steering AI-assisted development in VS Code. Use these guidelines to keep your agents organized, efficient, and focused on core functionality.

---

## 1. Plan in Phases and Modules  
- Break work into discrete phases (e.g., setup, core features, integration, testing, optimization).  
- Within each phase, group related functionality into modules.  

## 2. Keep an Implementation Log  
- Record each completed task with date, file(s) modified, and a brief description.  
- Before starting a new phase, review the log to confirm completed steps and remaining work.  

## 3. Generate and Follow Instructions  
- For each module or feature, write a short “implementation instructions” snippet.  
- Let the agent execute one step at a time, then pause to verify results.  

## 4. Minimize Compute and I/O  
- Avoid loading entire folders or huge files; parse only what’s needed.  
- Use intelligent file queries (e.g., `Path.glob`, search by function name) instead of brute-force reads.  

## 5. Focus on Core Functionality  
- Prioritize correctness of core APIs before adding bells and whistles.  
- Defer non-critical enhancements (e.g., performance tuning) until after basic features pass tests.  

## 6. Incremental Testing  
- After implementing or modifying a piece of functionality, run focused unit tests.  
- Only when unit tests pass, run