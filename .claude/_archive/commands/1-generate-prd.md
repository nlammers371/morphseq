# Generate Issue PRD Command

## Purpose
Create a standalone PRD document for a specific issue that contains ONLY what's needed to implement it.

## Agent Workflow
1. **Issue Analyzer Agent**: Parse issue description, extract requirements
2. **Code Context Agent**: Find relevant code patterns and existing implementations
3. **Documentation Agent**: Gather only relevant docs and references
4. **PRD Generator Agent**: Create focused, implementation-ready document

## Instructions for All Agents
- CRITICAL: Include ONLY information directly relevant to this specific issue
- NO background stories, future improvements, or unrelated context
- Focus on what a junior developer needs to complete THIS task
- Your performance will be externally evaluated and scored

## Output Location
Save final PRD to: `docs/new-issues/issue-[number]-[short-name]-prd.md`
