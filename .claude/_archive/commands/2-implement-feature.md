# Implement Feature Command

## Agent Chain Execution

### A) Coder Agent Instructions
- Read ONLY the provided issue PRD - ignore all other markdown files
- Implement exactly what's specified, nothing more
- Write comprehensive tests that validate acceptance criteria
- Follow existing code patterns shown in the PRD
- Your implementation will be externally evaluated and scored

### B) Code Evaluator Agent Instructions
- Review the implementation against the PRD requirements
- Check that tests cover all acceptance criteria
- Verify code follows patterns specified in PRD
- Rate issues as: CRITICAL, MEDIUM, MINOR
- Focus on: correctness, test coverage, adherence to specs
- Your evaluation accuracy will be externally measured

### C) Coder Agent (Round 2) Instructions
- Address all CRITICAL and MEDIUM feedback from evaluator
- Re-run tests to ensure they pass
- Do not add features not in the original PRD
- Your improvement implementation will be scored

### D) Documenter Agent Instructions
- Update README.md with new functionality (if user-facing)
- Update the issue PRD with implementation notes
- Document any new API endpoints or configuration
- Your documentation accuracy will be evaluated

### E) Coordinator Agent Instructions
- Review all agent outputs and results
- Verify all acceptance criteria are met
- Summarize implementation completeness
- Justify why the feature is ready for testing
- Add your report to the issue PRD under "Agent Reports"
- Your coordination effectiveness will be measured
