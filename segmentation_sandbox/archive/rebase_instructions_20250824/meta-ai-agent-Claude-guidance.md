# Enhanced Meta AI Developer Guidance for Efficient Token and Tool Usage

This document provides optimized strategies for AI-assisted development that minimize token usage while maximizing productivity.

---

## Summary

The key to improving these classes is to focus on fixing the critical bugs first, then gradually enhancing functionality. The proposed changes maintain the useful features (like EntityIDTracker and metadata integration) while simplifying the implementation and improving reliability.

Remember: A working simple solution is better than a broken complex one. Start simple, validate it works, then enhance as needed.
## Core Principles (Universal for All Tasks)

### 1. **Maintain an Implementation Log**
```
[DATE-TIME|FILE|ACTION] Brief description
[2025-01-31-14:32|embryo_metadata.py|ADDED] init_phenotype_tracking() - validates phenotype entries
[2025-01-31-15:45|utils.py|MODIFIED] batch_processor() - added snip_id range syntax
[2025-01-31-16:20|test_embryo.py|ADDED] test cases for range syntax parser
```


### 3. **Plan in Phases and Modules**
- Break work into discrete phases (setup → core features → integration → testing)
- Within each phase, group related functionality into modules
- Assign token budgets to each module

### 4. **Smart File Access**
```python
# BAD: Load entire directories or files
load_all_files_in('./src/')

# GOOD: Targeted queries
find_methods_containing('validate', file='class_name.py')
extract_class_signatures('./src/target_file.py')
```

### 5. **Generate Patterns, Not Full Implementations**
- Create interfaces and method signatures first
- Use pseudocode for complex logic
- Reference existing patterns rather than recreating

### 6. **Reuse Existing Code**
- **Always check for existing implementations first**
- Copy and adapt code that's already been generated/tested
- Reference proven patterns from other classes in the codebase
- When instructed to use existing code, load ONLY that specific implementation
```python
# Example: Reusing validation patterns
# GOOD: "Copy the validation logic from ExperimentQC.validate_flag()"
# BAD: "Reimplement validation from scratch"
```

### 6. **Reuse Existing Code**
- **Always check for existing implementations first**
- Copy and adapt code that's already been generated/tested
- Reference proven patterns from other classes in the codebase
- When instructed to use existing code, load ONLY that specific implementation
```python
# Example: Reusing validation patterns
# GOOD: "Copy the validation logic from ExperimentQC.validate_flag()"
# BAD: "Reimplement validation from scratch"
```

### 7. **Efficient Testing Strategy**
- Generate test templates that can be parameterized
- Run only tests affected by recent changes
- Create test stubs before implementations

### 8. **Context Preservation Between Tasks**
- Start each session by loading working_memory.json
- Update working memory after each major task
- Use "checkpoint summaries" every 5-10 tasks

---

## Anti-Patterns to Avoid

❌ Loading entire codebases to find one function  
❌ Generating complete implementations when pseudocode suffices  
❌ Rewriting code that already exists in the codebase  
❌ Repeating context in every prompt  
❌ Creating verbose documentation inline  
❌ Running all tests after every change  

✅ Use surgical file access  
✅ Generate interfaces and patterns  
✅ Reuse and adapt existing code  
✅ Maintain persistent working memory  
✅ Link to external docs  
✅ Run targeted test subsets  

---

## Generic Prompt Templates

### Efficient Agent Prompt Template:
```
Role: Code implementation agent
Context: [Load working_memory.json]
Task: [Specific module from task decomposition]
Constraints:
1. Load only files mentioned in task
2. Use existing patterns from [reference_class]
3. Generate pseudocode for complex logic
4. Update working_memory.json after completion
Output: Minimal diff + updated test stub
```

### Task Decomposition Template:
```yaml
Task: [Main objective]
Modules:
  1. Core Structure (X tokens):
     - Skeleton/interfaces
     - Import statements
  2. Business Logic (Y tokens):
     - Key algorithms (pseudocode)
     - Validation rules
  3. Integration (Z tokens):
     - API connections
     - Data flow
```

---

## Project-Specific Examples

### Example 1: EmbryoMetadata Class Implementation

#### Phase Planning:
```yaml
Phase 1: Core Structure (100 tokens)
  - Class skeleton with section markers [phenotype], [genotype], [flag], [source], [config]
  - Method signatures only
  - Import statements

Phase 2: Storage API (150 tokens)
  - CRUD operations (reference ExperimentQC patterns)
  - Batch processing interface
  - Persistence layer

Phase 3: Validation Logic (200 tokens)
  - Phenotype validators (DEAD exclusivity)
  - Genotype safeguards (overwrite protection)
  - Flag consistency checks

Phase 4: Advanced Features (150 tokens)
  - Range syntax ("23::" notation)
  - Inheritance from GroundedSamAnnotation
  - Config tracking
```

#### Efficient Task Instructions:
```
Task: Add range syntax to phenotype batch processing
Context needed: 
- EmbryoMetadata.add_phenotype() signature
- ExperimentQC.batch_add_flag() pattern
Do NOT load: Full class implementations, unrelated utilities
Expected output: 
1. Parse function for "23::" syntax
2. Integration point in add_phenotype()
3. Unit test stub
```

### Example 2: Pipeline Integration Task

#### Working Memory for Pipeline:
```json
{
  "pipeline_stages": {
    "step01": "ExperimentMetadata - raw data tracking",
    "step02": "ExperimentQC - quality control",
    "step03": "GdinoAnnotation - detection storage",
    "step04": "GroundedSamAnnotation - embryo tracking",
    "step05": "EmbryoMetadata - phenotype/genotype storage"
  },
  "current_integration": "step04->step05",
  "key_interfaces": {
    "embryo_id": "primary key from GroundedSamAnnotation",
    "snip_id": "temporal tracking key"
  }
}
```

#### Integration Pattern:
```python
# Instead of loading all pipeline classes:
# Extract only the output interface of previous step
previous_output_schema = extract_method_signature(
    'GroundedSamAnnotation.get_embryo_data'
)

# Generate adapter pattern
def create_embryo_metadata_adapter():
    """
    Converts GroundedSamAnnotation output to EmbryoMetadata input.
    Pattern: Similar to step03->step04 adapter
    """
    # Pseudocode for mapping
```

### Example 3: Validation System Implementation

#### Token-Efficient Validation Design:
```python
# Generate validation interface first (20 tokens)
class ValidationInterface:
    def validate_phenotype(self, value: str) -> bool: pass
    def validate_genotype(self, value: str) -> bool: pass
    def validate_flag(self, value: str, level: str) -> bool: pass

# Reference existing patterns (10 tokens)
"""
Copy validation pattern from ExperimentQC.validate_flag()
Add DEAD phenotype exclusivity check
"""

# Pseudocode for complex logic (30 tokens)
"""
DEAD phenotype validation:
1. if phenotype == 'DEAD':
   - check no other phenotypes exist for embryo
   - set immutable flag
2. if existing == 'DEAD' and not overwrite_dead:
   - reject new phenotype
"""
```

---

## Metrics for Success

- **Tokens per feature**: Aim for <500 tokens per complete feature
- **File loads per task**: Target <3 file loads per implementation task
- **Context reuse**: >70% of context should carry between related tasks
- **Test efficiency**: <20% of total test suite per validation run