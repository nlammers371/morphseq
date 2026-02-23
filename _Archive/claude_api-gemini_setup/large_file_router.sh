#!/bin/bash
# intelligent-large-file-router.sh
# Hook script for automatically routing large file operations to appropriate models
# Place in ~/.claude/hooks/ and configure in settings.json

set -euo pipefail

# Configuration thresholds
LARGE_FILE_THRESHOLD=500        # Lines threshold for "large" files
HUGE_FILE_THRESHOLD=2000        # Lines threshold for "huge" files requiring Gemini Flash
COMPLEX_CODE_PATTERNS="class|function|def|async|interface|type"
MAX_GEMINI_TIMEOUT=30           # Seconds to wait for Gemini response

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >&2
}

# Read JSON input from Claude Code
json_input=$(cat)

# Extract tool information
tool_name=$(echo "$json_input" | jq -r '.tool_name // "unknown"')
file_path=$(echo "$json_input" | jq -r '.tool_input.file_path // empty')
proposed_content=$(echo "$json_input" | jq -r '.tool_input.content // empty')

log "Hook triggered for tool: $tool_name, file: $file_path"

# Only process file-related operations
case "$tool_name" in
    "Edit"|"Write"|"MultiEdit")
        ;;
    *)
        log "Non-file operation, allowing Claude to proceed"
        exit 0
        ;;
esac

# Skip if no file path provided
if [[ -z "$file_path" || "$file_path" == "null" ]]; then
    log "No file path provided, allowing Claude to proceed"
    exit 0
fi

# Get file information
if [[ -f "$file_path" ]]; then
    file_size=$(wc -l < "$file_path" 2>/dev/null || echo "0")
    file_ext="${file_path##*.}"
else
    # New file case - analyze proposed content
    file_size=$(echo "$proposed_content" | wc -l)
    file_ext="${file_path##*.}"
fi

log "File size: $file_size lines, extension: $file_ext"

# Decision logic for routing
route_to_model() {
    local model="$1"
    local reason="$2"
    local analysis_prompt="$3"
    
    log "Routing to $model: $reason"
    
    case "$model" in
        "gemini-flash")
            route_to_gemini_flash "$analysis_prompt" "$reason"
            ;;
        "codex")
            route_to_codex "$analysis_prompt" "$reason"
            ;;
        "gemini-pro")
            route_to_gemini_pro "$analysis_prompt" "$reason"
            ;;
        *)
            log "Unknown model: $model, falling back to Claude"
            exit 0
            ;;
    esac
}

# Gemini Flash routing for large context analysis
route_to_gemini_flash() {
    local prompt="$1"
    local reason="$2"
    
    echo "🔄 Analyzing with Gemini Flash ($reason)" >&2
    
    # Prepare context for Gemini
    local full_context=""
    if [[ -f "$file_path" ]]; then
        full_context="Current file content:\n$(cat "$file_path")\n\n"
    fi
    full_context+="Proposed changes:\n$proposed_content\n\n$prompt"
    
    # Call Gemini Flash with timeout
    local gemini_result
    if gemini_result=$(timeout $MAX_GEMINI_TIMEOUT gemini -p "$full_context" 2>/dev/null); then
        echo "📊 Gemini Flash Analysis Complete:" >&2
        echo "$gemini_result" >&2
        echo "" >&2
        echo "💡 Recommendation: Consider this analysis before proceeding with the edit." >&2
        exit 2  # Block Claude's original action
    else
        log "Gemini Flash failed or timed out, falling back to Claude"
        exit 0
    fi
}

# Codex routing for complex code operations
route_to_codex() {
    local prompt="$1"
    local reason="$2"
    
    echo "🤖 Consulting Codex ($reason)" >&2
    
    # Prepare code context
    local code_context=""
    if [[ -f "$file_path" ]]; then
        code_context="File: $file_path\n$(cat "$file_path")\n\nProposed changes:\n$proposed_content"
    else
        code_context="New file: $file_path\nContent:\n$proposed_content"
    fi
    
    # Call Codex (assuming codex CLI is available)
    local codex_result
    if command -v codex >/dev/null 2>&1; then
        if codex_result=$(codex --suggest "$prompt" --context "$code_context" 2>/dev/null); then
            echo "🔍 Codex Code Review:" >&2
            echo "$codex_result" >&2
            echo "" >&2
            echo "✅ Please review Codex suggestions before implementing." >&2
            exit 2  # Block Claude's original action
        else
            log "Codex failed, falling back to Claude"
            exit 0
        fi
    else
        log "Codex CLI not available, falling back to Claude"
        exit 0
    fi
}

# Gemini Pro routing for strategic analysis
route_to_gemini_pro() {
    local prompt="$1"
    local reason="$2"
    
    echo "🧠 Consulting Gemini Pro ($reason)" >&2
    
    local strategic_context="File: $file_path\nSize: $file_size lines\nOperation: $tool_name\n\n$prompt"
    
    local gemini_result
    if gemini_result=$(timeout $MAX_GEMINI_TIMEOUT gemini -p "$strategic_context" --model=gemini-2.5-pro 2>/dev/null); then
        echo "🎯 Gemini Pro Strategic Analysis:" >&2
        echo "$gemini_result" >&2
        echo "" >&2
        exit 2  # Block Claude's original action
    else
        log "Gemini Pro failed or timed out, falling back to Claude"
        exit 0
    fi
}

# Main routing decision logic
make_routing_decision() {
    # Check for huge files requiring Gemini Flash's large context
    if [[ $file_size -gt $HUGE_FILE_THRESHOLD ]]; then
        route_to_model "gemini-flash" \
            "file exceeds $HUGE_FILE_THRESHOLD lines" \
            "This is a large file ($file_size lines). Analyze the structure, identify key patterns, and suggest the best approach for making changes. Focus on maintainability and potential impacts of modifications."
        return
    fi
    
    # Check for large files that might benefit from analysis
    if [[ $file_size -gt $LARGE_FILE_THRESHOLD ]]; then
        # For code files, consider complexity
        if [[ "$file_ext" =~ ^(py|js|ts|java|cpp|c|h|go|rs|php)$ ]]; then
            # Check if the file contains complex patterns
            if [[ -f "$file_path" ]] && grep -qE "$COMPLEX_CODE_PATTERNS" "$file_path"; then
                route_to_model "codex" \
                    "large code file with complex patterns" \
                    "Review this code for potential improvements, bug fixes, and optimization opportunities. Pay attention to code quality, performance, and maintainability."
                return
            else
                route_to_model "gemini-flash" \
                    "large file requiring context analysis" \
                    "Analyze this file structure and the proposed changes. Ensure the modifications fit well within the existing codebase architecture."
                return
            fi
        else
            # Non-code large files
            route_to_model "gemini-flash" \
                "large non-code file" \
                "Analyze this large file and the proposed changes. Consider document structure, consistency, and overall coherence."
            return
        fi
    fi
    
    # Check for complex code operations regardless of size
    if [[ "$file_ext" =~ ^(py|js|ts|java|cpp|c|h|go|rs|php)$ ]]; then
        # Check if proposed content contains complex patterns
        if echo "$proposed_content" | grep -qE "(algorithm|optimization|refactor|architecture|design.pattern)"; then
            route_to_model "gemini-pro" \
                "complex architectural changes detected" \
                "This appears to involve significant architectural or algorithmic changes. Provide strategic guidance on the best approach, considering long-term maintainability and system design."
            return
        fi
    fi
    
    # Default: let Claude handle normally
    log "File meets criteria for normal Claude processing"
    exit 0
}

# Execute routing decision
make_routing_decision

# Fallback (should never reach here)
log "Unexpected fallback, allowing Claude to proceed"
exit 0