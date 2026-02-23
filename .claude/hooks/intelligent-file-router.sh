#!/bin/bash
# intelligent-file-router.sh
# Advanced hook for intelligent routing of ALL file operations to appropriate AI models
# Analyzes file characteristics, operation complexity, and content to select optimal model

set -euo pipefail

# Configuration files
CONFIG_FILE="$HOME/.claude/config/thresholds.conf"
LOG_FILE="$HOME/.claude/logs/file-router.log"

# Load configuration
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
else
    # Default thresholds
    LARGE_FILE_THRESHOLD=500
    HUGE_FILE_THRESHOLD=2000
    GEMINI_FLASH_TIMEOUT=30
    GEMINI_PRO_TIMEOUT=60
fi

# Enhanced patterns for different types of analysis
COMPLEX_CODE_PATTERNS="class|function|def|async|interface|type|algorithm|optimization|refactor|architecture|design.pattern"
SECURITY_PATTERNS="auth|token|password|key|secret|crypto|hash|encrypt|security|vulnerability"
PERFORMANCE_PATTERNS="cache|optimize|performance|memory|cpu|query|database|index|scale"
API_PATTERNS="endpoint|route|api|rest|graphql|middleware|request|response"

# Logging function with timestamps
log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Create log directory if needed
    mkdir -p "$(dirname "$LOG_FILE")"
    
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
    echo "[$level] $message" >&2
}

# Enhanced JSON parsing with error handling
parse_json() {
    local key="$1"
    local default="$2"
    echo "$json_input" | jq -r ".$key // \"$default\"" 2>/dev/null || echo "$default"
}

# Read JSON input from Claude Code
json_input=$(cat)

# Extract operation details with better error handling
tool_name=$(parse_json "tool_name" "unknown")
file_path=$(parse_json "tool_input.file_path" "")
proposed_content=$(parse_json "tool_input.content" "")
old_string=$(parse_json "tool_input.old_string" "")
new_string=$(parse_json "tool_input.new_string" "")

log "INFO" "Hook triggered - tool: $tool_name, file: $file_path"

# Only process file-related operations
case "$tool_name" in
    "Edit"|"Write"|"MultiEdit"|"NotebookEdit")
        ;;
    *)
        log "INFO" "Non-file operation ($tool_name), allowing Claude to proceed"
        exit 0
        ;;
esac

# Skip if no file path provided
if [[ -z "$file_path" || "$file_path" == "null" ]]; then
    log "INFO" "No file path provided, allowing Claude to proceed"
    exit 0
fi

# Get comprehensive file information
get_file_info() {
    local path="$1"
    
    if [[ -f "$path" ]]; then
        file_size=$(wc -l < "$path" 2>/dev/null || echo "0")
        file_ext="${path##*.}"
        file_type=$(file -b "$path" 2>/dev/null || echo "unknown")
    else
        # New file case - analyze proposed content
        file_size=$(echo "$proposed_content" | wc -l)
        file_ext="${path##*.}"
        file_type="new-file"
    fi
}

get_file_info "$file_path"
log "INFO" "File analysis - size: $file_size lines, ext: $file_ext, type: $file_type"

# Enhanced content analysis
analyze_content_complexity() {
    local content="$1"
    local complexity_score=0
    
    # Check for various complexity indicators
    [[ "$content" =~ $COMPLEX_CODE_PATTERNS ]] && ((complexity_score += 10))
    [[ "$content" =~ $SECURITY_PATTERNS ]] && ((complexity_score += 15))
    [[ "$content" =~ $PERFORMANCE_PATTERNS ]] && ((complexity_score += 10))
    [[ "$content" =~ $API_PATTERNS ]] && ((complexity_score += 8))
    
    # Check for nested structures (rough indicator)
    nested_braces=$(echo "$content" | grep -o '{' | wc -l)
    ((complexity_score += nested_braces / 5))
    
    echo $complexity_score
}

# Route to appropriate model with enhanced error handling and our wrapper
route_to_model() {
    local model="$1"
    local reason="$2"
    local analysis_prompt="$3"
    
    log "INFO" "Routing to $model: $reason"
    
    case "$model" in
        "gemini-flash")
            route_to_gemini_flash "$analysis_prompt" "$reason"
            ;;
        "gemini-pro")
            route_to_gemini_pro "$analysis_prompt" "$reason"
            ;;
        "parallel-analysis")
            route_to_parallel_analysis "$analysis_prompt" "$reason"
            ;;
        *)
            log "WARN" "Unknown model: $model, falling back to Claude"
            exit 0
            ;;
    esac
}

# Gemini Flash routing using our enhanced wrapper
route_to_gemini_flash() {
    local prompt="$1"
    local reason="$2"
    
    echo "🔍 **Delegating to Gemini Flash** ($reason)" >&2
    echo "Task: Use gemini-flash to $prompt" >&2
    echo "" >&2
    
    log "INFO" "Delegating to gemini-flash: $reason"
    
    # Suggest delegation to Claude instead of blocking
    exit 1  # Return non-zero to signal intervention, but let Claude decide
}

# Gemini Pro routing for strategic analysis
route_to_gemini_pro() {
    local prompt="$1"
    local reason="$2"
    
    echo "🧠 **Delegating to Gemini Pro** ($reason)" >&2
    echo "Task: Use gemini-pro to $prompt" >&2
    echo "" >&2
    
    log "INFO" "Delegating to gemini-pro: $reason"
    
    # Suggest delegation to Claude instead of blocking
    exit 1  # Return non-zero to signal intervention, but let Claude decide
}

# Parallel analysis routing for complex multi-faceted operations
route_to_parallel_analysis() {
    local prompt="$1"
    local reason="$2"
    
    echo "💡 **Delegating to Parallel Coordinator** ($reason)" >&2
    echo "Task: Use parallel-coordinator to break this into simultaneous analyses:" >&2
    echo "- Task 1: Use gemini-flash to analyze file structure and patterns" >&2
    echo "- Task 2: Use gemini-pro to assess architectural implications" >&2
    echo "- Task 3: Use gemini-flash to identify potential conflicts or issues" >&2
    echo "" >&2
    
    log "INFO" "Delegating to parallel-coordinator: $reason"
    
    # Suggest delegation to Claude instead of blocking
    exit 1  # Return non-zero to signal intervention, but let Claude decide
}

# Enhanced decision-making logic
make_intelligent_routing_decision() {
    # Analyze all available content
    all_content="$proposed_content $old_string $new_string"
    complexity_score=$(analyze_content_complexity "$all_content")
    
    log "INFO" "Complexity analysis - score: $complexity_score"
    
    # Critical security operations - always route to Pro for careful analysis
    if [[ "$all_content" =~ $SECURITY_PATTERNS ]]; then
        route_to_model "gemini-pro" \
            "security-related changes detected (score: $complexity_score)" \
            "analyze this security-related code change. Assess potential vulnerabilities, security implications, and recommend best practices. Pay special attention to authentication, authorization, data protection, and potential attack vectors."
        return
    fi
    
    # Huge files require Flash's large context
    if [[ $file_size -gt $HUGE_FILE_THRESHOLD ]]; then
        route_to_model "gemini-flash" \
            "large file requiring extensive context analysis ($file_size lines)" \
            "analyze this large file ($file_size lines) and the proposed changes. Consider file structure, maintainability, performance implications, and how the changes fit within the broader codebase architecture."
        return
    fi
    
    # Complex architectural changes - route to Pro
    if [[ $complexity_score -gt 25 ]]; then
        route_to_model "gemini-pro" \
            "high complexity architectural changes (score: $complexity_score)" \
            "analyze these complex changes with architectural implications. Consider design patterns, code quality, maintainability, performance, and long-term system evolution. Provide strategic guidance on implementation approach."
        return
    fi
    
    # Multi-faceted operations - consider parallel analysis
    if [[ $file_size -gt $LARGE_FILE_THRESHOLD && $complexity_score -gt 15 ]]; then
        route_to_model "parallel-analysis" \
            "large complex file requiring multi-faceted analysis (size: $file_size, complexity: $complexity_score)" \
            "comprehensively analyze this large, complex file operation"
        return
    fi
    
    # Performance-critical code - route to Flash for comprehensive analysis
    if [[ "$all_content" =~ $PERFORMANCE_PATTERNS ]]; then
        route_to_model "gemini-flash" \
            "performance-related changes detected" \
            "analyze these performance-related changes. Identify potential bottlenecks, optimization opportunities, resource usage patterns, and scalability implications. Consider both immediate and long-term performance impacts."
        return
    fi
    
    # API-related changes - consider Flash for comprehensive endpoint analysis
    if [[ "$all_content" =~ $API_PATTERNS && $file_size -gt 100 ]]; then
        route_to_model "gemini-flash" \
            "API-related changes in substantial file" \
            "analyze these API changes comprehensively. Review endpoint design, request/response patterns, error handling, authentication, rate limiting, and integration implications."
        return
    fi
    
    # Large files with moderate complexity - route to Flash
    if [[ $file_size -gt $LARGE_FILE_THRESHOLD ]]; then
        route_to_model "gemini-flash" \
            "large file requiring context-aware analysis ($file_size lines)" \
            "analyze this file and proposed changes within the broader context. Ensure changes maintain code quality, consistency, and don't introduce unintended side effects."
        return
    fi
    
    # Default: let Claude handle normally
    log "INFO" "File meets criteria for normal Claude processing (size: $file_size, complexity: $complexity_score)"
    exit 0
}

# Execute intelligent routing decision
make_intelligent_routing_decision

# Fallback (should never reach here)
log "WARN" "Unexpected fallback, allowing Claude to proceed"
exit 0