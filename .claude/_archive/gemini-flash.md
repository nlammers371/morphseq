#!/bin/bash
# gemini-flash - Direct command interface for Gemini Flash agent
# Usage: gemini-flash "your analysis request"

set -euo pipefail

# Configuration
GEMINI_WRAPPER="$HOME/.claude/bin/gemini-wrapper.sh"
LOG_FILE="$HOME/.claude/logs/command-usage.log"

# Logging function
log_usage() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local request="$1"
    
    mkdir -p "$(dirname "$LOG_FILE")"
    echo "[$timestamp] [gemini-flash] $request" >> "$LOG_FILE"
}

# Help function
show_help() {
    cat << EOF
🔍 Gemini Flash - High-Speed Large-Context Analysis

USAGE:
    gemini-flash "your analysis request"
    gemini-flash [OPTIONS] "your analysis request"

OPTIONS:
    -h, --help      Show this help message
    -v, --verbose   Enable verbose output
    -t, --timeout   Override default timeout (30s)
    
EXAMPLES:
    gemini-flash "Find all authentication patterns in this codebase"
    gemini-flash "Analyze React hooks usage across all components"
    gemini-flash "Detect performance bottlenecks and optimization opportunities"
    gemini-flash "Catalog all API endpoints with request/response patterns"
    gemini-flash "Identify security vulnerabilities and code quality issues"

STRENGTHS:
    • 1M+ token context window for comprehensive analysis
    • Fast execution (~30 seconds)
    • Pattern detection and bulk analysis
    • Cost-effective for large-scale searches
    
USE CASES:
    • Large codebase searches and pattern detection
    • Dependency analysis and cataloging  
    • Code quality scans and vulnerability detection
    • Multi-file analysis and cross-reference identification
    • Performance bottleneck detection across entire projects

For strategic analysis or architectural guidance, use: gemini-pro
For multi-faceted analysis, use: parallel-coordinator
EOF
}

# Parse arguments
VERBOSE=false
TIMEOUT=""
ANALYSIS_REQUEST=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        *)
            ANALYSIS_REQUEST="$1"
            shift
            ;;
    esac
done

# Validate input
if [[ -z "$ANALYSIS_REQUEST" ]]; then
    echo "Error: Please provide an analysis request" >&2
    echo "Usage: gemini-flash \"your analysis request\"" >&2
    echo "Use 'gemini-flash --help' for more information" >&2
    exit 1
fi

# Log the usage
log_usage "$ANALYSIS_REQUEST"

# Prepare the command
WRAPPER_ARGS=("-m" "gemini-2.5-flash" "--all-files")

if [[ -n "$TIMEOUT" ]]; then
    WRAPPER_ARGS+=("-t" "$TIMEOUT")
fi

if $VERBOSE; then
    WRAPPER_ARGS+=("-v")
fi

# Construct the analysis prompt
ANALYSIS_PROMPT="$ANALYSIS_REQUEST

Please provide comprehensive analysis with:
1. Specific file paths and line numbers where relevant
2. Clear categorization of findings
3. Actionable recommendations
4. Severity assessment for any issues identified
5. Context about how findings relate to overall codebase architecture"

# Execute the analysis
echo "🔍 Executing Gemini Flash Analysis..."
echo "Request: $ANALYSIS_REQUEST"
echo "======================================="

if [[ -x "$GEMINI_WRAPPER" ]]; then
    "$GEMINI_WRAPPER" "${WRAPPER_ARGS[@]}" -p "$ANALYSIS_PROMPT"
else
    # Fallback to direct gemini command
    if $VERBOSE; then
        echo "Note: Using direct gemini command (wrapper not found)" >&2
    fi
    
    TIMEOUT_CMD=""
    if [[ -n "$TIMEOUT" ]]; then
        TIMEOUT_CMD="timeout $TIMEOUT"
    else
        TIMEOUT_CMD="timeout 30"
    fi
    
    $TIMEOUT_CMD gemini "${WRAPPER_ARGS[@]}" -p "$ANALYSIS_PROMPT"
fi

echo "======================================="
echo "✅ Gemini Flash Analysis Complete"

if $VERBOSE; then
    echo "Analysis logged to: $LOG_FILE"
fi