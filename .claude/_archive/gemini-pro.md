#!/bin/bash
# gemini-pro - Direct command interface for Gemini Pro agent
# Usage: gemini-pro "your strategic analysis request"

set -euo pipefail

# Configuration
GEMINI_WRAPPER="$HOME/.claude/bin/gemini-wrapper.sh"
LOG_FILE="$HOME/.claude/logs/command-usage.log"

# Logging function
log_usage() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local request="$1"
    
    mkdir -p "$(dirname "$LOG_FILE")"
    echo "[$timestamp] [gemini-pro] $request" >> "$LOG_FILE"
}

# Help function
show_help() {
    cat << EOF
🧠 Gemini Pro - Strategic Analysis & Architectural Reasoning

USAGE:
    gemini-pro "your strategic analysis request"
    gemini-pro [OPTIONS] "your strategic analysis request"

OPTIONS:
    -h, --help      Show this help message
    -v, --verbose   Enable verbose output
    -t, --timeout   Override default timeout (60s)
    
EXAMPLES:
    gemini-pro "Analyze the overall architecture and suggest improvements"
    gemini-pro "Develop a refactoring strategy for this legacy codebase"
    gemini-pro "Assess security vulnerabilities and provide remediation plan"
    gemini-pro "Evaluate migration from React to Vue with timeline and risks"
    gemini-pro "Design a scalable microservices architecture for this monolith"

STRENGTHS:
    • Deep strategic reasoning and complex analysis
    • Architectural guidance and design recommendations
    • Risk assessment and mitigation strategies
    • Long-term planning and technical decision support
    
USE CASES:
    • Architectural analysis and system design guidance
    • Strategic refactoring and modernization planning
    • Security architecture review and vulnerability assessment
    • Technology migration strategy and risk evaluation
    • Performance architecture and scalability planning
    • Technical debt assessment and prioritization

For bulk searches and pattern detection, use: gemini-flash
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
    echo "Error: Please provide a strategic analysis request" >&2
    echo "Usage: gemini-pro \"your strategic analysis request\"" >&2
    echo "Use 'gemini-pro --help' for more information" >&2
    exit 1
fi

# Log the usage
log_usage "$ANALYSIS_REQUEST"

# Prepare the command
WRAPPER_ARGS=("-m" "gemini-2.5-pro" "--all-files")

if [[ -n "$TIMEOUT" ]]; then
    WRAPPER_ARGS+=("-t" "$TIMEOUT")
else
    WRAPPER_ARGS+=("-t" "60")  # Default Pro timeout
fi

if $VERBOSE; then
    WRAPPER_ARGS+=("-v")
fi

# Construct the strategic analysis prompt
ANALYSIS_PROMPT="$ANALYSIS_REQUEST

Please provide comprehensive strategic analysis with:
1. High-level architectural overview and assessment
2. Strategic recommendations with clear priorities
3. Risk analysis and mitigation strategies
4. Implementation roadmap with timeline considerations
5. Long-term implications and scalability considerations
6. Trade-offs analysis between different approaches
7. Resource requirements and complexity assessment"

# Execute the analysis
echo "🧠 Executing Gemini Pro Strategic Analysis..."
echo "Request: $ANALYSIS_REQUEST"
echo "======================================="

if [[ -x "$GEMINI_WRAPPER" ]]; then
    "$GEMINI_WRAPPER" "${WRAPPER_ARGS[@]}" -p "$ANALYSIS_PROMPT"
else
    # Fallback to direct gemini command
    if $VERBOSE; then
        echo "Note: Using direct gemini command (wrapper not found)" >&2
    fi
    
    TIMEOUT_CMD="timeout 60"
    if [[ -n "$TIMEOUT" ]]; then
        TIMEOUT_CMD="timeout $TIMEOUT"
    fi
    
    $TIMEOUT_CMD gemini "${WRAPPER_ARGS[@]}" -p "$ANALYSIS_PROMPT"
fi

echo "======================================="
echo "✅ Gemini Pro Strategic Analysis Complete"

if $VERBOSE; then
    echo "Analysis logged to: $LOG_FILE"
fi