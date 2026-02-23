#!/bin/bash
# parallel-coordinator - Direct command interface for Parallel Coordinator agent
# Usage: parallel-coordinator "your multi-faceted analysis request"

set -euo pipefail

# Configuration
LOG_FILE="$HOME/.claude/logs/command-usage.log"

# Logging function
log_usage() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local request="$1"
    
    mkdir -p "$(dirname "$LOG_FILE")"
    echo "[$timestamp] [parallel-coordinator] $request" >> "$LOG_FILE"
}

# Help function
show_help() {
    cat << EOF
💡 Parallel Coordinator - Multi-Agent Orchestration

USAGE:
    parallel-coordinator "your multi-faceted analysis request"
    parallel-coordinator [OPTIONS] "your analysis request"

OPTIONS:
    -h, --help      Show this help message
    -v, --verbose   Enable verbose output
    
EXAMPLES:
    parallel-coordinator "Comprehensive security audit of authentication system"
    parallel-coordinator "Full-stack analysis of payment processing feature" 
    parallel-coordinator "Performance and security review of API layer"
    parallel-coordinator "Technology migration assessment for React to Vue"
    parallel-coordinator "Complete analysis of user management system"

STRENGTHS:
    • Orchestrates multiple sub-agents simultaneously (up to 8 concurrent)
    • Comprehensive multi-aspect analysis
    • Parallel execution for faster results
    • Combines strategic and tactical insights
    
COORDINATION PATTERNS:
    • Multi-directory parallel searches
    • Mixed analysis strategies (Flash + Pro)
    • Comprehensive feature investigations
    • Technology assessment workflows
    
TYPICAL SUB-AGENT BREAKDOWN:
    • 2-4 Gemini Flash tasks for bulk analysis and pattern detection
    • 1-2 Gemini Pro tasks for strategic reasoning and architecture
    • Clear task separation to avoid overlap
    • Structured result aggregation

For simple bulk analysis, use: gemini-flash
For strategic analysis only, use: gemini-pro
EOF
}

# Parse arguments
VERBOSE=false
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
        *)
            ANALYSIS_REQUEST="$1"
            shift
            ;;
    esac
done

# Validate input
if [[ -z "$ANALYSIS_REQUEST" ]]; then
    echo "Error: Please provide a multi-faceted analysis request" >&2
    echo "Usage: parallel-coordinator \"your analysis request\"" >&2
    echo "Use 'parallel-coordinator --help' for more information" >&2
    exit 1
fi

# Log the usage
log_usage "$ANALYSIS_REQUEST"

# Analyze the request and suggest parallel task breakdown
echo "💡 Parallel Coordinator - Multi-Agent Analysis"
echo "Request: $ANALYSIS_REQUEST"
echo "======================================="

# Determine task breakdown based on request content
suggest_task_breakdown() {
    local request="$1"
    
    echo "📋 Suggested Parallel Task Breakdown:"
    echo ""
    
    # Security-focused analysis
    if [[ "$request" =~ [Ss]ecurity|[Aa]uth|[Vv]ulner|[Tt]oken ]]; then
        cat << EOF
Task 1: Use gemini-flash to scan for hardcoded secrets and API keys
Task 2: Use gemini-flash to identify input validation patterns and vulnerabilities
Task 3: Use gemini-pro to analyze overall security architecture and design patterns
Task 4: Use gemini-flash to audit authentication and authorization implementations
Task 5: Use gemini-flash to check for common security anti-patterns
EOF
    
    # Performance-focused analysis  
    elif [[ "$request" =~ [Pp]erformance|[Ss]low|[Oo]ptimi|[Bb]ottleneck ]]; then
        cat << EOF
Task 1: Use gemini-flash to identify database query patterns and N+1 issues
Task 2: Use gemini-flash to analyze frontend rendering and re-render patterns
Task 3: Use gemini-flash to audit API response times and caching strategies
Task 4: Use gemini-pro to develop comprehensive optimization strategy
Task 5: Use gemini-flash to identify memory leaks and resource management
EOF
    
    # Feature analysis
    elif [[ "$request" =~ [Ff]eature|[Ff]ull.stack|[Pp]ayment|[Uu]ser|[Aa]pi ]]; then
        cat << EOF
Task 1: Use gemini-flash to trace feature flow in frontend components
Task 2: Use gemini-flash to analyze backend API endpoints and middleware
Task 3: Use gemini-flash to identify related database operations
Task 4: Use gemini-pro to evaluate feature architecture and design
Task 5: Use gemini-flash to find error handling and validation logic
EOF
    
    # Migration analysis
    elif [[ "$request" =~ [Mm]igrat|[Uu]pgrade|[Mm]odern ]]; then
        cat << EOF
Task 1: Use gemini-flash to catalog current implementation patterns
Task 2: Use gemini-flash to identify technology-specific dependencies
Task 3: Use gemini-pro to develop strategic migration approach
Task 4: Use gemini-flash to assess third-party library compatibility
Task 5: Use gemini-pro to evaluate new ecosystem and timeline
EOF
    
    # General comprehensive analysis
    else
        cat << EOF
Task 1: Use gemini-flash to perform comprehensive pattern analysis
Task 2: Use gemini-flash to identify code quality and structural issues
Task 3: Use gemini-pro to provide strategic architectural assessment
Task 4: Use gemini-flash to analyze dependencies and integrations
Task 5: Use gemini-pro to recommend implementation approach
EOF
    fi
    
    echo ""
    echo "🔧 Execution Approach:"
    echo "1. Tasks 1, 2, 4, 5 run in parallel using Gemini Flash (fast, broad analysis)"
    echo "2. Task 3 uses Gemini Pro for deep strategic reasoning" 
    echo "3. Results aggregated and synthesized into coherent analysis"
    echo "4. Estimated completion: 60-90 seconds"
    echo ""
}

suggest_task_breakdown "$ANALYSIS_REQUEST"

echo "⚡ To execute this analysis, you can:"
echo "1. Copy the suggested tasks and use them with Claude's Task tool"
echo "2. Run individual components with 'gemini-flash' and 'gemini-pro' commands"
echo "3. Ask Claude to \"Use parallel-coordinator for this analysis\""
echo ""

if $VERBOSE; then
    echo "💡 Pro Tip: Parallel coordination works best when:"
    echo "• Tasks are independent and don't require sequential dependencies"
    echo "• Analysis covers multiple aspects (frontend/backend, security/performance)"
    echo "• Comprehensive coverage is more important than deep specialization"
    echo ""
    echo "Analysis logged to: $LOG_FILE"
fi

echo "======================================="
echo "✅ Parallel Coordination Plan Complete"