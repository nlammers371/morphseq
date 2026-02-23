#!/bin/bash
# keyword-detector.sh
# Intelligent keyword detection hook for suggesting appropriate sub-agent delegation

set -euo pipefail

# Read user prompt from stdin
USER_PROMPT=$(cat)

# Configuration
KEYWORDS_FILE="$HOME/.claude/config/keywords.conf"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load keywords from config file
if [ -f "$KEYWORDS_FILE" ]; then
    SEARCH_KEYWORDS=$(cat "$KEYWORDS_FILE")
else
    # Default keywords if config doesn't exist
    SEARCH_KEYWORDS="search
analyze
find
scan
look for
what are
how does
identify
detect
catalog
audit
review
trace
investigate"
fi

# Function to check for bulk analysis indicators
check_bulk_analysis() {
    local prompt="$1"
    
    # Check for file quantity indicators
    if echo "$prompt" | grep -qi -E "(all|every|entire|complete|comprehensive|full|across|throughout).*file"; then
        return 0
    fi
    
    # Check for multi-directory indicators
    if echo "$prompt" | grep -qi -E "(directory|directories|folder|codebase|project|repo)"; then
        return 0
    fi
    
    # Check for pattern matching across codebase
    if echo "$prompt" | grep -qi -E "(pattern|patterns|everywhere|globally|system-wide)"; then
        return 0
    fi
    
    return 1
}

# Function to check for strategic analysis indicators  
check_strategic_analysis() {
    local prompt="$1"
    
    # Check for architectural terms
    if echo "$prompt" | grep -qi -E "(architect|architecture|design|strategy|strategic|refactor|migration|security)"; then
        return 0
    fi
    
    # Check for high-level analysis terms
    if echo "$prompt" | grep -qi -E "(overview|analysis|assessment|evaluation|recommendation|approach|planning)"; then
        return 0
    fi
    
    # Check for complex reasoning indicators
    if echo "$prompt" | grep -qi -E "(complex|complicated|sophisticated|advanced|comprehensive|in-depth)"; then
        return 0
    fi
    
    return 1
}

# Function to check for parallel coordination indicators
check_parallel_coordination() {
    local prompt="$1"
    
    # Check for multiple aspect indicators
    if echo "$prompt" | grep -qi -E "(and|also|both|multiple|various|different|several)"; then
        return 0
    fi
    
    # Check for directory separation
    if echo "$prompt" | grep -qi -E "(frontend.*backend|backend.*frontend|client.*server|api.*ui)"; then
        return 0
    fi
    
    # Check for multi-faceted analysis
    if echo "$prompt" | grep -qi -E "(performance.*security|security.*performance|flow.*validation|component.*data)"; then
        return 0
    fi
    
    return 1
}

# Analyze the prompt and generate suggestions
SUGGESTIONS=""

# Check if prompt contains search/analysis keywords
KEYWORD_MATCH=false
while IFS= read -r keyword; do
    if echo "$USER_PROMPT" | grep -qi "$keyword"; then
        KEYWORD_MATCH=true
        break
    fi
done <<< "$SEARCH_KEYWORDS"

# Generate suggestions based on analysis
if $KEYWORD_MATCH; then
    if check_parallel_coordination "$USER_PROMPT"; then
        SUGGESTIONS="$SUGGESTIONS
💡 This looks like a multi-faceted task that could benefit from parallel analysis. Consider using the 'parallel-coordinator' sub-agent to break this into simultaneous tasks."
    elif check_strategic_analysis "$USER_PROMPT"; then
        SUGGESTIONS="$SUGGESTIONS
🧠 This appears to be a strategic or architectural question. The 'gemini-pro' sub-agent excels at complex reasoning and high-level analysis."
    elif check_bulk_analysis "$USER_PROMPT"; then
        SUGGESTIONS="$SUGGESTIONS
🔍 This looks like a large-scale search or analysis task. The 'gemini-flash' sub-agent can handle comprehensive codebase analysis with its 1M token context."
    fi
fi

# Check for file operation patterns that suggest bulk analysis
if echo "$USER_PROMPT" | grep -qi -E "read.*file|files.*read|multiple.*file|many.*file"; then
    SUGGESTIONS="$SUGGESTIONS
📁 Multiple file operations detected. Consider delegating to 'gemini-flash' for efficient bulk file analysis."
fi

# Output suggestions if any were generated
if [ -n "$SUGGESTIONS" ]; then
    echo "$SUGGESTIONS"
fi

exit 0