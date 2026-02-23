#!/bin/bash
# context-monitor.sh
# Enhanced context monitoring hook with intelligent delegation suggestions

set -euo pipefail

# Read the conversation from stdin
CONVERSATION=$(cat)

# Configuration files
THRESHOLDS_FILE="$HOME/.claude/config/thresholds.conf"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load thresholds from config file
if [ -f "$THRESHOLDS_FILE" ]; then
    source "$THRESHOLDS_FILE"
else
    # Default thresholds if config doesn't exist
    LINE_THRESHOLD=100
    WORD_THRESHOLD=5000
    COMPLEXITY_THRESHOLD=50
fi

# Get conversation metrics
NUM_LINES=$(echo "$CONVERSATION" | wc -l)
NUM_WORDS=$(echo "$CONVERSATION" | wc -w)

# Function to assess conversation complexity
assess_complexity() {
    local conversation="$1"
    local complexity_score=0
    
    # Count technical terms (rough complexity indicator)
    local technical_terms=$(echo "$conversation" | grep -oi -E "(function|class|method|api|database|server|client|component|module|library|framework|algorithm|architecture|security|performance)" | wc -l)
    
    # Count file references
    local file_refs=$(echo "$conversation" | grep -oi -E "\.[a-z]{1,4}[^a-z]" | wc -l)
    
    # Count code blocks (rough indicator)
    local code_blocks=$(echo "$conversation" | grep -c '```' || true)
    
    complexity_score=$((technical_terms + file_refs + code_blocks))
    echo $complexity_score
}

# Function to suggest appropriate delegation based on conversation content
suggest_delegation() {
    local conversation="$1"
    
    # Check for bulk analysis patterns
    if echo "$conversation" | grep -qi -E "(search.*file|find.*pattern|analyze.*code|scan.*directory)"; then
        echo "🔍 **Delegation Suggestion**: This conversation involves file searching and code analysis. Consider using the 'gemini-flash' sub-agent for comprehensive codebase analysis."
        return
    fi
    
    # Check for architectural discussion patterns
    if echo "$conversation" | grep -qi -E "(architect|design|strategy|refactor|migration|security.*review)"; then
        echo "🧠 **Delegation Suggestion**: This conversation involves architectural or strategic topics. Consider using the 'gemini-pro' sub-agent for deep reasoning and strategic analysis."
        return
    fi
    
    # Check for multi-faceted analysis patterns
    if echo "$conversation" | grep -qi -E "(both.*and|multiple.*aspect|various.*component|different.*part)"; then
        echo "💡 **Delegation Suggestion**: This conversation involves multiple aspects or components. Consider using the 'parallel-coordinator' to break tasks into simultaneous sub-analyses."
        return
    fi
    
    # Generic delegation suggestion for long conversations
    echo "⚡ **Delegation Suggestion**: This conversation is getting extensive. Consider delegating complex analysis tasks to specialized sub-agents (gemini-flash for searches, gemini-pro for strategic thinking, parallel-coordinator for multi-faceted tasks)."
}

# Get complexity assessment
COMPLEXITY_SCORE=$(assess_complexity "$CONVERSATION")

# Generate output based on thresholds
OUTPUT=""

# Check line threshold
if [ "$NUM_LINES" -gt "$LINE_THRESHOLD" ]; then
    OUTPUT="$OUTPUT
📏 **Context Alert**: Conversation length ($NUM_LINES lines) exceeds threshold ($LINE_THRESHOLD lines)."
fi

# Check word threshold  
if [ "$NUM_WORDS" -gt "$WORD_THRESHOLD" ]; then
    OUTPUT="$OUTPUT
📝 **Context Alert**: Conversation size ($NUM_WORDS words) exceeds threshold ($WORD_THRESHOLD words)."
fi

# Check complexity threshold
if [ "$COMPLEXITY_SCORE" -gt "$COMPLEXITY_THRESHOLD" ]; then
    OUTPUT="$OUTPUT
🧩 **Complexity Alert**: Conversation complexity ($COMPLEXITY_SCORE) exceeds threshold ($COMPLEXITY_THRESHOLD)."
fi

# Add delegation suggestions if any thresholds exceeded
if [ -n "$OUTPUT" ]; then
    DELEGATION_SUGGESTION=$(suggest_delegation "$CONVERSATION")
    OUTPUT="$OUTPUT

$DELEGATION_SUGGESTION

💡 **Tip**: Sub-agents can handle specialized tasks more efficiently while keeping the main conversation focused."
fi

# Output alerts and suggestions if any were generated
if [ -n "$OUTPUT" ]; then
    echo "$OUTPUT"
fi

exit 0