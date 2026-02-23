
#!/bin/bash
# keyword-detector.sh
# Hook script to detect keywords in user prompts and suggest sub-agent delegation.

set -euo pipefail

# Read the user's prompt from stdin
USER_PROMPT=$(cat)

# Read keywords from the config file
KEYWORDS_FILE="/Users/marazzanocolon/coding/morphseq/claude_api-gemini_setup/keywords.conf"
if [ -f "$KEYWORDS_FILE" ]; then
    KEYWORDS=$(paste -sd'|' "$KEYWORDS_FILE")
else
    # Default keywords if the file doesn't exist
    KEYWORDS="search|analyze|find|scan|look for|what are|how does"
fi

# Check if the prompt contains any of the keywords
if echo "$USER_PROMPT" | grep -qE "($KEYWORDS)"; then
    # If a keyword is found, inject a suggestion into Claude's context
    echo "Based on your prompt, this might be a good opportunity to use a sub-agent like 'gemini-flash' for a comprehensive search. For example: 'Task: Use gemini-flash to [your request].'"
fi

exit 0
