
#!/bin/bash
# context-monitor.sh
# Hook script to monitor conversation length and suggest delegation.

set -euo pipefail

# Read the conversation from stdin
CONVERSATION=$(cat)

# Read the threshold from the config file
THRESHOLDS_FILE="/Users/marazzanocolon/coding/morphseq/claude_api-gemini_setup/thresholds.conf"
if [ -f "$THRESHOLDS_FILE" ]; then
    LINE_THRESHOLD=$(grep "line_threshold" "$THRESHOLDS_FILE" | cut -d'=' -f2)
else
    # Default threshold if the file doesn't exist
    LINE_THRESHOLD=100
fi

# Get the number of lines in the conversation
NUM_LINES=$(echo "$CONVERSATION" | wc -l)

# Check if the conversation length exceeds the threshold
if [ "$NUM_LINES" -gt "$LINE_THRESHOLD" ]; then
    # If the threshold is exceeded, inject a suggestion into Claude's context
    echo "This conversation is getting long. To keep things manageable, consider delegating complex tasks to a sub-agent like 'gemini-flash' or 'gemini-pro'."
fi

exit 0
