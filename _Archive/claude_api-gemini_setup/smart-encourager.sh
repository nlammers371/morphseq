
#!/bin/bash
# smart-encourager.sh
# Hook script to inject suggestions for Claude to delegate tasks.

set -euo pipefail

# Read the user's prompt from stdin
USER_PROMPT=$(cat)

# Define a list of encouraging messages
MESSAGES=(
    "Is this a task that could be handled more efficiently by 'gemini-flash'?"
    "Remember, for deep architectural questions, 'gemini-pro' is a great resource."
    "Could this be broken down into parallel tasks for the 'parallel-coordinator'?"
    "Don't forget you have powerful sub-agents at your disposal!"
)

# Randomly select a message
RANDOM_INDEX=$((RANDOM % ${#MESSAGES[@]}))
SELECTED_MESSAGE=${MESSAGES[$RANDOM_INDEX]}

# Inject the message into Claude's context
# We can add more sophisticated logic here to decide when to show a message.
# For now, we'll show it periodically.
SHOW_MESSAGE=$((RANDOM % 3))

if [ "$SHOW_MESSAGE" -eq 0 ]; then
    echo "$SELECTED_MESSAGE"
fi

exit 0
