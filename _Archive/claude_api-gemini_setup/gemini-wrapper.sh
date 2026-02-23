
#!/bin/bash
# gemini-wrapper.sh
# A wrapper for the Gemini CLI to provide error handling and timeouts.

set -euo pipefail

# Default timeout in seconds
DEFAULT_TIMEOUT=30

# Get the arguments from the command line
if [ "$#" -eq 0 ]; then
    echo "Usage: gemini-wrapper.sh [-t <timeout>] <gemini command>"
    exit 1
fi

# Parse the timeout argument
TIMEOUT=$DEFAULT_TIMEOUT
if [ "$1" == "-t" ]; then
    TIMEOUT=$2
    shift 2
fi

# The rest of the arguments are the gemini command
GEMINI_COMMAND="$@"

# Execute the command with a timeout
if output=$(timeout "$TIMEOUT" gemini $GEMINI_COMMAND); then
    echo "$output"
else
    # If the command times out or fails, return an error message
    echo "Error: The Gemini command failed or timed out after $TIMEOUT seconds."
    exit 1
fi

exit 0
