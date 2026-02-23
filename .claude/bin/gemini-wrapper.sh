#!/bin/bash
# gemini-wrapper.sh
# Enhanced wrapper for Gemini CLI with comprehensive error handling and timeout management

set -euo pipefail

# Configuration
CONFIG_FILE="$HOME/.claude/config/thresholds.conf"
LOG_FILE="$HOME/.claude/logs/gemini-wrapper.log"

# Load configuration
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
else
    # Default settings
    GEMINI_FLASH_TIMEOUT=30
    GEMINI_PRO_TIMEOUT=60
fi

# Logging function
log_message() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Create log directory if it doesn't exist
    mkdir -p "$(dirname "$LOG_FILE")"
    
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
    
    # Also output to stderr for debugging
    echo "[$level] $message" >&2
}

# Usage function
show_usage() {
    cat << EOF
Usage: gemini-wrapper.sh [OPTIONS] <gemini-command>

OPTIONS:
    -m, --model MODEL     Specify Gemini model (gemini-2.5-flash-latest, gemini-2.5-pro)
    -t, --timeout SECONDS Override default timeout
    -r, --retry COUNT     Number of retry attempts (default: 2)
    -v, --verbose         Enable verbose logging
    -h, --help           Show this help message

EXAMPLES:
    gemini-wrapper.sh -m gemini-2.5-flash-latest --all-files -p "Find all API endpoints"
    gemini-wrapper.sh -m gemini-2.5-pro -t 90 --all-files -p "Analyze architecture"

MODELS AND TIMEOUTS:
    gemini-2.5-flash-latest: ${GEMINI_FLASH_TIMEOUT}s timeout (fast, bulk analysis)
    gemini-2.5-pro:          ${GEMINI_PRO_TIMEOUT}s timeout (deep reasoning)
EOF
}

# Parse command line arguments
TIMEOUT=""
RETRY_COUNT=2
VERBOSE=false
MODEL=""
GEMINI_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL="$2"
            GEMINI_ARGS+=("-m" "$2")
            shift 2
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -r|--retry)
            RETRY_COUNT="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        --)
            shift
            GEMINI_ARGS+=("$@")
            break
            ;;
        *)
            GEMINI_ARGS+=("$1")
            shift
            ;;
    esac
done

# Validate arguments
if [ ${#GEMINI_ARGS[@]} -eq 0 ]; then
    echo "Error: No Gemini command specified" >&2
    show_usage
    exit 1
fi

# Determine timeout based on model if not explicitly set
if [ -z "$TIMEOUT" ]; then
    case "$MODEL" in
        *"flash"*)
            TIMEOUT=$GEMINI_FLASH_TIMEOUT
            ;;
        *"pro"*)
            TIMEOUT=$GEMINI_PRO_TIMEOUT
            ;;
        *)
            TIMEOUT=$GEMINI_FLASH_TIMEOUT  # Default to flash timeout
            ;;
    esac
fi

# Verbose logging
if $VERBOSE; then
    log_message "INFO" "Starting Gemini wrapper with model: $MODEL, timeout: ${TIMEOUT}s, retries: $RETRY_COUNT"
    log_message "INFO" "Command: gemini ${GEMINI_ARGS[*]}"
fi

# Function to execute Gemini with error handling
execute_gemini() {
    local attempt="$1"
    
    log_message "INFO" "Attempt $attempt: Executing Gemini CLI"
    
    # Create a temporary file for error capture
    local error_file=$(mktemp)
    local output_file=$(mktemp)
    
    # Execute with timeout and capture both stdout and stderr
    if timeout "$TIMEOUT" gemini "${GEMINI_ARGS[@]}" > "$output_file" 2> "$error_file"; then
        # Success case
        local output=$(cat "$output_file")
        rm -f "$error_file" "$output_file"
        
        log_message "INFO" "Gemini CLI completed successfully"
        echo "$output"
        return 0
    else
        # Failure case
        local exit_code=$?
        local error_output=$(cat "$error_file" 2>/dev/null || echo "No error output captured")
        rm -f "$error_file" "$output_file"
        
        # Analyze the type of failure
        case $exit_code in
            124)
                log_message "ERROR" "Gemini CLI timed out after ${TIMEOUT}s"
                echo "Error: Gemini CLI timed out after ${TIMEOUT} seconds." >&2
                ;;
            1)
                log_message "ERROR" "Gemini CLI failed: $error_output"
                echo "Error: Gemini CLI failed: $error_output" >&2
                ;;
            127)
                log_message "ERROR" "Gemini CLI not found - check installation"
                echo "Error: Gemini CLI not found. Please ensure it's installed and in your PATH." >&2
                ;;
            *)
                log_message "ERROR" "Gemini CLI failed with exit code $exit_code: $error_output"
                echo "Error: Gemini CLI failed with exit code $exit_code: $error_output" >&2
                ;;
        esac
        
        return $exit_code
    fi
}

# Function to check Gemini CLI availability
check_gemini_cli() {
    if ! command -v gemini &> /dev/null; then
        log_message "ERROR" "Gemini CLI not found in PATH"
        echo "Error: Gemini CLI not found. Please install it first:" >&2
        echo "  npm install -g @google/generative-ai" >&2
        echo "  Or follow the installation guide at: https://ai.google.dev/tutorials/get_started_cli" >&2
        return 1
    fi
    
    # Check if authenticated
    if ! gemini auth list &> /dev/null; then
        log_message "ERROR" "Gemini CLI not authenticated"
        echo "Error: Gemini CLI not authenticated. Please run:" >&2
        echo "  gemini auth login" >&2
        return 1
    fi
    
    return 0
}

# Function to implement exponential backoff
calculate_backoff_delay() {
    local attempt="$1"
    local base_delay=2
    local max_delay=30
    
    local delay=$((base_delay ** (attempt - 1)))
    if [ $delay -gt $max_delay ]; then
        delay=$max_delay
    fi
    
    echo $delay
}

# Main execution logic
main() {
    # Pre-flight checks
    if ! check_gemini_cli; then
        exit 1
    fi
    
    # Execute with retry logic
    local attempt=1
    while [ $attempt -le $RETRY_COUNT ]; do
        if execute_gemini "$attempt"; then
            # Success
            log_message "INFO" "Gemini wrapper completed successfully on attempt $attempt"
            exit 0
        else
            local exit_code=$?
            
            # Don't retry on certain error types
            case $exit_code in
                127)  # Command not found
                    log_message "ERROR" "Permanent failure: Gemini CLI not found"
                    exit $exit_code
                    ;;
            esac
            
            if [ $attempt -lt $RETRY_COUNT ]; then
                local delay=$(calculate_backoff_delay $attempt)
                log_message "INFO" "Attempt $attempt failed, retrying in ${delay}s..."
                
                if $VERBOSE; then
                    echo "Attempt $attempt failed, retrying in ${delay}s..." >&2
                fi
                
                sleep $delay
            else
                log_message "ERROR" "All $RETRY_COUNT attempts failed"
                echo "Error: All $RETRY_COUNT attempts failed." >&2
                exit $exit_code
            fi
        fi
        
        ((attempt++))
    done
}

# Execute main function
main "$@"