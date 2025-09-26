#!/bin/bash
COMMIT_MSG_FILE=$1
BRANCH_NAME=$(git symbolic-ref --short HEAD)

if [[ $BRANCH_NAME =~ ^issue/([0-9]+) ]]; then
    ISSUE_NO="${BASH_REMATCH[1]}"
    FIRST_LINE=$(head -n 1 "$COMMIT_MSG_FILE")

    pattern="#${ISSUE_NO}"
    if [[ ! $FIRST_LINE =~ $pattern ]]; then
        TEMP_FILE=$(mktemp)
        echo "$FIRST_LINE #$ISSUE_NO" > "$TEMP_FILE"

        if [ "$(wc -l < "$COMMIT_MSG_FILE")" -gt 1 ]; then
            tail -n +2 "$COMMIT_MSG_FILE" >> "$TEMP_FILE"
        fi

        cat "$TEMP_FILE" > "$COMMIT_MSG_FILE"
        rm "$TEMP_FILE"
    fi
fi
