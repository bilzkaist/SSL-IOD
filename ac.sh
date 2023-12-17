#!/bin/bash

# Check if arguments are provided
if [ -z "$1" ]; then
  commit_msg="Auto commit at $(date +'%Y-%m-%d %H:%M:%S')"
else
  commit_msg="$1"
fi

if [ -z "$2" ]; then
  branch="main"  # Change this to 'main' if needed
else
  branch="$2"
fi

# Add all changes, commit with the provided/default message, and push to the specified/default branch
git add .
git commit -m "$commit_msg"
git push origin "$branch"