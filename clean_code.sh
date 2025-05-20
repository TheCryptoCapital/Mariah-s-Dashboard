#!/bin/bash

TARGET=${1:-.}

echo "🔍 [1/4] Running Ruff fix on $TARGET..."
ruff check "$TARGET" --fix

echo "🧹 [2/4] Sorting imports with isort..."
isort "$TARGET"

echo "🎨 [3/4] Formatting with Black..."
black "$TARGET"

echo "📎 [4/4] Scanning for duplicate code with jscpd..."
jscpd "$TARGET" --min-tokens 30 --reporters console --pattern "**/*.py"

echo "✅ Code cleanup and duplication scan complete!"

