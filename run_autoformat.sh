#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
    PYTHON="$(command -v python3 || command -v python)"
fi

cd "$REPO_ROOT"

"$PYTHON" -m black .
"$PYTHON" -m docformatter -i -r . --exclude venv .venv
"$PYTHON" -m isort --gitignore .
