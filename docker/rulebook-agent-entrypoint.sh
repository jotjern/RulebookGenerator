#!/bin/sh
set -eu

VENV_DIR="${VENV_DIR:-/opt/scenicrules-venv}"
export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/bin:${PATH}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export SCENIC_RULES_ROOT="${SCENIC_RULES_ROOT:-/workspace}"

exec "$@"
