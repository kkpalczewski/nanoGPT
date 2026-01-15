#!/bin/bash
# Setup for Lambda Labs / ephemeral GPU environments
set -e
[ -f .env ] && source .env
pip install -r requirements.txt

[ -n "$GIT_USER_EMAIL" ] && git config --global user.email "$GIT_USER_EMAIL" && echo "Git email set to $GIT_USER_EMAIL"
[ -n "$GIT_USER_NAME" ] && git config --global user.name "$GIT_USER_NAME" && echo "Git name set to $GIT_USER_NAME"

wandb login --relogin ${WANDB_API_KEY:-}

if [ "$1" == "--data" ]; then
    echo "Preparing datasets..."
    python data/shakespeare/prepare.py
    python data/shakespeare_char/prepare.py
    python data/openwebtext/prepare.py
fi

# Print node info for multi-node setup
echo ""
echo "=== Node info (for multi-node training) ==="
echo "Hostname: $(hostname)"
echo "Private IP: $(hostname -I | awk '{print $1}')"
echo "/etc/hosts entry: $(hostname -I | awk '{print $1}')   $(hostname)"
