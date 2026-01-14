#!/bin/bash
# Setup for Lambda Labs / ephemeral GPU environments
set -e
source .env
pip install -r requirements.txt

git config --global user.email "$GIT_USER_EMAIL" && echo "Git email set to $GIT_USER_EMAIL"
git config --global user.name "$GIT_USER_NAME" && echo "Git name set to $GIT_USER_NAME"