#!/bin/zsh
set -e
cd "$(dirname "$0")"

if [[ ! -f .env ]]; then
  echo "Missing .env file. First run: cp .env.example .env"
  echo "Then add your MySQL username, password, and database name to .env."
  read -k 1 "REPLY?Press any key to close..."
  echo
  exit 1
fi

set -a
source .env
set +a
export DT_DATA_SOURCE="${DT_DATA_SOURCE:-mysql}"
if [[ -z "${DT_MYSQL_USER:-}" ]]; then
  read "DT_MYSQL_USER?MySQL username: "
  export DT_MYSQL_USER
fi
if [[ -z "${DT_MYSQL_PASSWORD:-}" ]]; then
  read -s "DT_MYSQL_PASSWORD?MySQL password: "
  echo
  export DT_MYSQL_PASSWORD
fi

echo "Starting the Flask HTML app at http://127.0.0.1:5000"
echo "Leave this window open; press Control+C to stop the app."
python3 server.py
