#!/usr/bin/env bash
set -euo pipefail

# Load .env from current directory
if [[ -f ".env" ]]; then
  # export all variables defined in .env
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
else
  echo ".env not found in $(pwd)" >&2
  exit 1
fi

: "${LLM_BASE_URL:?LLM_BASE_URL not set}"
: "${LLM_API_KEY:?LLM_API_KEY not set}"
: "${LLM_MODEL:?LLM_MODEL not set}"

curl -v \
  -X POST "${LLM_BASE_URL}/chat/completions" \
  -H "Authorization: Bearer ${LLM_API_KEY}" \
  -H "Content-Type: application/json" \
  -d @<(cat <<EOF
{
  "model": "${LLM_MODEL}",
  "messages": [
    {
      "role": "user",
      "content": "Hello, world!"
    }
  ]
}
EOF
)