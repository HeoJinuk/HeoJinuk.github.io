#!/bin/zsh
set -eu
cd "$(dirname "$0")"
if command -v node >/dev/null 2>&1; then
  editor_node="$(command -v node)"
else
  editor_node="$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node"
fi
if [[ ! -x "$editor_node" ]]; then
  print 'Node.js 20 이상을 설치한 뒤 다시 실행해 주세요.'
  read -r '?종료하려면 Enter를 누르세요.'
  exit 1
fi
"$editor_node" scripts/editor.mjs || {
  read -r '?위 오류를 확인한 뒤 Enter를 누르세요.'
  exit 1
}
