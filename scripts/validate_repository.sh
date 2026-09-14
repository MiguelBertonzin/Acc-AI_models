#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

status=0

echo "[1/3] Arquivos acima do limite de 100 MiB"
large_files="$(find . -path './.git' -prune -o -type f -size +100M -print)"
if [[ -n "$large_files" ]]; then
  printf '%s\n' "$large_files"
  status=1
else
  echo "OK"
fi

echo "[2/3] Caches e temporários que não deveriam ser versionados"
ignored_tracked="$(git ls-files -ci --exclude-standard)"
if [[ -n "$ignored_tracked" ]]; then
  printf '%s\n' "$ignored_tracked"
  status=1
else
  echo "OK"
fi

echo "[3/3] Padrões comuns de credenciais"
credential_pattern='github_pat_[A-Za-z0-9_]+|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{20,}|-----BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY-----'
if git grep -n -I -E "$credential_pattern" -- . ':!scripts/validate_repository.sh'; then
  status=1
else
  echo "OK"
fi

exit "$status"

