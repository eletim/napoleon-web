#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT_DIR="$SCRIPT_DIR"
ROOT_DIR="${NAPOLEON_DEV_ROOT:-$DEFAULT_ROOT_DIR}"
ENV_FILE="$ROOT_DIR/apps/web/.env.local"
TAILSCALE_SERVE_COMMAND=(tailscale serve --bg --http=5173 http://127.0.0.1:5173)

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

normalize_hosts() {
  local input="$1"
  local normalized=""
  local host

  IFS=',' read -ra hosts <<< "$input"
  for host in "${hosts[@]}"; do
    host="$(trim "$host")"
    if [[ -n "$host" ]]; then
      if [[ -n "$normalized" ]]; then
        normalized+=","
      fi
      normalized+="$host"
    fi
  done

  printf '%s' "$normalized"
}

read_allowed_hosts() {
  local file="$1"
  local line

  line="$(grep -E '^VITE_ALLOWED_HOSTS=' "$file" | tail -n 1 || true)"
  if [[ -n "$line" ]]; then
    printf '%s' "${line#VITE_ALLOWED_HOSTS=}"
  fi
}

is_interactive() {
  [[ -t 0 || "${NAPOLEON_DEV_FORCE_INTERACTIVE:-}" == "1" ]]
}

create_env_file_interactively() {
  local answer
  local raw_hosts
  local allowed_hosts

  printf 'apps/web/.env.local がありません。\n'
  printf '外部アクセス用の設定を生成しますか？ [y/N]: '
  read -r answer

  if [[ "$answer" != "y" && "$answer" != "Y" ]]; then
    printf '.env.local を生成せず、localhost限定で起動します。\n'
    return
  fi

  printf '許可するホスト名を入力してください。\n'
  printf '複数指定する場合はカンマ区切りです: '
  read -r raw_hosts
  allowed_hosts="$(normalize_hosts "$raw_hosts")"

  if [[ -z "$allowed_hosts" ]]; then
    printf '.env.local を生成せず、localhost限定で起動します。\n'
    return
  fi

  if [[ -f "$ENV_FILE" ]]; then
    return
  fi

  mkdir -p "$(dirname "$ENV_FILE")"
  printf 'VITE_ALLOWED_HOSTS=%s\n' "$allowed_hosts" > "$ENV_FILE"
  printf 'apps/web/.env.local を生成しました。\n'
}

unset VITE_ALLOWED_HOSTS

if [[ ! -f "$ENV_FILE" ]] && is_interactive; then
  create_env_file_interactively
fi

if [[ -f "$ENV_FILE" ]]; then
  VITE_ALLOWED_HOSTS="$(read_allowed_hosts "$ENV_FILE")"
  if [[ -n "$VITE_ALLOWED_HOSTS" ]]; then
    export VITE_ALLOWED_HOSTS
  else
    unset VITE_ALLOWED_HOSTS
  fi
fi

if [[ "${NAPOLEON_DEV_DRY_RUN:-}" == "1" ]]; then
  if [[ -f "$ENV_FILE" ]]; then
    printf 'env_file_exists=true\n'
  else
    printf 'env_file_exists=false\n'
  fi
  printf 'VITE_ALLOWED_HOSTS=%s\n' "${VITE_ALLOWED_HOSTS:-}"
  if [[ -n "${VITE_ALLOWED_HOSTS:-}" ]]; then
    printf 'tailscale_serve_enabled=true\n'
    printf 'tailscale_serve_command=%s\n' "${TAILSCALE_SERVE_COMMAND[*]}"
  else
    printf 'tailscale_serve_enabled=false\n'
  fi
  printf 'dev_server_started=false\n'
  exit 0
fi

cd "$ROOT_DIR"

if [[ -n "${VITE_ALLOWED_HOSTS:-}" ]]; then
  if ! command -v tailscale >/dev/null 2>&1; then
    printf 'エラー: tailscaleコマンドが見つかりません。\n' >&2
    exit 1
  fi

  if ! tailscale status >/dev/null 2>&1; then
    printf 'エラー: Tailscaleが接続されていません。\n' >&2
    exit 1
  fi

  if ! "${TAILSCALE_SERVE_COMMAND[@]}"; then
    printf 'エラー: Tailscale Serveの設定に失敗しました。\n' >&2
    exit 1
  fi
fi

if [[ "${NAPOLEON_DEV_TEST_MODE:-}" == "1" ]]; then
  printf 'dev_server_started=true\n'
  exit 0
fi

exec pnpm dev:raw
