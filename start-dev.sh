#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT_DIR="$SCRIPT_DIR"
ROOT_DIR="${NAPOLEON_DEV_ROOT:-$DEFAULT_ROOT_DIR}"
ROOT_ENV_FILE="$ROOT_DIR/.env"
ROOT_ENV_SAMPLE_FILE="$ROOT_DIR/.env.sample"
WEB_ENV_FILE="$ROOT_DIR/apps/web/.env.local"

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

normalize_dev_base_path() {
  local value

  value="$(trim "$1")"
  if [[ -z "$value" || "$value" == "/" ]]; then
    printf '/\n'
    return
  fi

  if [[ "$value" != /* ]]; then
    value="/$value"
  fi
  if [[ "$value" != */ ]]; then
    value="$value/"
  fi

  printf '%s\n' "$value"
}

DEV_BASE_PATH="$(normalize_dev_base_path "${NAPOLEON_DEV_BASE_PATH:-/napoleon/}")"
DEV_BASE_PREFIX="${DEV_BASE_PATH%/}"
if [[ -z "$DEV_BASE_PREFIX" ]]; then
  TAILSCALE_SERVE_PATH="/"
  TAILSCALE_SERVE_TARGET="http://127.0.0.1:5173"
else
  TAILSCALE_SERVE_PATH="$DEV_BASE_PREFIX"
  TAILSCALE_SERVE_TARGET="http://127.0.0.1:5173$DEV_BASE_PREFIX"
fi
TAILSCALE_SERVE_COMMAND=(tailscale serve --bg --set-path="$TAILSCALE_SERVE_PATH" "$TAILSCALE_SERVE_TARGET")

read_allowed_hosts() {
  local file="$1"
  local line

  line="$(grep -E '^VITE_ALLOWED_HOSTS=' "$file" | tail -n 1 || true)"
  if [[ -n "$line" ]]; then
    printf '%s' "${line#VITE_ALLOWED_HOSTS=}"
  fi
}

create_root_env_file_from_sample() {
  if [[ -f "$ROOT_ENV_FILE" || ! -f "$ROOT_ENV_SAMPLE_FILE" ]]; then
    return
  fi

  cp "$ROOT_ENV_SAMPLE_FILE" "$ROOT_ENV_FILE"
  printf '.env を .env.sample から生成しました。\n'
}

load_root_env_file() {
  local file="$1"
  local line
  local key
  local value

  if [[ ! -f "$file" ]]; then
    return
  fi

  while IFS= read -r line || [[ -n "$line" ]]; do
    line="$(trim "$line")"
    if [[ -z "$line" || "$line" == \#* ]]; then
      continue
    fi
    if [[ "$line" == export\ * ]]; then
      line="${line#export }"
      line="$(trim "$line")"
    fi
    if [[ "$line" != *=* ]]; then
      continue
    fi

    key="$(trim "${line%%=*}")"
    value="$(trim "${line#*=}")"
    if [[ "$value" == \"*\" && "$value" == *\" ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
      value="${value:1:${#value}-2}"
    fi

    # Keep this pattern aligned with apps/server/src/agentEnv.ts learnedPolicyEnvKeys.
    case "$key" in
      PORT|HOST|NAPOLEON_POLICY_[1-5]_DISPLAY_NAME|NAPOLEON_POLICY_[1-5]_ONNX_PATH|NAPOLEON_POLICY_[1-5]_METADATA_PATH)
        if [[ -z "${!key+x}" ]]; then
          export "$key=$value"
        fi
        ;;
    esac
  done < "$file"
}

count_ready_policy_slots() {
  local count=0
  local slot
  local display_name_key
  local onnx_path_key
  local metadata_path_key

  for slot in 1 2 3 4 5; do
    display_name_key="NAPOLEON_POLICY_${slot}_DISPLAY_NAME"
    onnx_path_key="NAPOLEON_POLICY_${slot}_ONNX_PATH"
    metadata_path_key="NAPOLEON_POLICY_${slot}_METADATA_PATH"
    if [[ -n "${!display_name_key:-}" && -n "${!onnx_path_key:-}" && -n "${!metadata_path_key:-}" ]]; then
      count=$((count + 1))
    fi
  done

  printf '%s' "$count"
}

count_incomplete_policy_slots() {
  local count=0
  local slot
  local display_name_key
  local onnx_path_key
  local metadata_path_key

  for slot in 1 2 3 4 5; do
    display_name_key="NAPOLEON_POLICY_${slot}_DISPLAY_NAME"
    onnx_path_key="NAPOLEON_POLICY_${slot}_ONNX_PATH"
    metadata_path_key="NAPOLEON_POLICY_${slot}_METADATA_PATH"
    if [[ -n "${!display_name_key:-}" && ( -z "${!onnx_path_key:-}" || -z "${!metadata_path_key:-}" ) ]]; then
      count=$((count + 1))
    fi
  done

  printf '%s' "$count"
}

read_tailscale_self_dns_name() {
  local status_json

  if ! command -v tailscale >/dev/null 2>&1; then
    return 1
  fi

  if ! command -v node >/dev/null 2>&1; then
    return 1
  fi

  if ! status_json="$(tailscale status --json 2>/dev/null)"; then
    return 1
  fi

  printf '%s' "$status_json" | node -e '
let input = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => {
  input += chunk;
});
process.stdin.on("end", () => {
  try {
    const status = JSON.parse(input);
    if (status?.BackendState !== "Running") {
      process.exit(1);
    }
    const dnsName = status?.Self?.DNSName;
    if (typeof dnsName !== "string") {
      process.exit(1);
    }
    const normalized = dnsName.trim().replace(/\.+$/, "");
    if (normalized.length === 0) {
      process.exit(1);
    }
    process.stdout.write(normalized);
  } catch {
    process.exit(1);
  }
});
'
}

create_env_file_from_tailscale() {
  local allowed_host

  if [[ -f "$WEB_ENV_FILE" ]]; then
    return
  fi

  if ! allowed_host="$(read_tailscale_self_dns_name)"; then
    printf 'Tailscale DNS名を取得できないため、.env.local を生成せずlocalhost限定で起動します。\n'
    return
  fi

  if [[ -f "$WEB_ENV_FILE" ]]; then
    return
  fi

  mkdir -p "$(dirname "$WEB_ENV_FILE")"
  printf 'VITE_ALLOWED_HOSTS=%s\n' "$allowed_host" > "$WEB_ENV_FILE"
  printf 'apps/web/.env.local をTailscale DNS名から生成しました。\n'
}

create_root_env_file_from_sample
load_root_env_file "$ROOT_ENV_FILE"

unset VITE_ALLOWED_HOSTS
unset NAPOLEON_WEB_BASE_PATH

if [[ ! -f "$WEB_ENV_FILE" ]]; then
  create_env_file_from_tailscale
fi

if [[ -f "$WEB_ENV_FILE" ]]; then
  VITE_ALLOWED_HOSTS="$(read_allowed_hosts "$WEB_ENV_FILE")"
  if [[ -n "$VITE_ALLOWED_HOSTS" ]]; then
    export VITE_ALLOWED_HOSTS
    NAPOLEON_WEB_BASE_PATH="$DEV_BASE_PATH"
    export NAPOLEON_WEB_BASE_PATH
  else
    unset VITE_ALLOWED_HOSTS
    unset NAPOLEON_WEB_BASE_PATH
  fi
fi

if [[ "${NAPOLEON_DEV_DRY_RUN:-}" == "1" ]]; then
  if [[ -f "$ROOT_ENV_FILE" ]]; then
    printf 'root_env_file_exists=true\n'
  else
    printf 'root_env_file_exists=false\n'
  fi
  printf 'learned_policy_slots_configured=%s\n' "$(count_ready_policy_slots)"
  printf 'learned_policy_slots_incomplete=%s\n' "$(count_incomplete_policy_slots)"
  if [[ -f "$WEB_ENV_FILE" ]]; then
    printf 'env_file_exists=true\n'
  else
    printf 'env_file_exists=false\n'
  fi
  printf 'VITE_ALLOWED_HOSTS=%s\n' "${VITE_ALLOWED_HOSTS:-}"
  printf 'NAPOLEON_WEB_BASE_PATH=%s\n' "${NAPOLEON_WEB_BASE_PATH:-}"
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
