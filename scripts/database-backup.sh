#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
umask 077
compose=(docker compose --env-file .env.production -f compose.production.yaml)
output="${1:-backups/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$output"
if [[ -e "$output/players.dump" || -e "$output/keycloak.dump" ]]; then
  echo "Backup files already exist in the destination." >&2; exit 1
fi
"${compose[@]}" stop server keycloak
trap '"${compose[@]}" start keycloak server' EXIT
"${compose[@]}" exec -T players-db pg_dump -U napoleon -d napoleon -Fc > "$output/players.dump.partial"
"${compose[@]}" exec -T keycloak-db pg_dump -U keycloak -d keycloak -Fc > "$output/keycloak.dump.partial"
mv "$output/players.dump.partial" "$output/players.dump"
mv "$output/keycloak.dump.partial" "$output/keycloak.dump"
echo "Backup saved: $output"
