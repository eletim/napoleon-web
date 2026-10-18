#!/usr/bin/env bash
# Restores both databases, overwriting the current database contents.
set -euo pipefail
cd "$(dirname "$0")/.."
source_dir="${1:?Specify a backup directory containing players.dump and keycloak.dump}"
[[ -s "$source_dir/players.dump" && -s "$source_dir/keycloak.dump" ]] || { echo "Incomplete backup." >&2; exit 1; }
compose=(docker compose --env-file .env.production -f compose.production.yaml)
"${compose[@]}" stop nginx server keycloak
"${compose[@]}" exec -T players-db pg_restore -l < "$source_dir/players.dump" > /dev/null
"${compose[@]}" exec -T keycloak-db pg_restore -l < "$source_dir/keycloak.dump" > /dev/null
"${compose[@]}" exec -T players-db pg_restore --exit-on-error --clean --if-exists --no-owner --single-transaction -U napoleon -d napoleon < "$source_dir/players.dump"
"${compose[@]}" exec -T keycloak-db pg_restore --exit-on-error --clean --if-exists --no-owner --single-transaction -U keycloak -d keycloak < "$source_dir/keycloak.dump"
"${compose[@]}" up -d keycloak server nginx
