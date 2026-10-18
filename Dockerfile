FROM node:22-bookworm-slim AS build
WORKDIR /app
RUN corepack enable && corepack prepare pnpm@9.15.0 --activate
COPY . .
RUN pnpm install --frozen-lockfile
RUN pnpm --filter @napoleon/server... build
ENV NAPOLEON_WEB_BASE_PATH=/napoleon/
RUN pnpm --filter @napoleon/web build

FROM node:22-bookworm-slim AS server
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build --chown=node:node /app /app
USER node
ENV NODE_ENV=production HOST=0.0.0.0 PORT=3000
EXPOSE 3000
CMD ["node", "apps/server/dist/index.js"]

FROM nginx:stable-bookworm AS web
COPY --from=build /app/apps/web/dist /usr/share/nginx/html/napoleon
COPY infra/nginx/default.conf.template /etc/nginx/templates/default.conf.template
