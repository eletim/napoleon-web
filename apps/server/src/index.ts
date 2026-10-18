import { buildApp } from "./app.js";
import { loadLocalEnvFile } from "./env.js";

loadLocalEnvFile();

const port = Number(process.env.PORT ?? 3000);
const host = process.env.HOST ?? "127.0.0.1";

const app = await buildApp({ logger: true });

try {
  await app.listen({ port, host });
} catch (error) {
  app.log.error(error);
  process.exit(1);
}

let closing = false;
for (const signal of ["SIGTERM", "SIGINT"] as const) process.on(signal, () => {
  if (closing) return; closing = true;
  void app.close().catch((error) => { app.log.error(error); process.exitCode = 1; });
});
