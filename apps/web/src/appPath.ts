export function resolveAppPath(
  path: string,
  basePath: string = import.meta.env.BASE_URL
): string {
  const normalizedBasePath = normalizeBasePath(basePath);
  const relativePath = path.replace(/^\/+/, "");

  return `${normalizedBasePath}${relativePath}`;
}

function normalizeBasePath(basePath: string): string {
  const trimmed = basePath.trim();

  if (trimmed.length === 0 || trimmed === "/") {
    return "/";
  }

  const leadingSlash = trimmed.startsWith("/") ? trimmed : `/${trimmed}`;
  return leadingSlash.endsWith("/") ? leadingSlash : `${leadingSlash}/`;
}
