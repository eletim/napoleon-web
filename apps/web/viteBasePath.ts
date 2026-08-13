export function normalizeViteBasePath(value: string | undefined): string {
  const trimmed = value?.trim();

  if (trimmed === undefined || trimmed.length === 0 || trimmed === "/") {
    return "/";
  }

  const leadingSlash = trimmed.startsWith("/") ? trimmed : `/${trimmed}`;
  return leadingSlash.endsWith("/") ? leadingSlash : `${leadingSlash}/`;
}
