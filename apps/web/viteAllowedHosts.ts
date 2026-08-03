export function parseAllowedHosts(value: string | undefined): string[] {
  if (value === undefined) {
    return [];
  }

  return value
    .split(",")
    .map((host) => host.trim())
    .filter((host) => host.length > 0);
}
