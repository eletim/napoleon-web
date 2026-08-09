export type ParsedValueMap = ReadonlyMap<string, string>;

export function parseOptionMap(
  argv: readonly string[],
  optionNames: ReadonlySet<string>
): ParsedValueMap {
  const normalizedArgv = stripLeadingSeparators(argv);
  const values = new Map<string, string>();

  for (let index = 0; index < normalizedArgv.length; index += 1) {
    const name = normalizedArgv[index];

    if (name === undefined) {
      throw new Error("Unexpected argument parser state.");
    }

    if (!name.startsWith("--")) {
      throw new Error(`Unknown argument: ${name}`);
    }

    if (!optionNames.has(name)) {
      throw new Error(`Unknown argument: ${name}`);
    }

    if (values.has(name)) {
      throw new Error(`Duplicate argument: ${name}`);
    }

    const value = normalizedArgv[index + 1];

    if (value === undefined || value.startsWith("--")) {
      throw new Error(`${name} requires a value.`);
    }

    values.set(name, value);
    index += 1;
  }

  return values;
}

export function requireValue(values: ParsedValueMap, name: string): string {
  const value = values.get(name);

  if (value === undefined) {
    throw new Error(`${name} is required.`);
  }

  return value;
}

export function optionalValue(values: ParsedValueMap, name: string): string | undefined {
  return values.get(name);
}

export function parseUnsignedInteger(name: string, value: string): number {
  if (!/^(0|[1-9][0-9]*)$/.test(value)) {
    throw new Error(`${name} must be an integer.`);
  }

  const parsed = Number(value);

  if (!Number.isSafeInteger(parsed)) {
    throw new Error(`${name} must be a safe integer.`);
  }

  return parsed;
}

export function parsePositiveInteger(name: string, value: string): number {
  const parsed = parseUnsignedInteger(name, value);

  if (parsed <= 0) {
    throw new Error(`${name} must be a positive integer.`);
  }

  return parsed;
}

export function parsePositiveNumber(name: string, value: string): number {
  const parsed = Number(value);

  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`${name} must be a finite positive number.`);
  }

  return parsed;
}

function stripLeadingSeparators(argv: readonly string[]): readonly string[] {
  let firstArgumentIndex = 0;

  while (argv[firstArgumentIndex] === "--") {
    firstArgumentIndex += 1;
  }

  return argv.slice(firstArgumentIndex);
}
