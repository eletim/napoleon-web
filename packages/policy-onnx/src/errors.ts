export class PolicyOnnxCompatibilityError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "PolicyOnnxCompatibilityError";
  }
}
