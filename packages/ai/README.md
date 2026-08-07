# @napoleon/ai

## Evaluation runner

`runEvaluation` runs reproducible AI-vs-AI games on top of the existing
`runAutomatedGame` simulation.

```ts
import { RuleBasedAgent, runEvaluation } from "@napoleon/ai";

const result = await runEvaluation({
  startSeed: 1000,
  gameCount: 10,
  rotationOffsets: [0, 1, 2, 3, 4],
  agents: Array.from({ length: 5 }, (_, index) => ({
    name: `RuleBasedAgent-${index}`,
    createAgent: ({ rng }) => new RuleBasedAgent(rng)
  }))
});
```

Each seed from `startSeed` to `startSeed + gameCount - 1` is run once for
each configured rotation offset. A rotation offset moves the configured agent
definitions around the fixed player seats while preserving deterministic game
seeds. Every scheduled game produces exactly one record:

- `status: "completed"` with seat agent names, Napoleon/adjutant/alliance
  roles, contract card count, point-card totals, winner, and contract success.
- `status: "failed"` with the same seat assignment fields and a
  `failureReason`.

The runner records failures instead of dropping failed games, so callers can
compare result counts against the expected `gameCount * rotationOffsets.length`.
