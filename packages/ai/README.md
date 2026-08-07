# @napoleon/ai

## Evaluation runner

`runEvaluation` runs reproducible AI-vs-AI games on top of the existing
`runAutomatedGame` simulation.

`runAutomatedGame` passes each agent a `PlayerObservation` containing the player
view, legal actions, and prior public bidding action history. Existing agents
can ignore the history; policy agents that need bidding history can use it
without accessing hidden card state.

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

## Evaluation reports

`createEvaluationReport` aggregates the reusable evaluation runner schema. It
does not run games; it only summarizes an existing `EvaluationRunRecord`.

```ts
import { createEvaluationReport, runEvaluation } from "@napoleon/ai";

const run = await runEvaluation(options);
const report = createEvaluationReport(run);
```

The report schema is stable JSON data with:

- overall completed and failed game counts
- agent summaries keyed by `sourceAgentIndex` with wins, losses, contract
  success rate, average team point cards, role results, seat results, sample
  counts, and comparison deltas against the participating-agent average
- Wilson 95% confidence intervals for win rate and contract success rate
- 95% confidence intervals for win-rate, contract-success-rate, and average
  point-card comparison deltas
- seat summaries for seat-index bias checks, including sample counts
- Napoleon, adjutant, and alliance role summaries, including sample counts
- failed-game counts by reason

Rates with no completed games and averages with no samples are represented as
`null`, not `NaN` or `Infinity`. Report output is sorted by agent name, seat
index, role order, and failure reason so the same input games produce the same
report even when the input array order changes.

Rate intervals use the Wilson score interval at 95% confidence. Proportion
comparison intervals use the Newcombe-Wilson method, and average point-card
comparison intervals use a normal approximation over the observed point-card
samples. Empty comparisons keep the interval bounds as `null`.
