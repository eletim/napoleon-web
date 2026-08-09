import type { GameStatusDisplay } from "./displayText";

interface GameStatusProps {
  display: GameStatusDisplay;
}

export function GameStatus({ display }: GameStatusProps) {
  const chips = [...display.primary, ...display.secondary];

  if (chips.length === 0) {
    return null;
  }

  return (
    <section className="game-status" aria-label="ゲーム状態">
      <div className="compact-status-row">
        {chips.map((chip) => (
          <span
            aria-label={chip.ariaLabel}
            className={["status-chip", `status-chip-${chip.tone ?? "phase"}`].join(" ")}
            key={`${chip.label}-${chip.value ?? ""}`}
          >
            <span className="status-chip-label">{chip.label}</span>
            {chip.value === undefined ? null : (
              <strong className="status-chip-value">{chip.value}</strong>
            )}
          </span>
        ))}
      </div>
    </section>
  );
}
