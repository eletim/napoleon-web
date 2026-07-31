import type { GameStatusDisplay, StatusChip } from "./displayText";

interface GameStatusProps {
  display: GameStatusDisplay;
}

export function GameStatus({ display }: GameStatusProps) {
  if (display.primary.length === 0 && display.secondary.length === 0) {
    return null;
  }

  return (
    <section className="game-status" aria-label="ゲーム状態">
      {display.primary.length > 0 ? (
        <StatusChipRow
          ariaLabel="主要ステータス"
          chips={display.primary}
          className="primary-status-row"
          chipClassName="status-chip-primary"
        />
      ) : null}

      {display.secondary.length > 0 ? (
        <StatusChipRow
          ariaLabel="補助ステータス"
          chips={display.secondary}
          className="secondary-status-row"
          chipClassName="status-chip-secondary"
        />
      ) : null}
    </section>
  );
}

interface StatusChipRowProps {
  ariaLabel: string;
  chips: readonly StatusChip[];
  className: string;
  chipClassName: string;
}

function StatusChipRow({
  ariaLabel,
  chips,
  className,
  chipClassName
}: StatusChipRowProps) {
  return (
    <div className={className} aria-label={ariaLabel}>
      {chips.map((chip) => (
        <span className={["status-chip", chipClassName].join(" ")} key={`${chip.label}-${chip.value ?? ""}`}>
          <span className="status-chip-label">{chip.label}</span>
          {chip.value === undefined ? null : (
            <strong className="status-chip-value">{chip.value}</strong>
          )}
        </span>
      ))}
    </div>
  );
}
