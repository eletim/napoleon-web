import type {
  AiPolicyComposition,
  AiPreset,
  AiPresetId,
  PublicPhasePolicyDescriptor,
  PublicPhasePolicyRegistry
} from "@napoleon/protocol";

interface GamePresetSelectorProps {
  presets: readonly AiPreset[];
  selectedPresetId: AiPresetId;
  disabled: boolean;
  onChange(presetId: AiPresetId): void;
}

export function GamePresetSelector({
  presets,
  selectedPresetId,
  disabled,
  onChange
}: GamePresetSelectorProps) {
  return (
    <section className="agent-setup" aria-label="対戦AI選択">
      <h2>対戦AI</h2>
      <label className="agent-selector preset-selector">
        <span>AI preset</span>
        <select
          aria-label="対戦AI"
          disabled={disabled}
          onChange={(event) => onChange(event.target.value as AiPresetId)}
          value={selectedPresetId}
        >
          {presets.map((preset) => (
            <option key={preset.id} value={preset.id}>
              {preset.displayName}
            </option>
          ))}
        </select>
      </label>
    </section>
  );
}

interface AiSettingsPanelProps {
  presets: readonly AiPreset[];
  policyRegistry: PublicPhasePolicyRegistry | undefined;
  drafts: Readonly<Partial<Record<AiPresetId, AiPolicyComposition>>>;
  disabled: boolean;
  message: string;
  onChange(presetId: AiPresetId, phase: keyof AiPolicyComposition, policyId: string): void;
  onSave(presetId: AiPresetId): void;
}

export function AiSettingsPanel({
  presets,
  policyRegistry,
  drafts,
  disabled,
  message,
  onChange,
  onSave
}: AiSettingsPanelProps) {
  return (
    <section className="ai-settings" aria-label="AI設定">
      <header className="ai-settings-header">
        <div>
          <h1>AI設定</h1>
          <p>各COM presetが使用するphase policyを設定します。</p>
        </div>
        <p aria-live="polite" className="ai-settings-message">{message}</p>
      </header>
      <div className="ai-preset-list">
        {presets.map((preset) => {
          const draft = drafts[preset.id] ?? preset.composition;
          return (
            <article className="ai-preset-card" key={preset.id}>
              <header>
                <h2>{preset.displayName}</h2>
                <code>{preset.id}</code>
              </header>
              <PolicySelect
                disabled={disabled}
                label="Playing policy"
                onChange={(policyId) => onChange(preset.id, "playing", policyId)}
                policies={policyRegistry?.playing ?? []}
                value={draft.playing}
              />
              <PolicySelect
                disabled={disabled}
                label="Bidding policy"
                onChange={(policyId) => onChange(preset.id, "bidding", policyId)}
                policies={policyRegistry?.bidding ?? []}
                value={draft.bidding}
              />
              <PolicySelect
                disabled={disabled}
                label="Non-playing policy"
                onChange={(policyId) => onChange(preset.id, "nonPlaying", policyId)}
                policies={policyRegistry?.nonPlaying ?? []}
                value={draft.nonPlaying}
              />
              <button
                className="primary-button ai-preset-save"
                disabled={disabled || policyRegistry === undefined}
                onClick={() => onSave(preset.id)}
                type="button"
              >
                保存・適用
              </button>
            </article>
          );
        })}
      </div>
    </section>
  );
}

export function isPresetCompositionAvailable(
  composition: AiPolicyComposition,
  registry: PublicPhasePolicyRegistry | undefined
): boolean {
  return registry !== undefined &&
    isPolicyAvailable(composition.playing, registry.playing) &&
    isPolicyAvailable(composition.bidding, registry.bidding) &&
    isPolicyAvailable(composition.nonPlaying, registry.nonPlaying);
}

function PolicySelect({
  disabled,
  label,
  policies,
  value,
  onChange
}: {
  disabled: boolean;
  label: string;
  policies: readonly PublicPhasePolicyDescriptor[];
  value: string;
  onChange(policyId: string): void;
}) {
  return (
    <label className="ai-policy-field">
      <span>{label}</span>
      <select
        aria-label={label}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
        value={value}
      >
        {policies.map((policy) => (
          <option disabled={!policy.isAvailable} key={policy.id} value={policy.id}>
            {policy.id}{policy.isAvailable ? "" : " (利用不可)"}
          </option>
        ))}
      </select>
    </label>
  );
}

function isPolicyAvailable(
  policyId: string,
  policies: readonly PublicPhasePolicyDescriptor[]
): boolean {
  return policies.some((policy) => policy.id === policyId && policy.isAvailable);
}
