export const DATASET_SCHEMA_VERSION = 1 as const;
export const DATASET_GENERATOR_VERSION = 1 as const;
export const MULTIPHASE_DATASET_SCHEMA_VERSION = 2 as const;
export const MULTIPHASE_DATASET_GENERATOR_VERSION = 2 as const;
export const RULE_BASED_AGENT_VERSION = 1 as const;
export const DATASET_FORMAT = "jsonl" as const;
export const PLAYING_SELF_PLAY_BINARY_DATASET_FORMAT = "playing-self-play-binary-v1" as const;
export const PLAYING_DATASET_SAMPLE_TYPE = "playing-training-sample" as const;
export const BIDDING_DATASET_SAMPLE_TYPE = "bidding-training-sample" as const;
export const EXCHANGE_DATASET_SAMPLE_TYPE = "exchange-training-sample" as const;
export const ADJUTANT_DATASET_SAMPLE_TYPE = "adjutant-training-sample" as const;
export const DATASET_SAMPLE_TYPE = PLAYING_DATASET_SAMPLE_TYPE;
export const DATASET_SAMPLE_TYPES = [
  PLAYING_DATASET_SAMPLE_TYPE,
  BIDDING_DATASET_SAMPLE_TYPE,
  EXCHANGE_DATASET_SAMPLE_TYPE,
  ADJUTANT_DATASET_SAMPLE_TYPE
] as const;
export const SHARD_FILE_DIGITS = 5;
export const MAX_SHARD_COUNT = 10 ** SHARD_FILE_DIGITS;
export const UINT32_MAX = 0xffffffff;
