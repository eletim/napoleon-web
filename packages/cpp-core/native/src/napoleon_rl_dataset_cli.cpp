#include "napoleon_core.hpp"
#include "napoleon_observation.hpp"
#include "napoleon_onnx_policy.hpp"
#include "napoleon_roster.hpp"
#include "napoleon_rule_based.hpp"

#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

namespace {

using napoleon::Action;
using napoleon::AgentIdentity;
using napoleon::AgentType;
using napoleon::Card;
using napoleon::GameResult;
using napoleon::GameState;
using napoleon::RosterAssignment;
using napoleon::RosterSpec;

constexpr std::uint32_t kDatasetSchemaVersion = 4;
constexpr std::uint32_t kDatasetGeneratorVersion = 1;
constexpr std::uint32_t kSampleSchemaVersion = 4;
constexpr std::uint32_t kShardSchemaVersion = 1;
constexpr std::uint32_t kRuleBasedAgentVersion = 1;
constexpr std::uint32_t kRewardVersion = 1;
constexpr int kCardCount = napoleon::observation::kCardCount;
constexpr int kSelfRoleCount = 4;
constexpr char kBinaryMagic[] = "NPSPBD01";

struct Sha256 {
  std::array<std::uint32_t, 8> state{
      0x6a09e667u,
      0xbb67ae85u,
      0x3c6ef372u,
      0xa54ff53au,
      0x510e527fu,
      0x9b05688cu,
      0x1f83d9abu,
      0x5be0cd19u};
  std::array<std::uint8_t, 64> buffer{};
  std::uint64_t bit_length = 0;
  std::size_t buffer_length = 0;

  void update(const std::uint8_t* data, std::size_t length) {
    bit_length += static_cast<std::uint64_t>(length) * 8u;
    for (std::size_t index = 0; index < length; ++index) {
      buffer[buffer_length++] = data[index];
      if (buffer_length == buffer.size()) {
        transform(buffer.data());
        buffer_length = 0;
      }
    }
  }

  void update(const std::vector<std::uint8_t>& data) {
    update(data.data(), data.size());
  }

  void update(const std::string& value) {
    update(reinterpret_cast<const std::uint8_t*>(value.data()), value.size());
  }

  std::string hexdigest() {
    const std::uint64_t final_bit_length = bit_length;
    buffer[buffer_length++] = 0x80u;
    if (buffer_length > 56) {
      while (buffer_length < 64) {
        buffer[buffer_length++] = 0;
      }
      transform(buffer.data());
      buffer_length = 0;
    }
    while (buffer_length < 56) {
      buffer[buffer_length++] = 0;
    }
    for (int shift = 56; shift >= 0; shift -= 8) {
      buffer[buffer_length++] = static_cast<std::uint8_t>((final_bit_length >> shift) & 0xffu);
    }
    transform(buffer.data());

    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (std::uint32_t word : state) {
      out << std::setw(8) << word;
    }
    return out.str();
  }

  static std::uint32_t rotr(std::uint32_t value, int shift) {
    return (value >> shift) | (value << (32 - shift));
  }

  void transform(const std::uint8_t* chunk) {
    static constexpr std::array<std::uint32_t, 64> k{
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
        0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
        0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
        0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
        0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
        0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
        0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
        0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
        0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
        0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
        0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
    std::array<std::uint32_t, 64> w{};
    for (std::size_t index = 0; index < 16; ++index) {
      w[index] = (static_cast<std::uint32_t>(chunk[index * 4]) << 24) |
                 (static_cast<std::uint32_t>(chunk[index * 4 + 1]) << 16) |
                 (static_cast<std::uint32_t>(chunk[index * 4 + 2]) << 8) |
                 static_cast<std::uint32_t>(chunk[index * 4 + 3]);
    }
    for (std::size_t index = 16; index < 64; ++index) {
      const std::uint32_t s0 =
          rotr(w[index - 15], 7) ^ rotr(w[index - 15], 18) ^ (w[index - 15] >> 3);
      const std::uint32_t s1 =
          rotr(w[index - 2], 17) ^ rotr(w[index - 2], 19) ^ (w[index - 2] >> 10);
      w[index] = w[index - 16] + s0 + w[index - 7] + s1;
    }

    std::uint32_t a = state[0];
    std::uint32_t b = state[1];
    std::uint32_t c = state[2];
    std::uint32_t d = state[3];
    std::uint32_t e = state[4];
    std::uint32_t f = state[5];
    std::uint32_t g = state[6];
    std::uint32_t h = state[7];
    for (std::size_t index = 0; index < 64; ++index) {
      const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const std::uint32_t ch = (e & f) ^ ((~e) & g);
      const std::uint32_t temp1 = h + s1 + ch + k[index] + w[index];
      const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temp2 = s0 + maj;
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
  }
};

struct CliOptions {
  std::filesystem::path output_directory;
  std::filesystem::path policy_onnx_path;
  std::filesystem::path policy_metadata_path;
  std::filesystem::path frozen_onnx_path;
  std::filesystem::path frozen_metadata_path;
  std::string policy_artifact_id = "cpp-current-policy";
  std::string frozen_artifact_id = "rl-v740";
  std::uint32_t start_seed = 0;
  std::uint32_t game_count = 0;
  std::uint32_t games_per_shard = 0;
  std::uint32_t roster_seed = 0;
  std::uint32_t max_concurrent_games = 1;
  std::uint32_t inference_max_batch_size = 1;
  double temperature = 1.0;
  std::string inference_device = "cpu";
  std::string policy_backend = "deterministic";
  napoleon::observation::PlayingObservationVariant playing_observation_variant =
      napoleon::observation::PlayingObservationVariant::Public;
  bool all_current = false;
};

struct TensorSample {
  std::vector<float> model_input;
  std::array<std::uint8_t, kCardCount> legal_play_mask{};
  std::uint8_t selected_card_index = 0;
  float behavior_log_probability = 0.0F;
  std::int8_t terminal_reward = 1;
  std::uint32_t seed = 0;
  std::uint16_t step = 0;
  std::uint8_t acting_player_index = 0;
  std::uint8_t self_role_index = 0;
};

struct ShardManifest {
  std::string file;
  std::uint32_t start_seed = 0;
  std::uint32_t end_seed = 0;
  std::uint32_t game_count = 0;
  std::uint64_t sample_count = 0;
  std::uint64_t byte_length = 0;
  std::string sha256;
};

void json_escape(std::ostream& out, const std::string& value) {
  out << '"';
  for (char ch : value) {
    switch (ch) {
      case '"':
        out << "\\\"";
        break;
      case '\\':
        out << "\\\\";
        break;
      case '\n':
        out << "\\n";
        break;
      default:
        out << ch;
        break;
    }
  }
  out << '"';
}

std::string json_string(const std::string& value) {
  std::ostringstream out;
  json_escape(out, value);
  return out.str();
}

std::string shard_file_name(std::uint32_t shard_index) {
  std::ostringstream out;
  out << "shard-" << std::setw(5) << std::setfill('0') << shard_index << ".bin";
  return out.str();
}

void write_u16_le(std::ostream& out, std::uint16_t value) {
  out.put(static_cast<char>(value & 0xffu));
  out.put(static_cast<char>((value >> 8) & 0xffu));
}

void write_u32_le(std::ostream& out, std::uint32_t value) {
  out.put(static_cast<char>(value & 0xffu));
  out.put(static_cast<char>((value >> 8) & 0xffu));
  out.put(static_cast<char>((value >> 16) & 0xffu));
  out.put(static_cast<char>((value >> 24) & 0xffu));
}

void write_float32_le(std::ostream& out, float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "float must be 32-bit");
  std::memcpy(&bits, &value, sizeof(bits));
  write_u32_le(out, bits);
}

void ensure_good(const std::ostream& out, const std::string& path) {
  if (!out.good()) {
    throw std::runtime_error("failed writing " + path);
  }
}

class BinaryShardWriter {
 public:
  BinaryShardWriter(
      const std::filesystem::path& output_directory,
      const std::filesystem::path& temp_root,
      std::uint32_t shard_index,
      std::uint32_t start_seed,
      napoleon::observation::PlayingObservationVariant playing_observation_variant)
      : output_directory_(output_directory),
        temp_directory_(temp_root / ("shard-" + std::to_string(shard_index))),
        file_name_(shard_file_name(shard_index)),
        start_seed_(start_seed),
        playing_observation_variant_(playing_observation_variant),
        model_input_feature_count_(static_cast<std::size_t>(
            napoleon::observation::playing_model_input_feature_count(playing_observation_variant))) {
    std::filesystem::create_directories(temp_directory_);
    for (std::size_t index = 0; index < field_files_.size(); ++index) {
      field_files_[index] = temp_directory_ / ("field-" + std::to_string(index) + ".bin");
      streams_[index].open(field_files_[index], std::ios::binary);
      if (!streams_[index].is_open()) {
        throw std::runtime_error("failed opening shard field temp file");
      }
    }
  }

  ~BinaryShardWriter() {
    for (std::ofstream& stream : streams_) {
      if (stream.is_open()) {
        stream.close();
      }
    }
  }

  void write_sample(const TensorSample& sample) {
    if (sample.model_input.size() != model_input_feature_count_) {
      throw std::runtime_error("tensor sample model input feature count mismatch");
    }
    for (float value : sample.model_input) {
      write_float32_le(streams_[0], value);
    }
    for (std::uint8_t value : sample.legal_play_mask) {
      streams_[1].put(static_cast<char>(value));
    }
    streams_[2].put(static_cast<char>(sample.selected_card_index));
    write_float32_le(streams_[3], sample.behavior_log_probability);
    streams_[4].put(static_cast<char>(sample.terminal_reward));
    write_u32_le(streams_[5], sample.seed);
    write_u16_le(streams_[6], sample.step);
    streams_[7].put(static_cast<char>(sample.acting_player_index));
    streams_[8].put(static_cast<char>(sample.self_role_index));

    for (std::size_t index = 0; index < streams_.size(); ++index) {
      ensure_good(streams_[index], field_files_[index].string());
    }
    sample_count_ += 1;
  }

  std::uint64_t sample_count() const {
    return sample_count_;
  }

  ShardManifest close(std::uint32_t end_seed, std::uint32_t game_count) {
    if (sample_count_ == 0) {
      throw std::runtime_error("binary shard must contain at least one sample");
    }
    for (std::ofstream& stream : streams_) {
      stream.close();
    }

    std::array<std::uint64_t, 9> field_lengths{};
    for (std::size_t index = 0; index < field_files_.size(); ++index) {
      field_lengths[index] = std::filesystem::file_size(field_files_[index]);
    }

    const std::string header = header_json(field_lengths);
    const std::filesystem::path output_path = output_directory_ / file_name_;
    std::ofstream out(output_path, std::ios::binary);
    if (!out.is_open()) {
      throw std::runtime_error("failed opening output shard " + output_path.string());
    }

    Sha256 hasher;
    std::uint64_t total_bytes = 0;
    const auto write_bytes = [&](const std::uint8_t* bytes, std::size_t length) {
      out.write(reinterpret_cast<const char*>(bytes), static_cast<std::streamsize>(length));
      hasher.update(bytes, length);
      total_bytes += length;
    };
    const auto write_string = [&](const std::string& bytes) {
      out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
      hasher.update(bytes);
      total_bytes += bytes.size();
    };

    write_string(std::string(kBinaryMagic, sizeof(kBinaryMagic) - 1));
    std::ostringstream length_stream;
    write_u32_le(length_stream, static_cast<std::uint32_t>(header.size()));
    write_string(length_stream.str());
    write_string(header);
    for (const std::filesystem::path& field_file : field_files_) {
      std::ifstream in(field_file, std::ios::binary);
      std::vector<std::uint8_t> buffer(1024 * 1024);
      while (in.good()) {
        in.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize read_count = in.gcount();
        if (read_count > 0) {
          write_bytes(buffer.data(), static_cast<std::size_t>(read_count));
        }
      }
    }
    out.close();
    if (!out.good()) {
      throw std::runtime_error("failed finalizing shard " + output_path.string());
    }
    std::filesystem::remove_all(temp_directory_);

    return ShardManifest{
        file_name_, start_seed_, end_seed, game_count, sample_count_, total_bytes, hasher.hexdigest()};
  }

 private:
  std::string header_json(const std::array<std::uint64_t, 9>& field_lengths) const {
    static constexpr std::array<const char*, 9> names{
        "modelInput",
        "legalPlayMask",
        "selectedCardIndex",
        "behaviorLogProbability",
        "terminalReward",
        "seed",
        "step",
        "actingPlayerIndex",
        "selfRoleIndex"};
    static constexpr std::array<const char*, 9> dtypes{
        "float32", "uint8", "uint8", "float32", "int8", "uint32", "uint16", "uint8", "uint8"};
    const std::array<std::string, 9> shapes{
        "[" + std::to_string(model_input_feature_count_) + "]",
        "[53]",
        "[]",
        "[]",
        "[]",
        "[]",
        "[]",
        "[]",
        "[]"};

    std::ostringstream out;
    std::uint64_t offset = 0;
    out << "{\"shardSchemaVersion\":" << kShardSchemaVersion
        << ",\"sampleType\":\"playing-self-play-sample\""
        << ",\"sampleSchemaVersion\":" << kSampleSchemaVersion
        << ",\"sampleCount\":" << sample_count_
        << ",\"playingObservationVariant\":";
    json_escape(out, napoleon::observation::playing_observation_variant_id(playing_observation_variant_));
    out << ",\"modelInputFeatureCount\":" << model_input_feature_count_
        << ",\"cardCount\":" << kCardCount
        << ",\"byteOrder\":\"little-endian\""
        << ",\"compression\":\"none\""
        << ",\"uncompressedByteLength\":";
    std::uint64_t total = 0;
    for (std::uint64_t length : field_lengths) {
      total += length;
    }
    out << total << ",\"fields\":[";
    for (std::size_t index = 0; index < names.size(); ++index) {
      if (index != 0) {
        out << ',';
      }
      out << "{\"name\":\"" << names[index]
          << "\",\"dtype\":\"" << dtypes[index]
          << "\",\"shape\":" << shapes[index]
          << ",\"byteOffset\":" << offset
          << ",\"byteLength\":" << field_lengths[index] << '}';
      offset += field_lengths[index];
    }
    out << "]}";
    return out.str();
  }

  std::filesystem::path output_directory_;
  std::filesystem::path temp_directory_;
  std::array<std::filesystem::path, 9> field_files_{};
  std::array<std::ofstream, 9> streams_{};
  std::string file_name_;
  std::uint32_t start_seed_ = 0;
  napoleon::observation::PlayingObservationVariant playing_observation_variant_;
  std::size_t model_input_feature_count_ = napoleon::observation::kPlayingModelInputFeatureCount;
  std::uint64_t sample_count_ = 0;
};

std::uint32_t parse_uint32(const std::string& value, const std::string& name) {
  unsigned long long parsed = 0;
  try {
    parsed = std::stoull(value);
  } catch (const std::exception&) {
    throw std::runtime_error(name + " must be an integer between 0 and 4294967295");
  }
  if (parsed > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error(name + " must be an integer between 0 and 4294967295");
  }
  return static_cast<std::uint32_t>(parsed);
}

double parse_temperature(const std::string& value) {
  double parsed = 0.0;
  try {
    parsed = std::stod(value);
  } catch (const std::exception&) {
    throw std::runtime_error("temperature must be a finite positive number");
  }
  if (!std::isfinite(parsed) || parsed <= 0.0) {
    throw std::runtime_error("temperature must be a finite positive number");
  }
  return parsed;
}

std::string parse_inference_device(const std::string& value) {
  if (value == "cpu" || value == "auto" || value == "cuda") {
    return value;
  }
  throw std::runtime_error("--inference-device must be one of cpu, auto, cuda");
}

std::string parse_policy_backend(const std::string& value) {
  if (value == "deterministic" || value == "onnx") {
    return value;
  }
  throw std::runtime_error("--policy-backend must be one of deterministic, onnx");
}

napoleon::observation::PlayingObservationVariant parse_playing_observation_variant(
    const std::string& value) {
  return napoleon::observation::parse_playing_observation_variant(value);
}

CliOptions parse_args(int argc, char** argv) {
  CliOptions options;
  for (int index = 1; index < argc; ++index) {
    const std::string arg = argv[index];
    const auto require_value = [&](const std::string& name) -> std::string {
      if (index + 1 >= argc) {
        throw std::runtime_error(name + " requires a value");
      }
      return argv[++index];
    };

    if (arg == "--output") {
      options.output_directory = require_value(arg);
    } else if (arg == "--policy-onnx") {
      options.policy_onnx_path = require_value(arg);
    } else if (arg == "--policy-metadata") {
      options.policy_metadata_path = require_value(arg);
    } else if (arg == "--policy-artifact-id") {
      options.policy_artifact_id = require_value(arg);
    } else if (arg == "--frozen-onnx") {
      options.frozen_onnx_path = require_value(arg);
    } else if (arg == "--frozen-metadata") {
      options.frozen_metadata_path = require_value(arg);
    } else if (arg == "--frozen-artifact-id") {
      options.frozen_artifact_id = require_value(arg);
    } else if (arg == "--start-seed") {
      options.start_seed = parse_uint32(require_value(arg), "start-seed");
    } else if (arg == "--games") {
      options.game_count = parse_uint32(require_value(arg), "games");
    } else if (arg == "--games-per-shard") {
      options.games_per_shard = parse_uint32(require_value(arg), "games-per-shard");
    } else if (arg == "--roster-seed") {
      options.roster_seed = parse_uint32(require_value(arg), "roster-seed");
    } else if (arg == "--max-concurrent-games" || arg == "--rollout-concurrency") {
      options.max_concurrent_games = parse_uint32(require_value(arg), "max-concurrent-games");
    } else if (arg == "--inference-max-batch-size") {
      options.inference_max_batch_size = parse_uint32(require_value(arg), "inference-max-batch-size");
    } else if (arg == "--inference-device") {
      options.inference_device = parse_inference_device(require_value(arg));
    } else if (arg == "--policy-backend") {
      options.policy_backend = parse_policy_backend(require_value(arg));
    } else if (arg == "--playing-observation-variant") {
      options.playing_observation_variant =
          parse_playing_observation_variant(require_value(arg));
    } else if (arg == "--temperature") {
      options.temperature = parse_temperature(require_value(arg));
    } else if (arg == "--all-current") {
      options.all_current = true;
    } else {
      throw std::runtime_error(
          "usage: napoleon_rl_dataset_cli --output <dir> --start-seed <uint32> "
          "--games <n> --games-per-shard <n> --policy-onnx <path> "
          "--policy-metadata <path> [--policy-artifact-id <id>] "
          "[--frozen-onnx <path>] [--frozen-metadata <path>] [--frozen-artifact-id <id>] "
          "[--roster-seed <uint32>] [--temperature <positive>] "
          "[--max-concurrent-games <n>] [--inference-max-batch-size <n>] "
          "[--inference-device cpu|auto|cuda] [--policy-backend deterministic|onnx] "
          "[--playing-observation-variant public|complete-info-compact] [--all-current]");
    }
  }

  if (options.output_directory.empty()) {
    throw std::runtime_error("--output is required");
  }
  if (options.policy_onnx_path.empty()) {
    throw std::runtime_error("--policy-onnx is required");
  }
  if (options.policy_metadata_path.empty()) {
    throw std::runtime_error("--policy-metadata is required");
  }
  if (options.frozen_onnx_path.empty()) {
    options.frozen_onnx_path = options.policy_onnx_path;
  }
  if (options.frozen_metadata_path.empty()) {
    options.frozen_metadata_path = options.policy_metadata_path;
  }
  if (options.policy_artifact_id.empty()) {
    throw std::runtime_error("--policy-artifact-id must not be empty");
  }
  if (options.frozen_artifact_id.empty()) {
    throw std::runtime_error("--frozen-artifact-id must not be empty");
  }
  if (options.game_count == 0) {
    throw std::runtime_error("--games must be positive");
  }
  if (options.games_per_shard == 0) {
    throw std::runtime_error("--games-per-shard must be positive");
  }
  if (options.max_concurrent_games == 0) {
    throw std::runtime_error("--max-concurrent-games must be positive");
  }
  if (options.inference_max_batch_size == 0) {
    throw std::runtime_error("--inference-max-batch-size must be positive");
  }
  if (options.inference_device == "cuda" && options.policy_backend != "onnx") {
    throw std::runtime_error("--inference-device cuda requires --policy-backend onnx");
  }
  if (options.start_seed > std::numeric_limits<std::uint32_t>::max() - options.game_count + 1u) {
    throw std::runtime_error("start-seed + games exceeds uint32 range");
  }
  return options;
}

TensorSample create_current_policy_sample(
    const napoleon::AgentRequest& request,
    const napoleon::AgentResult& result,
    std::uint32_t seed,
    napoleon::observation::PlayingObservationVariant observation_variant) {
  const std::size_t model_input_feature_count = static_cast<std::size_t>(
      napoleon::observation::playing_model_input_feature_count(observation_variant));
  if (request.playing_model_input.size() != model_input_feature_count) {
    throw std::runtime_error(
        "playing request model input feature count mismatch: expected " +
        std::to_string(model_input_feature_count) + ", got " +
        std::to_string(request.playing_model_input.size()));
  }
  if (request.legal_play_mask.size() != static_cast<std::size_t>(kCardCount)) {
    throw std::runtime_error("playing request legal mask must contain 53 entries");
  }
  if (result.selected_card_index < 0 || result.selected_card_index >= kCardCount) {
    throw std::runtime_error("selected card index out of range");
  }
  if (request.legal_play_mask[static_cast<std::size_t>(result.selected_card_index)] != 1) {
    throw std::runtime_error("selected current-policy card is not legal");
  }
  if (request.game_decision_count > std::numeric_limits<std::uint16_t>::max()) {
    throw std::runtime_error("game step exceeds uint16 dataset field range");
  }

  TensorSample sample;
  sample.model_input = request.playing_model_input;
  for (std::size_t index = 0; index < request.legal_play_mask.size(); ++index) {
    const int value = request.legal_play_mask[index];
    if (value != 0 && value != 1) {
      throw std::runtime_error("legal play mask must contain only 0/1 values");
    }
    sample.legal_play_mask[index] = static_cast<std::uint8_t>(value);
  }
  sample.selected_card_index = static_cast<std::uint8_t>(result.selected_card_index);
  sample.behavior_log_probability = static_cast<float>(result.behavior_log_probability);
  sample.seed = seed;
  sample.step = static_cast<std::uint16_t>(request.game_decision_count);
  sample.acting_player_index = static_cast<std::uint8_t>(request.player_index);
  const std::size_t self_role_offset =
      observation_variant == napoleon::observation::PlayingObservationVariant::Public
          ? model_input_feature_count - kSelfRoleCount
          : 339;
  for (int index = 0; index < kSelfRoleCount; ++index) {
    const float value = sample.model_input[self_role_offset + static_cast<std::size_t>(index)];
    if (value == 1.0F) {
      sample.self_role_index = static_cast<std::uint8_t>(index);
      return sample;
    }
  }
  throw std::runtime_error("playing request model input has no self role");
}

int terminal_reward_for_role(const GameResult& result, std::uint8_t role_index) {
  const std::string acting_team =
      role_index == 2 ? "alliance" : "napoleon-team";
  return acting_team == result.winner ? 1 : -1;
}

napoleon::onnx_policy::InferenceDevice policy_inference_device(const CliOptions& options) {
  return options.inference_device == "cuda"
             ? napoleon::onnx_policy::InferenceDevice::Cuda
             : napoleon::onnx_policy::InferenceDevice::Cpu;
}

std::unique_ptr<napoleon::onnx_policy::PolicySession> create_policy_session(
    const CliOptions& options,
    napoleon::onnx_policy::PolicyKey key,
    const std::filesystem::path& onnx_path,
    napoleon::observation::PlayingObservationVariant observation_variant) {
  const std::size_t model_input_feature_count = static_cast<std::size_t>(
      napoleon::observation::playing_model_input_feature_count(observation_variant));
  if (options.policy_backend == "onnx") {
    return napoleon::onnx_policy::create_onnxruntime_policy_session(
        napoleon::onnx_policy::PolicySessionConfig{
            key,
            onnx_path.string(),
            "model_input",
            "logits",
            policy_inference_device(options),
            model_input_feature_count});
  }

  return std::make_unique<napoleon::onnx_policy::DeterministicPolicySession>(
      napoleon::onnx_policy::DeterministicPolicySession::default_logits(),
      options.inference_device == "cuda"
          ? napoleon::onnx_policy::ExecutionProvider::Cuda
          : napoleon::onnx_policy::ExecutionProvider::Cpu,
      model_input_feature_count);
}

std::unique_ptr<napoleon::onnx_policy::BatchedPolicyExecutor> create_policy_executor(
    const CliOptions& options) {
  auto executor = std::make_unique<napoleon::onnx_policy::BatchedPolicyExecutor>(
      napoleon::onnx_policy::BatchedPolicyConfig{
          std::max<std::size_t>(1, options.inference_max_batch_size),
          options.temperature,
          options.roster_seed});
  const napoleon::onnx_policy::PolicyKey current_key{
      AgentType::CurrentPolicy,
      "current"};
  const napoleon::onnx_policy::PolicyKey frozen_key{
      AgentType::FrozenPolicy,
      options.frozen_artifact_id};
  executor->add_policy(
      current_key,
      create_policy_session(
          options,
          current_key,
          options.policy_onnx_path,
          options.playing_observation_variant));
  executor->add_policy(
      frozen_key,
      create_policy_session(
          options,
          frozen_key,
          options.frozen_onnx_path,
          napoleon::observation::PlayingObservationVariant::Public));
  return executor;
}

void submit_dataset_policy_requests(
    napoleon::SimulationRuntime& runtime,
    napoleon::onnx_policy::BatchedPolicyExecutor& executor,
    const std::vector<napoleon::AgentRequest>& requests,
    std::vector<std::vector<TensorSample>>& samples_by_game,
    std::uint32_t start_seed,
    napoleon::observation::PlayingObservationVariant observation_variant,
    std::size_t max_batch_size) {
  std::vector<napoleon::AgentResult> results;
  results.reserve(requests.size());

  std::map<std::string, std::vector<napoleon::AgentRequest>> playing_by_policy;
  for (const napoleon::AgentRequest& request : requests) {
    if (
        (request.agent.type == AgentType::CurrentPolicy ||
         request.agent.type == AgentType::FrozenPolicy) &&
        request.phase == napoleon::Phase::Playing) {
      playing_by_policy[napoleon::onnx_policy::policy_key_id(
          napoleon::onnx_policy::policy_key_from_agent(request.agent))].push_back(request);
      continue;
    }

    napoleon::AgentResult result;
    result.request_id = request.request_id;
    if (request.phase == napoleon::Phase::Bidding) {
      const auto bid_it = std::find_if(
          request.legal_actions.begin(),
          request.legal_actions.end(),
          [](const Action& action) {
            return action.type == Action::Type::Bid;
          });
      result.action = bid_it == request.legal_actions.end() ? request.legal_actions.front() : *bid_it;
    } else {
      result.action = request.legal_actions.front();
    }
    results.push_back(result);
  }

  for (const auto& [_, policy_requests] : playing_by_policy) {
    std::size_t offset = 0;
    while (offset < policy_requests.size()) {
      const std::size_t batch_size = std::min(max_batch_size, policy_requests.size() - offset);
      std::vector<napoleon::AgentRequest> batch(
          policy_requests.begin() + static_cast<std::ptrdiff_t>(offset),
          policy_requests.begin() + static_cast<std::ptrdiff_t>(offset + batch_size));
      std::vector<napoleon::onnx_policy::PolicyActionResult> policy_results =
          executor.run(batch);
      if (policy_results.size() != batch.size()) {
        throw std::runtime_error("policy executor returned a mismatched result count");
      }
      std::unordered_map<std::uint64_t, const napoleon::AgentRequest*> request_by_id;
      for (const napoleon::AgentRequest& request : batch) {
        request_by_id[request.request_id] = &request;
      }
      for (const napoleon::onnx_policy::PolicyActionResult& policy_result : policy_results) {
        const auto request_it = request_by_id.find(policy_result.result.request_id);
        if (request_it == request_by_id.end()) {
          throw std::runtime_error("policy result request id was not in the submitted batch");
        }
        const napoleon::AgentRequest& request = *request_it->second;
        if (request.agent.type == AgentType::CurrentPolicy) {
          samples_by_game[request.game_index].push_back(
              create_current_policy_sample(
                  request,
                  policy_result.result,
                  start_seed + request.game_index,
                  observation_variant));
        }
        results.push_back(policy_result.result);
      }
      offset += batch_size;
    }
  }

  runtime.submit_agent_results(results);
}

std::string card_ids_json() {
  static constexpr std::array<const char*, 4> suits{"spades", "hearts", "diamonds", "clubs"};
  static constexpr std::array<const char*, 13> ranks{
      "A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2"};
  std::ostringstream out;
  out << '[';
  bool first = true;
  for (const char* suit : suits) {
    for (const char* rank : ranks) {
      if (!first) {
        out << ',';
      }
      first = false;
      out << '"' << suit << '-' << rank << '"';
    }
  }
  out << ",\"joker\"]";
  return out.str();
}

std::string sha256_hex(const std::string& value) {
  Sha256 hasher;
  hasher.update(value);
  return hasher.hexdigest();
}

std::string sha256_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("failed opening file for SHA-256: " + path.string());
  }
  Sha256 hasher;
  std::array<std::uint8_t, 1024 * 1024> buffer{};
  while (in.good()) {
    in.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize read_count = in.gcount();
    if (read_count > 0) {
      hasher.update(buffer.data(), static_cast<std::size_t>(read_count));
    }
  }
  return hasher.hexdigest();
}

std::string base_name(const std::filesystem::path& path) {
  return path.filename().string();
}

std::string read_text_file(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in.is_open()) {
    throw std::runtime_error("failed opening metadata JSON: " + path.string());
  }
  std::ostringstream buffer;
  buffer << in.rdbuf();
  if (!in.good() && !in.eof()) {
    throw std::runtime_error("failed reading metadata JSON: " + path.string());
  }
  return buffer.str();
}

std::optional<std::string> json_string_field(const std::string& json, const std::string& key) {
  const std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
  std::smatch match;
  if (!std::regex_search(json, match, pattern)) {
    return std::nullopt;
  }
  return match[1].str();
}

std::optional<int> json_integer_field(const std::string& json, const std::string& key) {
  const std::regex pattern("\"" + key + "\"\\s*:\\s*(\\d+)");
  std::smatch match;
  if (!std::regex_search(json, match, pattern)) {
    return std::nullopt;
  }
  return std::stoi(match[1].str());
}

void validate_policy_metadata(
    const std::string& label,
    const std::string& metadata_json,
    napoleon::observation::PlayingObservationVariant expected_variant) {
  const std::string expected_variant_id =
      napoleon::observation::playing_observation_variant_id(expected_variant);
  const int expected_encoder_schema =
      napoleon::observation::playing_encoder_schema_version(expected_variant);
  const int expected_model_input_schema =
      napoleon::observation::playing_model_input_schema_version(expected_variant);
  const int expected_feature_count =
      napoleon::observation::playing_model_input_feature_count(expected_variant);

  const std::string actual_variant_id =
      json_string_field(metadata_json, "playingObservationVariant").value_or("public");
  if (actual_variant_id != expected_variant_id) {
    throw std::runtime_error(
        label + " metadata playingObservationVariant mismatch: expected " +
        expected_variant_id + ", got " + actual_variant_id);
  }

  const auto require_int = [&](const std::string& key, int expected) {
    const std::optional<int> actual = json_integer_field(metadata_json, key);
    if (!actual.has_value() || *actual != expected) {
      throw std::runtime_error(
          label + " metadata " + key + " mismatch: expected " +
          std::to_string(expected) + ", got " +
          (actual.has_value() ? std::to_string(*actual) : std::string("<missing>")));
    }
  };

  require_int("playingEncoderSchemaVersion", expected_encoder_schema);
  require_int("modelInputSchemaVersion", expected_model_input_schema);
  const std::optional<int> model_input_feature_count =
      json_integer_field(metadata_json, "modelInputFeatureCount");
  if (model_input_feature_count.has_value() && *model_input_feature_count != expected_feature_count) {
    throw std::runtime_error(
        label + " metadata modelInputFeatureCount mismatch: expected " +
        std::to_string(expected_feature_count) + ", got " +
        std::to_string(*model_input_feature_count));
  }
  const std::optional<int> input_dim = json_integer_field(metadata_json, "input_dim");
  if (!input_dim.has_value() || *input_dim != expected_feature_count) {
    throw std::runtime_error(
        label + " metadata policyModel.input_dim mismatch: expected " +
        std::to_string(expected_feature_count) + ", got " +
        (input_dim.has_value() ? std::to_string(*input_dim) : std::string("<missing>")));
  }
}

std::string resolved_inference_device(const CliOptions& options) {
  return options.inference_device == "cuda" ? "cuda" : "cpu";
}

void write_roster_seat_manifest(
    std::ostream& out,
    const std::string& source,
    const CliOptions& options,
    const std::string& frozen_onnx_sha256,
    const std::string& frozen_metadata_sha256,
    const std::string& frozen_metadata_json) {
  if (source == "current-policy") {
    out << "{\"source\":\"current-policy\"}";
  } else if (source == "rule-based") {
    out << "{\"source\":\"rule-based\",\"version\":" << kRuleBasedAgentVersion << '}';
  } else if (source == "frozen-onnx") {
    out << "{\"source\":\"frozen-onnx\","
        << "\"artifactId\":";
    json_escape(out, options.frozen_artifact_id);
    out << ",\"onnxFileName\":";
    json_escape(out, base_name(options.frozen_onnx_path));
    out << ",\"metadataFileName\":";
    json_escape(out, base_name(options.frozen_metadata_path));
    out << ",\"onnxSha256\":\"" << frozen_onnx_sha256 << "\","
        << "\"metadataSha256\":\"" << frozen_metadata_sha256 << "\","
        << "\"requestedInferenceDevice\":";
    json_escape(out, options.inference_device);
    out << ",\"resolvedInferenceDevice\":";
    json_escape(out, resolved_inference_device(options));
    out << ",\"executionProvider\":";
    json_escape(out, resolved_inference_device(options));
    out << ",\"metadata\":" << frozen_metadata_json << '}';
  } else {
    throw std::runtime_error("invalid roster source");
  }
}

void write_manifest(
    const std::filesystem::path& output_directory,
    const CliOptions& options,
    const std::vector<ShardManifest>& shards,
    std::uint64_t sample_count,
    const napoleon::onnx_policy::BatchedPolicyStats& inference_stats) {
  const std::string card_ids = card_ids_json();
  const std::string policy_onnx_sha256 = sha256_file(options.policy_onnx_path);
  const std::string policy_metadata_sha256 = sha256_file(options.policy_metadata_path);
  const std::string frozen_onnx_sha256 = sha256_file(options.frozen_onnx_path);
  const std::string frozen_metadata_sha256 = sha256_file(options.frozen_metadata_path);
  const std::string policy_metadata_json = read_text_file(options.policy_metadata_path);
  const std::string frozen_metadata_json = read_text_file(options.frozen_metadata_path);
  const std::uint32_t end_seed = options.start_seed + options.game_count - 1u;
  const int model_input_feature_count =
      napoleon::observation::playing_model_input_feature_count(options.playing_observation_variant);
  const std::string observation_variant_id =
      napoleon::observation::playing_observation_variant_id(options.playing_observation_variant);
  std::ofstream out(output_directory / "manifest.json");
  if (!out.is_open()) {
    throw std::runtime_error("failed opening manifest.json");
  }
  out << "{\n";
  out << "  \"datasetSchemaVersion\": " << kDatasetSchemaVersion << ",\n";
  out << "  \"generatorVersion\": " << kDatasetGeneratorVersion << ",\n";
  out << "  \"format\": \"playing-self-play-binary-v1\",\n";
  out << "  \"sampleType\": \"playing-self-play-sample\",\n";
  out << "  \"sampleSchemaVersion\": " << kSampleSchemaVersion << ",\n";
  out << "  \"startSeed\": " << options.start_seed << ",\n";
  out << "  \"endSeed\": " << end_seed << ",\n";
  out << "  \"gameCount\": " << options.game_count << ",\n";
  out << "  \"sampleCount\": " << sample_count << ",\n";
  out << "  \"gamesPerShard\": " << options.games_per_shard << ",\n";
  out << "  \"shardCount\": " << shards.size() << ",\n";
  out << "  \"simulationBackend\": \"cpp\",\n";
  out << "  \"runtime\": {\"requestedInferenceDevice\": ";
  json_escape(out, options.inference_device);
  out << ", \"resolvedInferenceDevice\": ";
  json_escape(out, resolved_inference_device(options));
  out << ", \"executionProvider\": ";
  json_escape(out, resolved_inference_device(options));
  out << ", \"policyBackend\": ";
  json_escape(out, options.policy_backend);
  out << ", \"rolloutConcurrency\": " << options.max_concurrent_games
      << ", \"inferenceMaxBatchSize\": " << options.inference_max_batch_size << "},\n";
  out << "  \"inference\": {\"requestCount\": " << inference_stats.request_count
      << ", \"sessionRunCount\": " << inference_stats.session_run_count
      << ", \"meanBatchSize\": " << inference_stats.mean_batch_size
      << ", \"maxObservedBatchSize\": " << inference_stats.max_observed_batch_size
      << ", \"batchSizeHistogram\": {";
  bool first_bucket = true;
  for (const auto& [batch_size, count] : inference_stats.batch_size_histogram) {
    if (!first_bucket) {
      out << ", ";
    }
    first_bucket = false;
    out << "\"" << batch_size << "\": " << count;
  }
  out << "}},\n";
  out << "  \"playerCount\": 5,\n";
  out << "  \"cardCount\": 53,\n";
  out << "  \"cardIds\": " << card_ids << ",\n";
  out << "  \"cardIdsSha256\": \"" << sha256_hex(card_ids) << "\",\n";
  out << "  \"shards\": [\n";
  for (std::size_t index = 0; index < shards.size(); ++index) {
    const ShardManifest& shard = shards[index];
    out << "    {\"file\": \"" << shard.file << "\", \"startSeed\": " << shard.start_seed
        << ", \"endSeed\": " << shard.end_seed
        << ", \"gameCount\": " << shard.game_count
        << ", \"sampleCount\": " << shard.sample_count
        << ", \"byteLength\": " << shard.byte_length
        << ", \"sha256\": \"" << shard.sha256 << "\"}";
    out << (index + 1 == shards.size() ? "\n" : ",\n");
  }
  out << "  ],\n";
  out << "  \"playingEncoderSchemaVersion\": "
      << napoleon::observation::playing_encoder_schema_version(options.playing_observation_variant)
      << ",\n";
  out << "  \"playingModelInputSchemaVersion\": "
      << napoleon::observation::playing_model_input_schema_version(options.playing_observation_variant)
      << ",\n";
  out << "  \"playingObservationVariant\": ";
  json_escape(out, observation_variant_id);
  out << ",\n";
  out << "  \"modelInputFeatureCount\": " << model_input_feature_count << ",\n";
  out << "  \"behaviorPolicy\": {\n";
  out << "    \"type\": \"playing-onnx\",\n";
  out << "    \"artifactId\": ";
  json_escape(out, options.policy_artifact_id);
  out << ",\n";
  out << "    \"onnxFileName\": ";
  json_escape(out, base_name(options.policy_onnx_path));
  out << ",\n";
  out << "    \"metadataFileName\": ";
  json_escape(out, base_name(options.policy_metadata_path));
  out << ",\n";
  out << "    \"onnxSha256\": \"" << policy_onnx_sha256 << "\",\n";
  out << "    \"metadataSha256\": \"" << policy_metadata_sha256 << "\",\n";
  out << "    \"requestedInferenceDevice\": ";
  json_escape(out, options.inference_device);
  out << ",\n";
  out << "    \"resolvedInferenceDevice\": ";
  json_escape(out, resolved_inference_device(options));
  out << ",\n";
  out << "    \"executionProvider\": ";
  json_escape(out, resolved_inference_device(options));
  out << ",\n";
  out << "    \"metadata\": " << policy_metadata_json << "\n";
  out << "  },\n";
  out << "  \"policyArtifacts\": {\"current\": {\"artifactId\": ";
  json_escape(out, options.policy_artifact_id);
  out << ", \"onnxFileName\": ";
  json_escape(out, base_name(options.policy_onnx_path));
  out << ", \"metadataFileName\": ";
  json_escape(out, base_name(options.policy_metadata_path));
  out << ", \"onnxSha256\": \"" << policy_onnx_sha256
      << "\", \"metadataSha256\": \"" << policy_metadata_sha256
      << "\"}, \"frozen\": {\"artifactId\": ";
  json_escape(out, options.frozen_artifact_id);
  out << ", \"onnxFileName\": ";
  json_escape(out, base_name(options.frozen_onnx_path));
  out << ", \"metadataFileName\": ";
  json_escape(out, base_name(options.frozen_metadata_path));
  out << ", \"onnxSha256\": \"" << frozen_onnx_sha256
      << "\", \"metadataSha256\": \"" << frozen_metadata_sha256 << "\"}},\n";
  out << "  \"provenance\": {\"currentArtifactId\": ";
  json_escape(out, options.policy_artifact_id);
  out << ", \"currentOnnxSha256\": \"" << policy_onnx_sha256
      << "\", \"currentMetadataSha256\": \"" << policy_metadata_sha256
      << "\", \"frozenArtifactId\": ";
  json_escape(out, options.frozen_artifact_id);
  out << ", \"frozenOnnxSha256\": \"" << frozen_onnx_sha256
      << "\", \"frozenMetadataSha256\": \"" << frozen_metadata_sha256
      << "\", \"playingObservationVariant\": ";
  json_escape(out, observation_variant_id);
  out << ", \"modelInputFeatureCount\": " << model_input_feature_count
      << ", \"frozenPlayingObservationVariant\": \"public\", "
      << "\"frozenModelInputFeatureCount\": " << napoleon::observation::kPlayingModelInputFeatureCount
      << ", \"behaviorSamples\": \"current-policy-only\", \"rawCacheCompatible\": true, "
      << "\"rosterSpec\": {\"kind\": \"current-plus-opponent-pool\", "
      << "\"currentSeatRotation\": \"game-index-mod-player-count\", "
      << "\"opponentPool\": [\"rule-based\", \"frozen-onnx\"]}},\n";
  out << "  \"opponentPool\": {\"weighted\": ["
      << "{\"source\": \"rule-based\", \"weight\": 1}, "
      << "{\"source\": \"frozen-onnx\", \"artifactId\": ";
  json_escape(out, options.frozen_artifact_id);
  out << ", \"weight\": 1}]},\n";
  out << "  \"seatRotation\": {\"current\": \"game-index-mod-player-count\", "
      << "\"rosterSeed\": " << options.roster_seed << "},\n";
  out << "  \"samplingAlgorithm\": \"masked-categorical\",\n";
  out << "  \"temperature\": " << options.temperature << ",\n";
  out << "  \"reward\": {\"type\": \"terminal-team-win\", \"version\": " << kRewardVersion << "},\n";
  out << "  \"nonPlayingAgent\": {\"type\": \"rule-based\", \"version\": "
      << kRuleBasedAgentVersion << "},\n";
  out << "  \"rolloutRoster\": {\"assignment\": \"rotate-by-seed\", \"seats\": [";
  if (options.all_current) {
    for (int index = 0; index < napoleon::kPlayerCount; ++index) {
      if (index != 0) {
        out << ',';
      }
      write_roster_seat_manifest(
          out, "current-policy", options, frozen_onnx_sha256, frozen_metadata_sha256, frozen_metadata_json);
    }
  } else {
    const std::array<std::string, napoleon::kPlayerCount> seat_sources{
        "current-policy", "rule-based", "frozen-onnx", "rule-based", "frozen-onnx"};
    for (std::size_t index = 0; index < seat_sources.size(); ++index) {
      if (index != 0) {
        out << ',';
      }
      write_roster_seat_manifest(
          out, seat_sources[index], options, frozen_onnx_sha256, frozen_metadata_sha256, frozen_metadata_json);
    }
  }
  out << "]},\n";
  out << "  \"tensorSchema\": {\"shardSchemaVersion\": 1, \"byteOrder\": \"little-endian\", "
      << "\"compression\": \"none\", \"fields\": [";
  out << "{\"name\":\"modelInput\",\"dtype\":\"float32\",\"shape\":[" << model_input_feature_count << "]},";
  out << "{\"name\":\"legalPlayMask\",\"dtype\":\"uint8\",\"shape\":[53]},";
  out << "{\"name\":\"selectedCardIndex\",\"dtype\":\"uint8\",\"shape\":[]},";
  out << "{\"name\":\"behaviorLogProbability\",\"dtype\":\"float32\",\"shape\":[]},";
  out << "{\"name\":\"terminalReward\",\"dtype\":\"int8\",\"shape\":[]},";
  out << "{\"name\":\"seed\",\"dtype\":\"uint32\",\"shape\":[]},";
  out << "{\"name\":\"step\",\"dtype\":\"uint16\",\"shape\":[]},";
  out << "{\"name\":\"actingPlayerIndex\",\"dtype\":\"uint8\",\"shape\":[]},";
  out << "{\"name\":\"selfRoleIndex\",\"dtype\":\"uint8\",\"shape\":[]}";
  out << "]}\n";
  out << "}\n";
  out.close();
  if (!out.good()) {
    throw std::runtime_error("failed writing manifest.json");
  }
}

RosterSpec create_roster_spec(const CliOptions& options) {
  if (options.all_current) {
    return napoleon::self_play_roster(napoleon::current_policy_agent());
  }
  return napoleon::current_plus_opponent_pool_roster(
      napoleon::current_policy_agent(),
      {
          napoleon::WeightedAgent{napoleon::rule_based_agent(), 1},
          napoleon::WeightedAgent{napoleon::frozen_policy_agent(options.frozen_artifact_id), 1},
      },
      true,
      0);
}

void generate_dataset(const CliOptions& options) {
  const std::filesystem::path output = std::filesystem::absolute(options.output_directory);
  const std::string policy_metadata_json = read_text_file(options.policy_metadata_path);
  const std::string frozen_metadata_json = read_text_file(options.frozen_metadata_path);
  validate_policy_metadata(
      "current policy",
      policy_metadata_json,
      options.playing_observation_variant);
  validate_policy_metadata(
      "frozen policy",
      frozen_metadata_json,
      napoleon::observation::PlayingObservationVariant::Public);

  if (std::filesystem::exists(output)) {
    throw std::runtime_error("output directory already exists: " + output.string());
  }
  std::filesystem::create_directories(output.parent_path());
  const std::filesystem::path temp =
      output.parent_path() /
      ("." + output.filename().string() + ".tmp-" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directories(temp);

  const RosterSpec roster_spec = create_roster_spec(options);
  auto policy_executor = create_policy_executor(options);
  std::vector<ShardManifest> shards;
  std::uint64_t total_samples = 0;

  try {
    std::filesystem::create_directories(output);
    std::optional<BinaryShardWriter> writer;
    std::uint32_t shard_index = 0;
    std::uint32_t shard_start_seed = options.start_seed;
    std::uint32_t shard_game_count = 0;

    const auto build_payload =
        [variant = options.playing_observation_variant](
            const GameState& state,
            int player_index,
            napoleon::AgentRequest& request) {
          const napoleon::observation::PlayingObservationVariant request_variant =
              request.agent.type == AgentType::CurrentPolicy
                  ? variant
                  : napoleon::observation::PlayingObservationVariant::Public;
          napoleon::onnx_policy::attach_playing_model_input(
              state,
              player_index,
              request_variant,
              request);
        };

    napoleon::SimulationRuntime runtime(napoleon::SimulationRuntimeConfig{
        roster_spec,
        options.start_seed,
        options.roster_seed,
        std::max<std::size_t>(1, options.max_concurrent_games),
        build_payload});
    std::vector<std::vector<TensorSample>> samples_by_game(options.game_count);
    std::vector<bool> completed(options.game_count, false);
    std::uint32_t next_game_to_add = 0;
    std::uint32_t next_game_to_write = 0;

    while (next_game_to_write < options.game_count) {
      const std::size_t active_count = runtime.active_game_count();
      const std::size_t open_slots =
          options.max_concurrent_games > active_count ? options.max_concurrent_games - active_count : 0;
      if (open_slots > 0 && next_game_to_add < options.game_count) {
        const std::size_t add_count = std::min<std::size_t>(
            open_slots,
            options.game_count - next_game_to_add);
        runtime.add_games(add_count);
        next_game_to_add += static_cast<std::uint32_t>(add_count);
      }

      runtime.advance_runnable_games();
      const std::vector<napoleon::AgentRequest> requests = runtime.collect_agent_requests();
      if (!requests.empty()) {
        submit_dataset_policy_requests(
            runtime,
            *policy_executor,
            requests,
            samples_by_game,
            options.start_seed,
            options.playing_observation_variant,
            std::max<std::size_t>(1, options.inference_max_batch_size));
      }

      for (const napoleon::FinishedGame& finished : runtime.collect_finished_games()) {
        if (finished.game_index >= options.game_count) {
          throw std::runtime_error("finished game index out of range");
        }
        std::vector<TensorSample>& samples = samples_by_game[finished.game_index];
        if (samples.empty()) {
          throw std::runtime_error(
              "current-policy roster produced no samples for seed " +
              std::to_string(finished.seed));
        }
        for (TensorSample& sample : samples) {
          sample.terminal_reward = static_cast<std::int8_t>(
              terminal_reward_for_role(finished.result, sample.self_role_index));
        }
        completed[finished.game_index] = true;
      }

      while (next_game_to_write < options.game_count && completed[next_game_to_write]) {
        const std::uint32_t seed = options.start_seed + next_game_to_write;
        if (!writer.has_value()) {
          shard_start_seed = seed;
          writer.emplace(
              output,
              temp,
              shard_index,
              shard_start_seed,
              options.playing_observation_variant);
          shard_game_count = 0;
        }
        for (const TensorSample& sample : samples_by_game[next_game_to_write]) {
          writer->write_sample(sample);
        }
        total_samples += samples_by_game[next_game_to_write].size();
        samples_by_game[next_game_to_write].clear();
        shard_game_count += 1;

        if (shard_game_count == options.games_per_shard ||
            next_game_to_write + 1u == options.game_count) {
          shards.push_back(writer->close(seed, shard_game_count));
          writer.reset();
          shard_index += 1;
        }
        next_game_to_write += 1;
      }
    }

    write_manifest(output, options, shards, total_samples, policy_executor->stats());
    std::filesystem::remove_all(temp);
  } catch (...) {
    std::filesystem::remove_all(temp);
    std::filesystem::remove_all(output);
    throw;
  }

  const napoleon::onnx_policy::BatchedPolicyStats inference_stats = policy_executor->stats();
  std::cout << "{\"outputDirectory\":" << json_string(output.string())
            << ",\"gameCount\":" << options.game_count
            << ",\"sampleCount\":" << total_samples
            << ",\"shardCount\":" << shards.size()
            << ",\"format\":\"playing-self-play-binary-v1\""
            << ",\"playingObservationVariant\":"
            << json_string(napoleon::observation::playing_observation_variant_id(
                   options.playing_observation_variant))
            << ",\"modelInputFeatureCount\":"
            << napoleon::observation::playing_model_input_feature_count(
                   options.playing_observation_variant)
            << ",\"requestedInferenceDevice\":" << json_string(options.inference_device)
            << ",\"resolvedInferenceDevice\":" << json_string(resolved_inference_device(options))
            << ",\"executionProvider\":" << json_string(resolved_inference_device(options))
            << ",\"policyBackend\":" << json_string(options.policy_backend)
            << ",\"rolloutConcurrency\":" << options.max_concurrent_games
            << ",\"inferenceMaxBatchSize\":" << options.inference_max_batch_size
            << ",\"inference\":{\"requestCount\":" << inference_stats.request_count
            << ",\"sessionRunCount\":" << inference_stats.session_run_count
            << ",\"meanBatchSize\":" << inference_stats.mean_batch_size
            << ",\"maxObservedBatchSize\":" << inference_stats.max_observed_batch_size
            << ",\"batchSizeHistogram\":{";
  bool first_bucket = true;
  for (const auto& [batch_size, count] : inference_stats.batch_size_histogram) {
    if (!first_bucket) {
      std::cout << ',';
    }
    first_bucket = false;
    std::cout << "\"" << batch_size << "\":" << count;
  }
  std::cout << "}}}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    generate_dataset(parse_args(argc, argv));
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
