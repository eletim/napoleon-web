#include "napoleon_core.hpp"
#include "napoleon_observation.hpp"
#include "napoleon_roster.hpp"
#include "napoleon_rule_based.hpp"

#include <array>
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
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
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
constexpr int kModelInputFeatureCount = napoleon::observation::kPlayingModelInputFeatureCount;
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
  double temperature = 1.0;
  bool all_current = false;
};

struct TensorSample {
  std::array<float, kModelInputFeatureCount> model_input{};
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
      std::uint32_t start_seed)
      : output_directory_(output_directory),
        temp_directory_(temp_root / ("shard-" + std::to_string(shard_index))),
        file_name_(shard_file_name(shard_index)),
        start_seed_(start_seed) {
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
    static constexpr std::array<const char*, 9> shapes{
        "[6246]", "[53]", "[]", "[]", "[]", "[]", "[]", "[]", "[]"};

    std::ostringstream out;
    std::uint64_t offset = 0;
    out << "{\"shardSchemaVersion\":" << kShardSchemaVersion
        << ",\"sampleType\":\"playing-self-play-sample\""
        << ",\"sampleSchemaVersion\":" << kSampleSchemaVersion
        << ",\"sampleCount\":" << sample_count_
        << ",\"modelInputFeatureCount\":" << kModelInputFeatureCount
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
          "[--roster-seed <uint32>] [--temperature <positive>] [--all-current]");
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
  if (options.start_seed > std::numeric_limits<std::uint32_t>::max() - options.game_count + 1u) {
    throw std::runtime_error("start-seed + games exceeds uint32 range");
  }
  return options;
}

int card_model_index(Card card) {
  if (card.id == 52) {
    return 52;
  }
  const int suit_index = static_cast<int>(card.id / 13);
  const int rank_id = static_cast<int>(card.id % 13);
  static constexpr std::array<int, 13> rank_to_model{0, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  return suit_index * 13 + rank_to_model[static_cast<std::size_t>(rank_id)];
}

std::vector<Action> legal_actions_with_exchange(const GameState& state, int player_index) {
  std::vector<Action> actions = napoleon::get_legal_actions(state, player_index);
  if (!actions.empty() || state.phase != napoleon::Phase::Exchanging ||
      state.current_player_index != player_index || state.is_game_over) {
    return actions;
  }

  const auto& hand = state.hands[static_cast<std::size_t>(player_index)];
  if (hand.size() < 3) {
    return actions;
  }

  for (std::size_t first = 0; first + 2 < hand.size(); ++first) {
    for (std::size_t second = first + 1; second + 1 < hand.size(); ++second) {
      for (std::size_t third = second + 1; third < hand.size(); ++third) {
        Action action;
        action.type = Action::Type::DiscardCards;
        action.player_index = player_index;
        action.cards = {hand[first], hand[second], hand[third]};
        actions.push_back(action);
      }
    }
  }
  return actions;
}

Action select_non_current_action(
    const AgentIdentity& agent,
    const GameState& state,
    int player_index,
    std::uint32_t seed,
    std::uint32_t step) {
  if (state.phase == napoleon::Phase::Playing && !state.is_trick_complete) {
    napoleon::SeededRandom rng(seed ^ (step + 0x9e3779b9u));
    if (agent.type == AgentType::RuleBased || agent.type == AgentType::FrozenPolicy) {
      return napoleon::select_rule_based_action(state, player_index, rng);
    }
  }

  std::vector<Action> actions = legal_actions_with_exchange(state, player_index);
  if (actions.empty()) {
    throw std::runtime_error("no legal action available");
  }
  return actions.front();
}

int self_role_index(const napoleon::observation::PlayingModelInput& input) {
  for (int index = 0; index < kSelfRoleCount; ++index) {
    if (input.observation.self_role_one_hot[static_cast<std::size_t>(index)] == 1) {
      return index;
    }
  }
  throw std::runtime_error("playing model input has no self role");
}

TensorSample create_current_policy_sample(
    const GameState& state,
    const Action& selected_action,
    std::uint32_t seed,
    std::uint32_t step) {
  const auto input = napoleon::observation::create_playing_model_input(
      state, state.current_player_index);
  const int selected_index = card_model_index(selected_action.card);
  if (selected_index < 0 || selected_index >= kCardCount) {
    throw std::runtime_error("selected card index out of range");
  }
  if (input.legal_play_mask[static_cast<std::size_t>(selected_index)] != 1) {
    throw std::runtime_error("selected current-policy card is not legal");
  }

  int legal_count = 0;
  TensorSample sample;
  for (std::size_t index = 0; index < input.legal_play_mask.size(); ++index) {
    sample.legal_play_mask[index] =
        static_cast<std::uint8_t>(input.legal_play_mask[index] == 0 ? 0 : 1);
    legal_count += sample.legal_play_mask[index] == 1 ? 1 : 0;
  }
  if (legal_count <= 0) {
    throw std::runtime_error("current-policy playing decision has no legal cards");
  }

  sample.model_input = input.model_input;
  sample.selected_card_index = static_cast<std::uint8_t>(selected_index);
  sample.behavior_log_probability =
      legal_count == 1 ? 0.0F : static_cast<float>(-std::log(static_cast<double>(legal_count)));
  sample.seed = seed;
  sample.step = static_cast<std::uint16_t>(step);
  sample.acting_player_index = static_cast<std::uint8_t>(state.current_player_index);
  sample.self_role_index = static_cast<std::uint8_t>(self_role_index(input));
  return sample;
}

int terminal_reward_for_role(const GameResult& result, std::uint8_t role_index) {
  const std::string acting_team =
      role_index == 2 ? "alliance" : "napoleon-team";
  return acting_team == result.winner ? 1 : -1;
}

std::vector<TensorSample> run_game_samples(
    std::uint32_t seed,
    const RosterAssignment& roster) {
  GameState state = napoleon::create_initial_game(seed);
  std::uint32_t step = 0;
  std::vector<TensorSample> samples;
  samples.reserve(16);

  while (!state.is_game_over && state.phase != napoleon::Phase::Finished) {
    if (state.is_trick_complete) {
      Action advance;
      advance.type = Action::Type::AdvanceToNextTrick;
      napoleon::apply_action(state, advance);
      continue;
    }

    const int player_index = state.current_player_index;
    const AgentIdentity& agent = roster.agents[static_cast<std::size_t>(player_index)];
    std::vector<Action> actions = legal_actions_with_exchange(state, player_index);
    if (actions.empty()) {
      throw std::runtime_error("game reached a decision state with no legal actions");
    }

    Action action;
    const std::uint32_t next_step = step + 1u;
    if (agent.type == AgentType::CurrentPolicy && state.phase == napoleon::Phase::Playing) {
      action = actions.front();
      samples.push_back(create_current_policy_sample(state, action, seed, next_step));
    } else {
      action = select_non_current_action(agent, state, player_index, seed, next_step);
    }

    napoleon::apply_action(state, action);
    step = next_step;
    if (step > std::numeric_limits<std::uint16_t>::max()) {
      throw std::runtime_error("game step exceeds uint16 dataset field range");
    }
  }

  if (!state.result.has_value()) {
    throw std::runtime_error("finished game has no result");
  }
  for (TensorSample& sample : samples) {
    sample.terminal_reward =
        static_cast<std::int8_t>(terminal_reward_for_role(*state.result, sample.self_role_index));
  }
  return samples;
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

void write_roster_seat_manifest(
    std::ostream& out,
    const std::string& source,
    const CliOptions& options,
    const std::string& frozen_onnx_sha256,
    const std::string& frozen_metadata_sha256) {
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
        << "\"requestedInferenceDevice\":\"cpu\","
        << "\"resolvedInferenceDevice\":\"cpu\","
        << "\"executionProvider\":\"cpu\","
        << "\"metadata\":{\"metadataSchemaVersion\":1,\"producer\":\"cpp-rl-dataset-smoke\"}}";
  } else {
    throw std::runtime_error("invalid roster source");
  }
}

void write_manifest(
    const std::filesystem::path& output_directory,
    const CliOptions& options,
    const std::vector<ShardManifest>& shards,
    std::uint64_t sample_count) {
  const std::string card_ids = card_ids_json();
  const std::string policy_onnx_sha256 = sha256_file(options.policy_onnx_path);
  const std::string policy_metadata_sha256 = sha256_file(options.policy_metadata_path);
  const std::string frozen_onnx_sha256 = sha256_file(options.frozen_onnx_path);
  const std::string frozen_metadata_sha256 = sha256_file(options.frozen_metadata_path);
  const std::uint32_t end_seed = options.start_seed + options.game_count - 1u;
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
  out << "  \"playingEncoderSchemaVersion\": " << napoleon::observation::kPlayingEncoderSchemaVersion
      << ",\n";
  out << "  \"playingModelInputSchemaVersion\": "
      << napoleon::observation::kPlayingModelInputSchemaVersion << ",\n";
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
  out << "    \"requestedInferenceDevice\": \"cpu\",\n";
  out << "    \"resolvedInferenceDevice\": \"cpu\",\n";
  out << "    \"executionProvider\": \"cpu\",\n";
  out << "    \"metadata\": {\"metadataSchemaVersion\": 1, \"producer\": \"cpp-rl-dataset-cli\", "
      << "\"sampleAttribution\": \"current-policy-only\", \"rawCacheCompatible\": true, "
      << "\"rosterSpec\": {\"kind\": \"current-plus-opponent-pool\", "
      << "\"currentSeatRotation\": \"game-index-mod-player-count\", "
      << "\"opponentPool\": [\"rule-based\", \"frozen-onnx\"]}}\n";
  out << "  },\n";
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
      write_roster_seat_manifest(out, "current-policy", options, frozen_onnx_sha256, frozen_metadata_sha256);
    }
  } else {
    const std::array<std::string, napoleon::kPlayerCount> seat_sources{
        "current-policy", "rule-based", "frozen-onnx", "rule-based", "frozen-onnx"};
    for (std::size_t index = 0; index < seat_sources.size(); ++index) {
      if (index != 0) {
        out << ',';
      }
      write_roster_seat_manifest(out, seat_sources[index], options, frozen_onnx_sha256, frozen_metadata_sha256);
    }
  }
  out << "]},\n";
  out << "  \"tensorSchema\": {\"shardSchemaVersion\": 1, \"byteOrder\": \"little-endian\", "
      << "\"compression\": \"none\", \"fields\": [";
  out << "{\"name\":\"modelInput\",\"dtype\":\"float32\",\"shape\":[6246]},";
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
          napoleon::WeightedAgent{napoleon::frozen_policy_agent("rl-v740"), 1},
      },
      true,
      0);
}

void generate_dataset(const CliOptions& options) {
  const std::filesystem::path output = std::filesystem::absolute(options.output_directory);
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
  std::vector<ShardManifest> shards;
  std::uint64_t total_samples = 0;

  try {
    std::filesystem::create_directories(output);
    std::optional<BinaryShardWriter> writer;
    std::uint32_t shard_index = 0;
    std::uint32_t shard_start_seed = options.start_seed;
    std::uint32_t shard_game_count = 0;

    for (std::uint32_t game_offset = 0; game_offset < options.game_count; ++game_offset) {
      const std::uint32_t seed = options.start_seed + game_offset;
      if (!writer.has_value()) {
        shard_start_seed = seed;
        writer.emplace(output, temp, shard_index, shard_start_seed);
        shard_game_count = 0;
      }

      const RosterAssignment roster =
          napoleon::sample_roster(roster_spec, options.roster_seed, game_offset);
      const std::vector<TensorSample> samples = run_game_samples(seed, roster);
      if (samples.empty()) {
        throw std::runtime_error("current-policy roster produced no samples for seed " + std::to_string(seed));
      }
      for (const TensorSample& sample : samples) {
        writer->write_sample(sample);
      }
      total_samples += samples.size();
      shard_game_count += 1;

      if (shard_game_count == options.games_per_shard ||
          game_offset + 1u == options.game_count) {
        shards.push_back(writer->close(seed, shard_game_count));
        writer.reset();
        shard_index += 1;
      }
    }

    write_manifest(output, options, shards, total_samples);
    std::filesystem::remove_all(temp);
  } catch (...) {
    std::filesystem::remove_all(temp);
    std::filesystem::remove_all(output);
    throw;
  }

  std::cout << "{\"outputDirectory\":" << json_string(output.string())
            << ",\"gameCount\":" << options.game_count
            << ",\"sampleCount\":" << total_samples
            << ",\"shardCount\":" << shards.size()
            << ",\"format\":\"playing-self-play-binary-v1\"}\n";
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
